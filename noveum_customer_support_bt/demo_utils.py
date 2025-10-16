import json
import os
import re
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

# NovaEval imports
from novaeval.agents.agent_data import AgentData, ToolSchema, ToolCall, ToolResult
from novaeval.datasets.agent_dataset import AgentDataset
from novaeval.evaluators.agent_evaluator import AgentEvaluator
from novaeval.models.gemini import GeminiModel
from novaeval.scorers.agent_scorers import (
    context_relevancy_scorer,
    role_adherence_scorer,
    task_progression_scorer,
    tool_relevancy_scorer,
    parameter_correctness_scorer
)
from dotenv import load_dotenv

load_dotenv()

print("✅ All imports successful!")

def list_dataset_files(directory: str = "split_datasets") -> List[str]:
    """
    List available JSON dataset files in the specified directory.
    
    Args:
        directory: Directory to search for dataset files
        
    Returns:
        List of JSON file names
    """
    try:
        if not os.path.exists(directory):
            print(f"❌ Directory {directory} does not exist")
            return []
            
        files = os.listdir(directory)
        json_files = [f for f in files if f.endswith('.json')]
        
        print(f"📊 Found {len(json_files)} JSON files in {directory}/:")
        for file in json_files:
            print(f"  - {file}")
            
        return json_files
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []

print("✅ list_dataset_files function defined!")

def load_and_analyze_dataset(file_name: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Load JSON dataset file and analyze span types.
    
    Args:
        file_name: Path to the JSON dataset file
        
    Returns:
        Tuple of (spans_data, span_types_dict)
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            spans_data = json.load(f)
            
        print(f"📊 Loaded {len(spans_data)} spans from {file_name}")
        print("\n🔍 Available span types:")
        
        # Analyze span types
        span_types = {}
        for span in spans_data:
            span_name = span.get('name', 'unknown')
            if span_name not in span_types:
                span_types[span_name] = 0
            span_types[span_name] += 1
            
        for span_type, count in span_types.items():
            print(f"  - {span_type}: {count}")
            
        return spans_data, span_types
        
    except FileNotFoundError:
        print(f"❌ File {file_name} not found")
        return [], {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return [], {}
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return [], {}

print("✅ load_and_analyze_dataset function defined!")

def parse_tools_from_prompt(prompt: str) -> List[ToolSchema]:
    """
    Parse tool definitions from LLM prompts using regex.
    
    Expected format: tool_name(param: type = default) -> return_type - description
    """
    # Pattern to match tool signatures
    pattern = r'(\w+)\(([^)]*)\)\s*->\s*(\w+)\s*-\s*(.+?)(?=\n\w+\(|$)'
    matches = re.findall(pattern, prompt, re.DOTALL)
    
    tools = []
    for match in matches:
        tool_name, params_str, return_type, description = match
        
        # Parse parameters
        args_schema = parse_params(params_str)
        
        tool = ToolSchema(
            name=tool_name,
            description=description.strip(),
            args_schema=args_schema,
            return_schema={"type": return_type}
        )
        tools.append(tool)
    
    return tools

print("✅ parse_tools_from_prompt function defined!")

def parse_params(params_str: str) -> Dict[str, Any]:
    """
    Parse parameter string into schema dictionary.
    
    Format: param_name: type = default_value
    """
    if not params_str.strip():
        return {}
    
    # Split parameters by comma
    params = [p.strip() for p in params_str.split(',') if p.strip()]
    schema = {}
    
    for param in params:
        if ':' in param:
            parts = param.split(':', 1)
            param_name = parts[0].strip()
            type_and_default = parts[1].strip()
            
            # Extract type and default value
            if '=' in type_and_default:
                type_part, default_part = type_and_default.split('=', 1)
                param_type = type_part.strip()
                default_val = default_part.strip().strip('"\'')
                schema[param_name] = {'type': param_type, 'default': default_val}
            else:
                param_type = type_and_default.strip()
                schema[param_name] = {'type': param_type}
    
    return schema

print("✅ parse_params function defined!")

def identify_span_type(span: Dict[str, Any]) -> str:
    """
    Identify span type based on attributes.
    """
    attributes = span.get('attributes', {})
    span_name = span.get('name', '')
    
    # Check for agent spans - expanded to include RAG evaluation spans
    agent_span_names = [
        'reddit_agent_run_1', 'reddit_agent_run_2', 
        'agent.query_generation', 'agent.comment_generation',
        'agent.rag_evaluation_metrics', 'agent.web_search_evaluation_metrics',
        'agent.query_routing'
    ]
    
    # Check for agent attributes or known agent span names
    if (any('chain.name' == key for key in attributes.keys()) or 
        span_name in agent_span_names or
        span_name.startswith('agent.')):
        return 'agent'
    
    # Check for LLM attributes
    if any('llm.model' == key for key in attributes.keys()):
        return 'llm'
    
    # Check for tool attributes
    tool_span_names = ['post_validation', 'email_generation_and_sending']
    if (any('tool.name' == key for key in attributes.keys()) or 
        span_name in tool_span_names):
        return 'tool'
    
    print('returning unknown type for span')
    print(span)
    return 'unknown'

print("✅ identify_span_type function defined!")

def map_span_to_agent_data(span: Dict[str, Any], count_unknowns: Optional[Dict[str, int]] = None) -> AgentData:
    """
    Map a single span from file_name to AgentData format.
    """

    attributes = span.get('attributes', {})
    events = span.get('events', [])
    span_type = identify_span_type(span)

    # Base mappings
    data = {
        'user_id': span.get('metadata', {}).get('user_id', None),
        'task_id': span.get('trace_id'),
        'turn_id': span.get('span_id'),
        'ground_truth': None,
        'expected_tool_call': None,
        'agent_name': span_type,
        'agent_role': span_type,
        'system_prompt': "You are a helpful customer support agent",
        'metadata': None,
        'exit_status': span.get('status'),
        'tools_available': [],
        'tool_calls': [],
        'parameters_passed': {},
        'tool_call_results': [],
        'retrieval_query': None,
        'retrieved_context': None,
        'agent_exit': False,
        'trace': None
    }

    # Span-specific mappings
    if span_type == 'agent':
        # Agent task - handle different span types
        span_name = span.get('name', '')
        
        if span_name.startswith('agent.rag_evaluation_metrics'):
            # RAG evaluation spans
            data['agent_task'] = attributes.get('input_query', 'RAG evaluation task')
            data['agent_response'] = attributes.get('output_response', '')
            data['retrieval_query'] = [attributes.get('input_query', '')]
            data['retrieved_context'] = [[attributes.get('retrieval.context_retrieved', '')]]
        elif span_name.startswith('agent.web_search_evaluation_metrics'):
            # Web search evaluation spans
            data['agent_task'] = attributes.get('input_query', 'Web search evaluation task')
            data['agent_response'] = attributes.get('output_response', '')
            data['retrieval_query'] = [attributes.get('input_query', '')]
            data['retrieved_context'] = [[attributes.get('web_search.search_results', '')]]
        elif span_name.startswith('agent.query_routing'):
            # Query routing spans
            data['agent_task'] = attributes.get('input_query', 'Query routing task')
            data['agent_response'] = attributes.get('routing_decision', '')
        else:
            # Standard agent spans
            chain_inputs = attributes.get('chain.inputs', {})
            if isinstance(chain_inputs, dict) and 'input' in chain_inputs:
                data['agent_task'] = chain_inputs['input']
            elif attributes.get("agent_task"):
                data['agent_task'] = attributes.get("agent_task")
            else:
                print('agent_task not found')
            
            # Agent response
            finish_values = attributes.get('agent.output.finish.return_values', {})
            if isinstance(finish_values, dict) and 'output' in finish_values:
                data['agent_response'] = finish_values['output']
            elif attributes.get("agent_response"):
                data['agent_response'] = attributes.get("agent_response")
            else:
                print("agent_response is not available  " + span['span_id'])
        # Tool calls from agent actions - handle different span types
        if span_name.startswith('agent.rag_evaluation_metrics'):
            # RAG evaluation spans don't have traditional tool calls
            # They have retrieval and response evaluation capabilities
            data['tools_available'] = [
                ToolSchema(
                    name="rag_evaluation",
                    description="Evaluates RAG system performance",
                    args_schema={},
                    return_schema={"type": "evaluation_result"}
                )
            ]
        elif span_name.startswith('agent.web_search_evaluation_metrics'):
            # Web search evaluation spans
            data['tools_available'] = [
                ToolSchema(
                    name="web_search_evaluation",
                    description="Evaluates web search performance",
                    args_schema={},
                    return_schema={"type": "evaluation_result"}
                )
            ]
        elif span_name.startswith('agent.query_routing'):
            # Query routing spans
            data['tools_available'] = [
                ToolSchema(
                    name="query_routing",
                    description="Routes queries to appropriate handlers",
                    args_schema={},
                    return_schema={"type": "routing_decision"}
                )
            ]
        else:
            # Standard agent tool calls
            tool_name = attributes.get('agent.output.action.tool')
            tool_input = attributes.get('agent.output.action.tool_input')
            
            if tool_name:
                tool_call = ToolCall(
                    tool_name=tool_name,
                    parameters={'input': tool_input} if tool_input else {},
                    call_id=span['span_id']
                )
                data['tool_calls'] = [tool_call]
                data['parameters_passed'] = {'input': tool_input} if tool_input else {}
                
                # Handle retrieval query for langchain_retriever
                if tool_name == 'langchain_retriever' and tool_input:
                    data['retrieval_query'] = [tool_input]
        
        # Agent exit status
        data['agent_exit'] = any(event.get('name') == 'agent_finish' for event in events)
        
        # Trace (dump events as JSON)
        if events:
            data['trace'] = json.dumps(events)
    
    elif span_type == 'llm':
        # Agent response from LLM output
        llm_input = attributes.get('llm.input.prompts', ['input is not available'])
        data['agent_task'] = llm_input[0]

        llm_responses = attributes.get('llm.output.response', [])
        if llm_responses:
            data['agent_response'] = llm_responses[0]
        else:
            print("llm_response is not available")
        # Parse tools from prompt
        prompts = attributes.get('llm.input.prompts', [])
        if prompts:
            try:
                tools = parse_tools_from_prompt(prompts[0])
                data['tools_available'] = tools
            except Exception:
                # Fallback to empty list if parsing fails
                data['tools_available'] = []
        
        data['parameters_passed'] = {}
    
    elif span_type == 'tool':
        # Agent response from tool output
        tool_output = attributes.get('tool.output.output')
        data['agent_task'] = f"This is a simple tool call, and the tool will execute as programmed. Its name is - {attributes.get('tool.name')}"
        if tool_output:
            data['agent_response'] = tool_output
        elif attributes.get("tool_response"):
            data['agent_response'] = attributes.get("tool_response")
        else:
            print("tool_output is not available " + span['span_id'])
        # Tool call results
        tool_name = attributes.get('tool.name')
        if tool_name and tool_output is not None:
            tool_result = ToolResult(
                call_id=span['span_id'],
                result=tool_output,
                success=span.get('status') == 'ok',
                error_message=None if span.get('status') == 'ok' else 'Tool execution failed'
            )
            data['tool_call_results'] = [tool_result]
            
            # Handle retrieved context for langchain_retriever
            if tool_name == 'langchain_retriever':
                data['retrieved_context'] = [[tool_output]]
        
        # Parameters from tool input
        tool_input_keys = [key for key in attributes.keys() if key.startswith('tool.input.')]
        tool_params = {}
        for key in tool_input_keys:
            param_name = key.replace('tool.input.', '')
            tool_params[param_name] = attributes[key]
        data['parameters_passed'] = tool_params
    else:
        if count_unknowns is not None and 'count' in count_unknowns:
            count_unknowns['count'] += 1
            print('Spans with unknown type: ' + str(count_unknowns['count']))
        else:
            print('Spans with unknown type detected')
    return AgentData(**data)


print("✅ map_span_to_agent_data function defined!")

def convert_spans_to_agent_dataset(spans_data: List[Dict[str, Any]]) -> tuple[List[AgentData], List[str], AgentDataset]:
    """
    Convert list of spans to AgentData objects and create AgentDataset.
    
    Args:
        spans_data: List of span dictionaries
        
    Returns:
        Tuple of (agent_data_list, errors, dataset)
    """
    print("🔄 Converting spans to AgentData objects...")
    
    agent_data_list = []
    errors = []
    count_unknowns = {'count': 0}
    
    for i, span in enumerate(spans_data):
        try:
            agent_data = map_span_to_agent_data(span, count_unknowns)
            agent_data_list.append(agent_data)
        except Exception as e:
            errors.append(f"Span {i}: {str(e)}")
            if len(errors) <= 5:  # Show first 5 errors only
                print(f"⚠️  Error processing span {i}: {e}")
                
    print(f"\n✅ Successfully converted {len(agent_data_list)} spans to AgentData")
    if errors:
        print(f"❌ {len(errors)} spans had errors")
        
    # Create AgentDataset
    dataset = AgentDataset()
    dataset.data = agent_data_list
    
    print(f"📊 AgentDataset created with {len(dataset.data)} records")
    
    return agent_data_list, errors, dataset

print("✅ convert_spans_to_agent_dataset function defined!")

def analyze_dataset_statistics(dataset: AgentDataset) -> Dict[str, Any]:
    """
    Analyze dataset statistics including agent types, tool usage, and response counts.
    
    Args:
        dataset: AgentDataset to analyze
        
    Returns:
        Dictionary containing statistics
    """
    print("📈 Dataset Statistics:")
    
    agent_types = {}
    tool_usage = {}
    with_responses = 0
    with_tool_calls = 0
    with_retrieval = 0
    
    for data in dataset.data:
        # Agent types
        if data.agent_name:
            agent_types[data.agent_name] = agent_types.get(data.agent_name, 0) + 1
        
        # Responses
        if data.agent_response:
            with_responses += 1
        
        # Tool calls
        if data.tool_calls:
            with_tool_calls += 1
            for tool_call in data.tool_calls:
                if hasattr(tool_call, 'tool_name'):
                    tool_usage[tool_call.tool_name] = tool_usage.get(tool_call.tool_name, 0) + 1
        
        # Retrieval
        if data.retrieval_query:
            with_retrieval += 1
            
    stats = {
        'agent_types': dict(agent_types),
        'records_with_responses': with_responses,
        'records_with_tool_calls': with_tool_calls,
        'records_with_retrieval': with_retrieval,
        'tool_usage': dict(tool_usage),
        'total_records': len(dataset.data)
    }
    
    print(f"\nAgent Types: {dict(agent_types)}")
    print(f"Records with responses: {with_responses}")
    print(f"Records with tool calls: {with_tool_calls}")
    print(f"Records with retrieval: {with_retrieval}")
    print(f"Tool usage: {dict(tool_usage)}")
    
    return stats

print("✅ analyze_dataset_statistics function defined!")

def setup_gemini_model(model_name: str = "gemini-2.0-flash-lite", temperature: float = 0.1, max_tokens: int = 1024) -> Optional[GeminiModel]:
    """
    Setup and initialize Gemini model with API key validation.
    
    Args:
        model_name: Gemini model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        GeminiModel instance or None if setup failed
    """
    # Check for API key
    if 'GEMINI_API_KEY' not in os.environ:
        print("⚠️  GEMINI_API_KEY environment variable not set!")
        print("Please set it before running evaluation:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return None
    else:
        print("✅ GEMINI_API_KEY found in environment")
        
    # Initialize Gemini model
    try:
        gemini_model = GeminiModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        print("✅ Gemini model initialized")
        return gemini_model
    except Exception as e:
        print(f"❌ Error initializing Gemini model: {e}")
        return None

print("✅ setup_gemini_model function defined!")

def setup_agent_evaluator(dataset: AgentDataset, gemini_model: GeminiModel, output_dir: str = "./demo_results", 
                         include_reasoning: bool = True, stream: bool = False) -> Optional[AgentEvaluator]:
    """
    Setup AgentEvaluator with scoring functions.
    
    Args:
        dataset: AgentDataset to evaluate
        gemini_model: Initialized GeminiModel
        output_dir: Directory for output files
        include_reasoning: Whether to include reasoning in results
        stream: Whether to stream results
        
    Returns:
        AgentEvaluator instance or None if setup failed
    """
    # Initialize scoring functions
    scoring_functions = [
        task_progression_scorer,
        context_relevancy_scorer,
        role_adherence_scorer,
        tool_relevancy_scorer,
        parameter_correctness_scorer
    ]
    
    print(f"✅ Initialized {len(scoring_functions)} scoring functions:")
    for func in scoring_functions:
        print(f"  - {func.__name__}")
        
    # Create AgentEvaluator
    try:
        evaluator = AgentEvaluator(
            agent_dataset=dataset,
            models=[gemini_model],
            scoring_functions=scoring_functions,
            output_dir=output_dir,
            stream=stream,
            include_reasoning=include_reasoning
        )
        print("\n✅ AgentEvaluator created with Gemini model and scoring functions")
        return evaluator
    except Exception as e:
        print(f"❌ Error creating AgentEvaluator: {e}")
        return None

print("✅ setup_agent_evaluator function defined!")

def run_evaluation(dataset: AgentDataset, evaluator: AgentEvaluator, sample_size: int = 25, 
                  file_name: str = "sample_evaluation") -> Optional[pd.DataFrame]:
    """
    Run agent evaluation on sample data and display results.
    
    Args:
        dataset: AgentDataset to evaluate
        evaluator: Initialized AgentEvaluator
        sample_size: Number of samples to evaluate
        file_name: Base name for output files
        
    Returns:
        DataFrame with results or None if evaluation failed
    """
    print("🚀 Running evaluation on sample data...")
    
    try:
        # Create a smaller dataset for demo purposes
        # Filter for records with agent responses or meaningful content
        sample_data = []
        for data in dataset.data:
            if (data.agent_response and data.agent_response.strip()) or data.agent_task:
                sample_data.append(data)
                if len(sample_data) >= sample_size:
                    break
        
        print(f"\n📊 Evaluating {len(sample_data)} sample records...")
        
        # Create a temporary dataset with just the sample data
        sample_dataset = AgentDataset()
        sample_dataset.data = sample_data
        
        # Create a new evaluator with the sample dataset
        sample_evaluator = AgentEvaluator(
            agent_dataset=sample_dataset,
            models=evaluator.models,
            scoring_functions=evaluator.scoring_functions,
            output_dir=f"{evaluator.output_dir}/{file_name}",
            stream=evaluator.stream,
            include_reasoning=evaluator.include_reasoning
        )
        
        # Run the evaluation
        sample_evaluator.run_all(save_every=1, file_type="csv")
        
        print("\n✅ Evaluation completed!")
        
        # Read and display results
        results_file = f"{evaluator.output_dir}/{file_name}/agent_evaluation_results.csv"
        
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            print("\n📊 Results Summary:")
            
            # Calculate averages for each scorer
            scorer_columns = [col for col in results_df.columns if col not in ['user_id', 'task_id', 'turn_id', 'agent_name'] and not col.endswith('_reasoning')]
            
            for col in scorer_columns:
                if results_df[col].dtype in ['float64', 'int64']:
                    avg_score = results_df[col].mean()
                    print(f"  - {col}: {avg_score:.2f}")
            
            # Show individual scores
            print("\n🔍 Individual Scores:")
            for i, row in results_df.iterrows():
                print(f"\n  Record {i+1} (Task: {row.get('task_id', 'N/A')}):")
                for col in scorer_columns:
                    if pd.notna(row[col]):
                        print(f"    - {col}: {row[col]}")
                        
            return results_df
        else:
            print("❌ Results file not found")
            return None
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

print("✅ run_evaluation function defined!")

def analyze_agent_behavior_patterns(dataset: AgentDataset) -> Dict[str, Any]:
    """
    Analyze agent behavior patterns including tool usage, task types, and response statistics.
    
    Args:
        dataset: AgentDataset to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    print("🔍 Dataset Analysis:")
    print("\n=== Agent Behavior Patterns ===")
    
    # Analyze tool usage patterns
    tool_patterns = {}
    task_types = {}
    response_lengths = []
    
    for data in dataset.data:
        # Tool usage
        if data.tool_calls:
            for tool_call in data.tool_calls:
                if hasattr(tool_call, 'tool_name'):
                    tool_name = tool_call.tool_name
                    if tool_name not in tool_patterns:
                        tool_patterns[tool_name] = {'count': 0, 'success_rate': 0}
                    tool_patterns[tool_name]['count'] += 1
        
        # Task analysis
        if data.agent_task:
            # Simple categorization
            task_lower = data.agent_task.lower()
            if 'user_input' in task_lower:
                task_types['user_input'] = task_types.get('user_input', 0) + 1
            elif 'exit' in task_lower:
                task_types['exit_command'] = task_types.get('exit_command', 0) + 1
            else:
                task_types['other'] = task_types.get('other', 0) + 1
        
        # Response analysis
        if data.agent_response:
            response_lengths.append(len(data.agent_response))
    
    print("\n📈 Tool Usage:")
    for tool, stats in tool_patterns.items():
        print(f"  - {tool}: {stats['count']} uses")
    
    print("\n📋 Task Types:")
    for task_type, count in task_types.items():
        print(f"  - {task_type}: {count}")
    
    analysis_results = {
        'tool_patterns': tool_patterns,
        'task_types': task_types,
        'response_lengths': response_lengths
    }
    
    if response_lengths:
        avg_response_length = sum(response_lengths) / len(response_lengths)
        print("\n📝 Response Statistics:")
        print(f"  - Average response length: {avg_response_length:.1f} characters")
        print(f"  - Min response length: {min(response_lengths)}")
        print(f"  - Max response length: {max(response_lengths)}")
        
        analysis_results['avg_response_length'] = avg_response_length
        analysis_results['min_response_length'] = min(response_lengths)
        analysis_results['max_response_length'] = max(response_lengths)
    
    return analysis_results

print("✅ analyze_agent_behavior_patterns function defined!")

def export_processed_dataset(dataset: AgentDataset, file_name: str = "processed_agent_dataset") -> bool:
    """
    Export processed AgentDataset to JSON and CSV formats.
    
    Args:
        dataset: AgentDataset to export
        file_name: Base name for export files
        
    Returns:
        True if export successful, False otherwise
    """
    print("💾 Exporting processed dataset...")
    
    success = True
    
    # Create directory if it doesn't exist
    output_dir = Path(file_name).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Export to JSON
        json_file = f'{file_name}.json'
        dataset.export_to_json(json_file)
        print(f"✅ Exported to {json_file}")
        
    except Exception as e:
        print(f"❌ JSON export error: {e}")
        success = False
    
    try:
        # Export to CSV (optional)
        csv_file = f'{file_name}.csv'
        dataset.export_to_csv(csv_file)
        print(f"✅ Exported to {csv_file}")
        
    except Exception as e:
        print(f"❌ CSV export error: {e}")
        success = False
    
    if success:
        print("✅ Export completed successfully!")
    else:
        print("❌ Some exports failed")
        
    return success

print("✅ export_processed_dataset function defined!")
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the evaluation process.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )
    
    print(f"✅ Logging configured at {log_level} level")

print("✅ setup_logging function defined!")

def validate_environment() -> Dict[str, bool]:
    """
    Check for required environment variables and dependencies.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'gemini_api_key': 'GEMINI_API_KEY' in os.environ,
        'pandas_available': True,
        'novaeval_available': True
    }
    
    try:
        import pandas  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        results['pandas_available'] = False
        
    try:
        from novaeval.agents.agent_data import AgentData as _AgentData  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        results['novaeval_available'] = False
    
    print("🔍 Environment validation:")
    for key, value in results.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
    
    return results

print("✅ validate_environment function defined!")

def print_demo_summary(file_name: str, spans_count: int, dataset_count: int, 
                      has_results: bool = False) -> None:
    """
    Print a summary of the demo execution.
    
    Args:
        file_name: Name of the processed file
        spans_count: Number of spans processed
        dataset_count: Number of AgentData records created
        has_results: Whether evaluation results are available
    """
    print("\n🎉 Demo completed successfully!")
    print("\n📋 Summary:")
    print(f"  - Processed {spans_count} spans from {file_name}")
    print(f"  - Created {dataset_count} AgentData records")
    print("  - Configured evaluation with Gemini model and 5 scorers")
    if has_results:
        print("  - Successfully evaluated sample data")
    print("  - Exported processed dataset for future use")

print("✅ print_demo_summary function defined!")


def run_complete_agent_evaluation(selected_file: str, 
                                sample_size: int = 25,
                                evaluation_name: str = "agent_evaluation",
                                model_name: str = "gemini-2.0-flash-lite",
                                temperature: float = 0.1,
                                max_tokens: int = 1024,
                                output_dir: str = "./evaluation_results") -> Dict[str, Any]:
    """
    Complete agent evaluation pipeline in a single method call.
    
    Args:
        selected_file: Path to the JSON dataset file to evaluate
        sample_size: Number of samples to evaluate (default: 25)
        evaluation_name: Name for the evaluation run (default: "agent_evaluation")
        model_name: Gemini model to use (default: "gemini-2.0-flash-lite")
        temperature: Model temperature (default: 0.1)
        max_tokens: Max tokens for model (default: 1024)
        output_dir: Output directory for results (default: "./evaluation_results")
        
    Returns:
        Dictionary containing all results and status information
    """
    
    print("🚀 Starting Complete Agent Evaluation Pipeline")
    print(f"📁 Processing file: {selected_file}")
    print("=" * 60)
    
    # Initialize results tracking
    results = {
        'success': False,
        'file_processed': selected_file,
        'spans_loaded': 0,
        'dataset_created': False,
        'dataset_size': 0,
        'evaluation_completed': False,
        'results_df': None,
        'export_success': False,
        'errors': []
    }
    
    try:
        # Step 1: Setup logging and validate environment
        print("\n📋 Step 1: Environment Setup")
        setup_logging(log_level="INFO")
        env_status = validate_environment()
        
        if not env_status['novaeval_available']:
            results['errors'].append("NovaEval not available")
            print("❌ NovaEval not available. Please install it first.")
            return results
        elif not env_status['gemini_api_key']:
            results['errors'].append("GEMINI_API_KEY not set")
            print("❌ GEMINI_API_KEY not set. Evaluation cannot proceed.")
            return results
        else:
            print("✅ Environment ready for evaluation!")
        
        # Step 2: Load and analyze dataset
        print("\n📋 Step 2: Loading Dataset")
        spans_data, span_types = load_and_analyze_dataset(selected_file)
        
        if not spans_data:
            results['errors'].append("Failed to load dataset")
            print("❌ Failed to load dataset")
            return results
            
        results['spans_loaded'] = len(spans_data)
        print(f"✅ Dataset loaded: {len(spans_data)} spans")
        
        # Step 3: Convert to AgentDataset format
        print("\n📋 Step 3: Converting to AgentDataset Format")
        agent_data_list, conversion_errors, dataset = convert_spans_to_agent_dataset(spans_data)
        
        if not dataset or not dataset.data:
            results['errors'].append("Failed to create dataset")
            results['errors'].extend(conversion_errors[:5])  # Add first 5 conversion errors
            print("❌ Failed to create dataset")
            return results
            
        results['dataset_created'] = True
        results['dataset_size'] = len(dataset.data)
        print(f"✅ AgentDataset created: {len(dataset.data)} records")
        
        # Step 4: Dataset analysis
        print("\n📋 Step 4: Dataset Analysis")
        stats = analyze_dataset_statistics(dataset)
        behavior_analysis = analyze_agent_behavior_patterns(dataset)
        
        # Step 5: Setup evaluation components
        print("\n📋 Step 5: Setting up Evaluation")
        gemini_model = setup_gemini_model(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if not gemini_model:
            results['errors'].append("Failed to setup Gemini model")
            print("❌ Failed to setup Gemini model")
            return results
            
        evaluator = setup_agent_evaluator(
            dataset=dataset,
            gemini_model=gemini_model,
            output_dir=output_dir,
            include_reasoning=True,
            stream=False
        )
        
        if not evaluator:
            results['errors'].append("Failed to setup evaluator")
            print("❌ Failed to setup evaluator")
            return results
            
        print("✅ Evaluation components ready!")
        
        # Step 6: Run evaluation
        print("\n📋 Step 6: Running Evaluation")
        print(f"🎯 Evaluating {sample_size} samples...")
        
        results_df = run_evaluation(
            dataset=dataset,
            evaluator=evaluator,
            sample_size=sample_size,
            file_name=evaluation_name
        )
        
        if results_df is not None:
            results['evaluation_completed'] = True
            results['results_df'] = results_df
            print("✅ Evaluation completed successfully!")
        else:
            results['errors'].append("Evaluation failed")
            print("❌ Evaluation failed")
        
        # Step 7: Export processed dataset
        print("\n📋 Step 7: Exporting Dataset")
        export_success = export_processed_dataset(
            dataset=dataset,
            file_name=f"./processed_datasets/{evaluation_name}_processed_dataset"
        )
        
        results['export_success'] = export_success
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 EVALUATION PIPELINE COMPLETED!")
        print(f"📊 Final Results:")
        print(f"  - File processed: {selected_file}")
        print(f"  - Spans loaded: {results['spans_loaded']}")
        print(f"  - Dataset size: {results['dataset_size']}")
        print(f"  - Evaluation completed: {results['evaluation_completed']}")
        print(f"  - Export successful: {results['export_success']}")
        
        if results['evaluation_completed']:
            print(f"  - Results saved to: {output_dir}/{evaluation_name}/")
        
        if results['errors']:
            print(f"  - Errors encountered: {len(results['errors'])}")
            
        results['success'] = results['evaluation_completed'] and results['export_success']
        
        return results
        
    except Exception as e:
        error_msg = f"Pipeline failed with error: {str(e)}"
        results['errors'].append(error_msg)
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return results

print("✅ run_complete_agent_evaluation function defined!")