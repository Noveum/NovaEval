#!/usr/bin/env python3
"""
Clean Agent Evaluation Script - Agent Scorers Only

This script uses ONLY the agent-specific scorers from agent_scorers.py
and outputs both scores and reasoning to CSV.
"""

import json
import os
import pandas as pd
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# NovaEval imports
from novaeval.agents.agent_dataset import AgentDataset
from novaeval.agents.agent_data import AgentData
from novaeval.agents.agent_scorers import AgentScorers

# Import SWE dataset creation functions
from swe_dataset_creation import preprocess_swe_csv, create_agent_dataset_from_swe_csv

# Model imports
from novaeval.models.azure_openai import AzureOpenAIModel
from novaeval.models.anthropic import AnthropicModel

def setup_model():
    """Set up the LLM model for evaluation."""
    # Try to use Azure OpenAI first, fall back to Anthropic, then mock if no API keys
    if (os.getenv("AZURE_OPENAI_API_KEY") and 
        os.getenv("AZURE_OPENAI_BASE_URL") and 
        os.getenv("AZURE_OPENAI_DEPLOYMENT")):
        print("🔧 Using Azure OpenAI model...")
        model = AzureOpenAIModel(
            model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            temperature=0.0
        )
        print(f"   Model Name: {model.model_name}")
        print(f"   API Version: {model.api_version}")
        print(f"   Base URL: {model.base_url}")
        return model
    elif os.getenv("ANTHROPIC_API_KEY"):
        print("🔧 Using Anthropic model...")
        model = AnthropicModel(
            model_name="claude-3-haiku-20240307",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0
        )
        print(f"   Model Name: {model.model_name}")
        return model
    else:
        print("⚠️  No API keys found. Creating mock model...")
        # Create a simple mock model for demonstration
        class MockModel:
            def __init__(self):
                self.model_name = "mock-model"
                self.api_version = "mock"
                self.base_url = "mock"
            
            def generate(self, prompt: str, **kwargs) -> str:
                return json.dumps({
                    "score": 3.5,
                    "reasoning": "Mock evaluation - no API key provided"
                })
            
            async def generate_async(self, prompt: str, **kwargs) -> str:
                return self.generate(prompt, **kwargs)
        
        model = MockModel()
        print(f"   Model Name: {model.model_name}")
        return model

def load_agent_dataset(csv_file: str, num_samples: int = 5) -> List[AgentData]:
    """Load the first N samples from the SWE CSV data using dataset creation pipeline."""
    print(f"Processing SWE CSV data from {csv_file}...")
    print(f"Will select first {num_samples} samples for evaluation")
    
    # Step 1: Preprocess the CSV
    processed_csv = "temp_processed_swe.csv"
    print("Step 1: Preprocessing CSV...")
    preprocess_swe_csv(csv_file, processed_csv)
    
    # Step 2: Create AgentDataset
    print("Step 2: Creating AgentDataset...")
    dataset = create_agent_dataset_from_swe_csv(processed_csv)
    
    # Step 3: Clean up temporary file
    if os.path.exists(processed_csv):
        os.remove(processed_csv)
        print(f"Cleaned up temporary file: {processed_csv}")
    
    print(f"Dataset created with {len(dataset.data)} total entries")
    samples = dataset.data[:num_samples]
    
    print(f"Selected first {len(samples)} samples for evaluation")
    return samples

def extract_score_and_reasoning(result) -> Tuple[float, str]:
    """Extract both score and reasoning from scorer result."""
    try:
        if hasattr(result, 'score') and hasattr(result, 'reasoning'):
            return float(result.score), str(result.reasoning)
        elif isinstance(result, dict):
            if 'score' in result and 'reasoning' in result:
                return float(result['score']), str(result['reasoning'])
            elif 'score' in result:
                return float(result['score']), "No reasoning provided"
            else:
                # Handle error dictionaries
                return 0.0, str(result)
        elif isinstance(result, (int, float)):
            return float(result), "No reasoning provided"
        elif isinstance(result, list) and result:
            # For list results, take average of scores and combine reasoning
            scores = []
            reasonings = []
            for item in result:
                if hasattr(item, 'score'):
                    scores.append(float(item.score))
                    reasonings.append(getattr(item, 'reasoning', 'No reasoning'))
                elif isinstance(item, dict) and 'score' in item:
                    scores.append(float(item['score']))
                    reasonings.append(item.get('reasoning', 'No reasoning'))
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            combined_reasoning = " | ".join(reasonings) if reasonings else "No reasoning provided"
            return avg_score, combined_reasoning
        elif result is None:
            return 0.0, "No result returned"
        else:
            # Handle error cases or unexpected formats
            error_msg = str(result) if result else "No result"
            return 0.0, f"Error or unexpected format: {error_msg}"
    except Exception as e:
        return 0.0, f"Error extracting score/reasoning: {str(e)}"

def get_original_metadata(agent_data: AgentData) -> Dict[str, Any]:
    """Extract original CSV metadata from agent data."""
    # Parse the metadata JSON string to get original values
    metadata = {}
    if agent_data.metadata:
        try:
            parsed_metadata = json.loads(str(agent_data.metadata))
            metadata = {
                'step_number': parsed_metadata.get('step_number', ''),
            }
        except (json.JSONDecodeError, TypeError):
            pass
    
    return {
        'traj_id': agent_data.task_id or '',
        'span_id': agent_data.turn_id or '',
        'step_number': metadata.get('step_number', ''),
    }

def evaluate_single_agent_csv(agent_data: AgentData, model, index: int) -> Dict[str, Any]:
    """Evaluate a single AgentData entry using only agent scorers."""
    # Get original identifiers
    original_data = get_original_metadata(agent_data)
    
    result_row = {
        'traj_id': original_data['traj_id'],
        'span_id': original_data['span_id'], 
        'step_number': original_data['step_number'],
        'agent_name': agent_data.agent_name or '',
        'exit_status': agent_data.exit_status or '',
    }
    
    # Run ONLY agent-specific scorers
    try:
        agent_scorers = AgentScorers(model)
        agent_results = agent_scorers.score_all(agent_data)
        
        # Extract scores AND reasoning for each scorer
        for scorer_name, result in agent_results.items():
            score, reasoning = extract_score_and_reasoning(result)
            result_row[f'{scorer_name}_score'] = score
            result_row[f'{scorer_name}_reasoning'] = reasoning
            
    except Exception as e:
        print(f"❌ Error running agent scorers for {agent_data.task_id}/{agent_data.turn_id}: {e}")
        # Add null scores and error reasoning for agent scorers (removed the three specified scorers)
        agent_scorer_names = ['tool_relevancy', 'parameter_correctness', 
                             'task_progression', 'context_relevancy', 'role_adherence']
        for name in agent_scorer_names:
            result_row[f'{name}_score'] = None
            result_row[f'{name}_reasoning'] = f"Error: {str(e)}"
    
    return result_row

def main():
    """Main evaluation workflow."""
    print("🚀 Starting Clean Agent Evaluation (Agent Scorers Only)")
    print("="*60)
    
    # Setup
    input_csv = "sample_10_swe.csv"
    num_samples = 25
    output_file = "agent_scores_with_reasoning.csv"
    
    # Check if input CSV exists
    if not os.path.exists(input_csv):
        print(f"❌ Input CSV file '{input_csv}' not found!")
        print("Please make sure the SWE CSV file is available.")
        return 1
    
    try:
        # Load model and data
        model = setup_model()
        agent_samples = load_agent_dataset(input_csv, num_samples)
        
        # Run evaluation on all samples
        all_results = []
        print(f"\n🔍 Evaluating {len(agent_samples)} samples...")
        for i, agent_data in enumerate(tqdm(agent_samples, desc="Evaluating entries", unit="entry")):
            result = evaluate_single_agent_csv(agent_data, model, i)
            all_results.append(result)
        
        # Create DataFrame and reorder columns (scores first, then reasonings)
        df = pd.DataFrame(all_results)
        
        # Separate columns by type
        identifier_columns = ['traj_id', 'span_id', 'step_number', 'agent_name', 'exit_status']
        score_columns = [col for col in df.columns if col.endswith('_score')]
        reasoning_columns = [col for col in df.columns if col.endswith('_reasoning')]
        other_columns = [col for col in df.columns if col not in identifier_columns + score_columns + reasoning_columns]
        
        # Reorder: identifiers, other columns, all scores, then all reasonings
        ordered_columns = identifier_columns + other_columns + sorted(score_columns) + sorted(reasoning_columns)
        df = df[ordered_columns]
        
        df.to_csv(output_file, index=False)
        
        print(f"\n📊 Results Summary:")
        print(f"   Total entries evaluated: {len(all_results)}")
        print(f"   Columns in output: {len(df.columns)}")
        print(f"   Output file: {output_file}")
        
        # Show column names
        print(f"\n📋 CSV Columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Show sample scores (first few columns only)
        print(f"\n🔍 Sample Results (scores only):")
        score_columns = [col for col in df.columns if col.endswith('_score')]
        identifier_columns = ['traj_id', 'span_id', 'step_number']
        display_columns = identifier_columns + score_columns[:4]  # Show first 4 score columns
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df[display_columns].head(3).to_string())
        
        print(f"\n💾 Complete results (scores + reasoning) saved to: {output_file}")
        print("\n✅ Agent evaluation completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 