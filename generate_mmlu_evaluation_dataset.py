#!/usr/bin/env python3
"""
Script to generate MMLU evaluation dataset using G-Eval scoring.

This script loads MMLU parquet data, runs LLM inference, and uses G-Eval
to evaluate the responses, generating dataset items matching the schema.
"""

import json
import uuid
import pandas as pd
from typing import List, Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
except ImportError:
    print("Please install langchain-openai: pip install langchain-openai")
    exit(1)


def create_geval_prompt(question: str, choices: List[str], llm_response: str, correct_answer: str) -> str:
    """
    Create G-Eval prompt for evaluating MMLU responses.
    
    Args:
        question: The MMLU question
        choices: List of answer choices
        llm_response: The LLM's response
        correct_answer: The correct answer
        
    Returns:
        G-Eval evaluation prompt
    """
    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""You are an expert evaluator assessing high school mathematics problem-solving skills. Evaluate the following response based on three criteria:

**Question:**
{question}

**Answer Choices:**
{choices_text}

**LLM Response:**
{llm_response}

**Correct Answer:**
{correct_answer}

**Evaluation Criteria:**
1. **Brevity**: Is the response concise and to the point? (1-10 scale)
2. **Accuracy**: Is the response mathematically correct and demonstrates proper problem-solving? (1-10 scale)  
3. **Relevance**: Does the response directly address the question asked? (1-10 scale)

**Instructions:**
- Rate each criterion on a scale of 1-10 (1=poor, 10=excellent)
- Provide a brief reason for each score
- Calculate overall quality score as average of the three criteria
- Be objective and focus on mathematical accuracy and logical reasoning

**Format your response as JSON:**
{{
    "brevity_score": <1-10>,
    "brevity_reason": "<brief explanation>",
    "accuracy_score": <1-10>, 
    "accuracy_reason": "<brief explanation>",
    "relevance_score": <1-10>,
    "relevance_reason": "<brief explanation>",
    "overall_score": <average of three scores>,
    "overall_reason": "<summary of evaluation>"
}}"""

    return prompt


def run_llm_inference(question: str, choices: List[str], llm) -> str:
    """
    Run LLM inference on MMLU question.
    
    Args:
        question: The MMLU question
        choices: List of answer choices
        llm: LangChain LLM instance
        
    Returns:
        LLM response
    """
    choices_text = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""You are a high school mathematics expert. Answer the following question by selecting the most appropriate choice.

**Question:**
{question}

**Answer Choices:**
{choices_text}

**Instructions:**
- Think through the problem step by step
- Show your mathematical reasoning clearly
- Consider all given information carefully
- Select the best answer choice
- Provide a brief explanation for your reasoning

**Response format:**
Answer: [choice number]
Reasoning: [brief explanation]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def run_geval_evaluation(geval_prompt: str, llm) -> Dict[str, Any]:
    """
    Run G-Eval evaluation using LLM.
    
    Args:
        geval_prompt: The G-Eval evaluation prompt
        llm: LangChain LLM instance
        
    Returns:
        Parsed evaluation results
    """
    response = llm.invoke([HumanMessage(content=geval_prompt)])
    
    try:
        # Extract JSON from response
        response_text = response.content
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        elif "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_text = response_text[json_start:json_end]
        else:
            json_text = response_text
            
        evaluation = json.loads(json_text)
        return evaluation
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing G-Eval response: {e}")
        # Return default evaluation if parsing fails
        return {
            "brevity_score": 5.0,
            "brevity_reason": "Unable to parse evaluation",
            "accuracy_score": 5.0,
            "accuracy_reason": "Unable to parse evaluation", 
            "relevance_score": 5.0,
            "relevance_reason": "Unable to parse evaluation",
            "overall_score": 5.0,
            "overall_reason": "Evaluation parsing failed"
        }


def create_dataset_item(
    question: str,
    choices: List[str],
    correct_answer: str,
    llm_response: str,
    evaluation: Dict[str, Any],
    row_index: int
) -> Dict[str, Any]:
    """
    Create a dataset item matching the noveum_dataset_items schema.
    
    Args:
        question: The MMLU question
        choices: List of answer choices
        correct_answer: The correct answer
        llm_response: The LLM's response
        evaluation: G-Eval evaluation results
        row_index: Row index for turn_id
        
    Returns:
        Dictionary matching the schema
    """
    # Generate UUID for item_key
    item_key = str(uuid.uuid4())
    
    # Create evaluation context (entire G-Eval prompt)
    # Convert numpy arrays to lists for JSON serialization
    choices_list = choices.tolist() if hasattr(choices, 'tolist') else list(choices)
    
    evaluation_context = {
        "question": question,
        "choices": choices_list,
        "llm_response": llm_response,
        "correct_answer": correct_answer,
        "evaluation_criteria": ["brevity", "accuracy", "relevance"]
    }
    
    # Create criteria string
    criteria = f"Brevity: {evaluation.get('brevity_score', 0)}/10 - {evaluation.get('brevity_reason', '')}; " \
              f"Accuracy: {evaluation.get('accuracy_score', 0)}/10 - {evaluation.get('accuracy_reason', '')}; " \
              f"Relevance: {evaluation.get('relevance_score', 0)}/10 - {evaluation.get('relevance_reason', '')}"
    
    return {
        # Primary key fields
        "item_id": "",
        "dataset_id": "sample",
        "item_key": item_key,
        "item_hash": "",
        "organization_id": "",
        "organization_slug": "noveum",
        "dataset_slug": "",
        
        # Version and deletion fields (defaults)
        "item_version": "",
        "deleted_at_version": "",
        "deleted_at_date": "1970-01-01T00:00:00.000000000",
        
        # Item metadata
        "item_type": "",
        "schema_version": "",
        
        # Source tracking
        "source_trace_id": "",
        "source_span_id": "",
        
        # Content fields
        "content": "",
        
        # Agent-related fields
        "agent_name": "",
        "agent_role": "",
        "agent_task": "",
        "agent_response": "",
        "system_prompt": "",
        
        # User and session fields
        "user_id": "",
        "session_id": "",
        
        # Turn and ground truth
        "turn_id": str(row_index + 1),
        "ground_truth": correct_answer,
        "expected_tool_call": "",
        
        # Tool-related fields (JSON defaults)
        "tools_available": "[]",
        "tool_calls": "[]",
        "tool_call_results": "[]",
        "parameters_passed": "{}",
        "retrieval_query": "[]",
        "retrieved_context": "[]",
        
        # Exit status
        "exit_status": "",
        "agent_exit": "",
        
        # Trace data
        "trace_data": "{}",
        
        # Conversation fields (not used for MMLU)
        "conversation_id": "",
        "speaker": "",
        "message": "",
        "conversation_context": "{}",
        
        # Input/Output fields (REQUIRED)
        "input_text": question,
        "output_text": llm_response,
        "expected_output": correct_answer,
        
        # Evaluation fields (REQUIRED)
        "evaluation_context": json.dumps(evaluation_context),
        "criteria": criteria,
        "quality_score": float(evaluation.get('overall_score', 0.0)),
        "validation_status": "",
        "validation_errors": "[]",
        
        # Tags and custom attributes
        "tags": "{}",
        "custom_attributes": "{}",
        
        # Timestamps
        "metadata": "{}"
    }


def generate_mmlu_evaluation_dataset(
    parquet_file: str,
    num_rows: int = 30,
    openai_api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Generate MMLU evaluation dataset.
    
    Args:
        parquet_file: Path to MMLU parquet file
        num_rows: Number of rows to process
        openai_api_key: OpenAI API key
        
    Returns:
        List of dataset items
    """
    # Initialize LangChain OpenAI model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,  # Lower temperature for more consistent evaluation
        openai_api_key=openai_api_key
    )
    
    # Load MMLU data
    print(f"Loading MMLU data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} rows")
    
    # Process first num_rows (or all available if less than requested)
    actual_rows = min(num_rows, len(df))
    df_subset = df.head(actual_rows)
    print(f"Processing {actual_rows} rows (requested {num_rows})")
    dataset_items = []
    
    for idx, row in df_subset.iterrows():
        print(f"Processing row {idx + 1}/{actual_rows}")
        
        question = row['question']
        choices = row['choices']
        # Convert numpy array to list if needed
        if hasattr(choices, 'tolist'):
            choices = choices.tolist()
        else:
            choices = list(choices)
        
        correct_answer_idx = int(row['answer'])
        correct_answer = choices[correct_answer_idx] if correct_answer_idx < len(choices) else "Unknown"
        
        # Run LLM inference
        print("  Running LLM inference...")
        llm_response = run_llm_inference(question, choices, llm)
        
        # Create G-Eval prompt
        geval_prompt = create_geval_prompt(question, choices, llm_response, correct_answer)
        
        # Run G-Eval evaluation
        print("  Running G-Eval evaluation...")
        evaluation = run_geval_evaluation(geval_prompt, llm)
        
        # Create dataset item
        item = create_dataset_item(
            question=question,
            choices=choices,
            correct_answer=correct_answer,
            llm_response=llm_response,
            evaluation=evaluation,
            row_index=idx
        )
        
        dataset_items.append(item)
        print(f"  Quality score: {evaluation.get('overall_score', 0.0)}")
    
    return dataset_items


def main():
    """Main function to generate and save the MMLU evaluation dataset."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Configuration
    parquet_file = "validation-00000-of-00001.parquet"
    num_rows = 29
    
    print(f"Generating MMLU evaluation dataset for {num_rows} rows")
    
    # Generate dataset
    dataset_items = generate_mmlu_evaluation_dataset(
        parquet_file=parquet_file,
        num_rows=num_rows,
        openai_api_key=api_key
    )
    
    # Save to JSON file
    output_file = "mmlu_evaluation_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_items, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(dataset_items)} dataset items")
    print(f"Saved to {output_file}")
    
    # Print summary statistics
    quality_scores = [item['quality_score'] for item in dataset_items]
    avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print("\nEvaluation Summary:")
    print(f"  Average quality score: {avg_score:.2f}")
    print(f"  Min quality score: {min(quality_scores):.2f}")
    print(f"  Max quality score: {max(quality_scores):.2f}")
    
    # Print sample item
    if dataset_items:
        print("\nSample item structure:")
        sample = {k: v for k, v in dataset_items[0].items() if v not in ["", "[]", "{}", 0.0, "1970-01-01T00:00:00.000000000"]}
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
