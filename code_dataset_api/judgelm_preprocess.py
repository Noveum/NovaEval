#!/usr/bin/env python3
"""
Script to preprocess JudgeLM dataset and map to NovaEval schema.

This script reads the JudgeLM dataset and transforms it to match the schema
defined in schema.tsx, creating the required field mappings.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional


def extract_criteria_text(text: str) -> str:
    """
    Extract criteria text from evaluation text using regex.
    Extracts text from "Assistant 1's" to "Assistant 2's".
    
    Args:
        text: The evaluation text containing assistant comparisons
        
    Returns:
        Extracted criteria text, or empty string if pattern not found
    """
    try:
        # Pattern to match text from "Assistant 1" to "Assistant 2" 
        pattern = r"Assistant 1.*?(?=Assistant 2)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(0).strip()
        else:
            # Fallback: return the entire text after the score line
            lines = text.split('\n')
            if len(lines) > 1:
                return '\n'.join(lines[1:]).strip()
            return text.strip()
    except Exception as e:
        print(f"Warning: Error extracting criteria text: {e}")
        return text.strip()


def preprocess_judgelm_item(item: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    """
    Transform a single JudgeLM item to match the target schema.
    
    Args:
        item: Original JudgeLM dataset item
        dataset_id: Dataset identifier (filename without extension)
        
    Returns:
        Transformed item matching the target schema
    """
    # Extract reference text
    reference_text = ""
    if isinstance(item.get('reference'), dict):
        reference_text = item['reference'].get('text', '')
    
    # Extract quality score (first score from score_w_reference)
    quality_score = 0.0
    if 'score_w_reference' in item and isinstance(item['score_w_reference'], list):
        if len(item['score_w_reference']) > 0:
            quality_score = float(item['score_w_reference'][0])
    
    # Extract criteria text using regex
    criteria_text = extract_criteria_text(item.get('text', ''))
    
    # Create the mapped item
    mapped_item = {
        # Core mappings as specified
        'item_id': item.get('review_id', ''),  # view_id -> item_id
        'dataset_id': dataset_id,  # filename -> dataset_id
        'question_id': str(item.get('question_id', '')),  # turn_id -> question_id  
        'input_text': item.get('question_body', ''),  # question_body -> input_text
        'output_text': item.get('answer1_body', ''),  # answer1_body -> output_text
        'expected_output': reference_text,  # reference.text -> expected_output
        'ground_truth': reference_text,  # reference.text -> ground_truth
        'quality_score': quality_score,  # score_w_reference[0] -> quality_score
        'criteria': criteria_text,  # extracted from text field
        
        # Additional schema fields with default values
        'item_key': f"{item.get('review_id', '')}_{item.get('question_id', '')}",
        'item_hash': str(uuid.uuid4()),
        'organization_id': '',
        'organization_slug': '',
        'dataset_slug': dataset_id,
        'item_version': '1.0',
        'deleted_at_version': '',
        'item_type': 'evaluation',
        'schema_version': '1.0',
        'source_trace_id': '',
        'source_span_id': '',
        'content': json.dumps({
            'question': item.get('question_body', ''),
            'answer1': item.get('answer1_body', ''),
            'answer2': item.get('answer2_body', ''),
            'reference': item.get('reference', {}),
            'original_text': item.get('text', ''),
            'scores': item.get('score', []),
            'scores_w_reference': item.get('score_w_reference', [])
        }),
        'metadata': json.dumps(item.get('metadata', {})),
        'agent_name': item.get('answer1_model_id', ''),
        'agent_role': 'assistant',
        'agent_task': 'question_answering',
        'agent_response': item.get('answer1_body', ''),
        'system_prompt': '',
        'user_id': '',
        'session_id': str(item.get('question_id', '')),
        'turn_id': str(item.get('question_id', '')),
        'expected_tool_call': '',
        'tools_available': '[]',
        'tool_calls': '[]',
        'tool_call_results': '[]',
        'parameters_passed': '{}',
        'retrieval_query': '[]',
        'retrieved_context': '[]',
        'exit_status': '',
        'agent_exit': '',
        'trace_data': '{}',
        'conversation_id': str(item.get('question_id', '')),
        'speaker': 'assistant',
        'message': item.get('answer1_body', ''),
        'conversation_context': '{}',
        'evaluation_context': json.dumps({
            'evaluator_model':'gpt-4o',
            'answer1_model': item.get('answer1_model_id', ''),
            'answer1_metadata': item.get('answer1_metadata', {}),
        }),
        'validation_status': 'valid',
        'validation_errors': '[]',
        'tags': '{}',
        'custom_attributes': json.dumps({
            'answer2_body': item.get('answer2_body', ''),
            'answer2_model_id': item.get('answer2_model_id', ''),
            'score_without_reference': item.get('score', []),
            'text_w_reference': item.get('text_w_reference', '')
        })
    }
    
    return mapped_item


def preprocess_judgelm_dataset(
    input_file: str,
    output_file: str,
    max_items: Optional[int] = None
) -> None:
    """
    Preprocess the entire JudgeLM dataset.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file  
        max_items: Maximum number of items to process (None for all)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Use filename (without extension) as dataset_id
    dataset_id = input_path.stem
    
    print(f"Processing JudgeLM dataset: {input_file}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Output file: {output_file}")
    
    processed_items = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_items and len(processed_items) >= max_items:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    processed_item = preprocess_judgelm_item(item, dataset_id)
                    processed_items.append(processed_item)
                    
                    if len(processed_items) % 1000 == 0:
                        print(f"Processed {len(processed_items)} items...")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing item at line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Save processed data
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_items, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Successfully processed {len(processed_items)} items")
        print(f"✅ Output saved to: {output_file}")
        
        # Print sample mapping for verification
        if processed_items:
            print("\n=== Sample mapping verification ===")
            sample = processed_items[0]
            print(f"item_id: {sample['item_id']}")
            print(f"dataset_id: {sample['dataset_id']}")
            print(f"question_id: {sample['question_id']}")
            print(f"input_text: {sample['input_text'][:100]}...")
            print(f"output_text: {sample['output_text'][:100]}...")
            print(f"expected_output: {sample['expected_output'][:100]}...")
            print(f"quality_score: {sample['quality_score']}")
            print(f"criteria: {sample['criteria'][:150]}...")
            
    except Exception as e:
        print(f"Error saving output file: {e}")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess JudgeLM dataset to match NovaEval schema"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file (e.g., /mnt/drive2/judgelm_train_100k.jsonl)"
    )
    parser.add_argument(
        "--output_file", 
        help="Path to output JSON file (e.g., ./judgelm_processed.json)",
        default="./judgelm_processed.json"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum number of items to process (default: process all)"
    )
    
    args = parser.parse_args()
    
    preprocess_judgelm_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        max_items=args.max_items
    )


if __name__ == "__main__":
    main()
