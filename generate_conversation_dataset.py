#!/usr/bin/env python3
"""
Script to generate a conversation dataset in the format specified by schema.tsx.

This script creates a JSON file containing conversation records that match the
noveum_dataset_items schema, with focus on conversation_id, speaker, message,
conversation_context, and turn_id fields.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    print("Please install langchain-openai: pip install langchain-openai")
    exit(1)


def generate_conversation_dataset(
    conversation_id: str,
    num_turns: int = 10,
    openai_api_key: str = None,
    conversation_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Generate a conversation dataset with LLM responses.

    Args:
        conversation_id: Unique identifier for the conversation
        num_turns: Number of conversation turns to generate (each turn is one message)
        openai_api_key: OpenAI API key
        conversation_index: Index of this conversation (to vary the prompts)

    Returns:
        List of dataset items matching the schema
    """
    # Initialize LangChain OpenAI model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key
    )

    # Conversation messages to accumulate context
    conversation_messages = [
        SystemMessage(content="You are a helpful AI assistant. Keep your responses conversational and informative.")
    ]

    dataset_items = []

    # Different sets of human prompts for variety across conversations
    prompt_sets = [
        # Conversation 1 prompts
        [
            "Hello! Can you tell me about machine learning?",
            "That's interesting. How does supervised learning work?",
            "Can you give me an example of a machine learning application?",
            "What are the challenges in machine learning?",
            "How can I get started with machine learning?"
        ],
        # Conversation 2 prompts
        [
            "Hi! What is artificial intelligence?",
            "How does AI differ from machine learning?",
            "What are some real-world applications of AI?",
            "What are the ethical concerns with AI?",
            "What programming languages are good for AI development?"
        ],
        # Conversation 3 prompts
        [
            "Hello! Can you explain deep learning?",
            "How do neural networks work?",
            "What is the difference between deep learning and traditional ML?",
            "What datasets are commonly used in deep learning?",
            "What tools do I need to start with deep learning?"
        ]
    ]

    # Select prompts based on conversation index, with fallback
    human_prompts = prompt_sets[min(conversation_index, len(prompt_sets) - 1)]

    for turn_id in range(1, num_turns + 1):
        # Determine if this is a human turn (odd) or LLM turn (even)
        is_human_turn = turn_id % 2 == 1

        if is_human_turn:
            # Human turn
            message_index = (turn_id - 1) // 2
            if message_index >= len(human_prompts):
                # If we run out of prompts, create a follow-up
                human_message = "Can you elaborate more on that?"
            else:
                human_message = human_prompts[message_index]

            speaker = "human"
            message = human_message
        else:
            # LLM turn - generate response based on conversation so far
            llm_response = llm.invoke(conversation_messages).content
            speaker = "LLM"
            message = llm_response

        # Create conversation context (all messages before this turn)
        conversation_context = json.dumps([
            {"speaker": "human" if isinstance(msg, HumanMessage) else "LLM" if isinstance(msg, AIMessage) else "system",
             "message": msg.content}
            for msg in conversation_messages[1:]  # Skip system message
        ])

        # Create dataset item
        item = create_dataset_item(
            conversation_id=conversation_id,
            speaker=speaker,
            message=message,
            conversation_context=conversation_context,
            turn_id=str(turn_id)
        )
        dataset_items.append(item)

        # Add message to conversation context for next turn
        if is_human_turn:
            conversation_messages.append(HumanMessage(content=message))
        else:
            conversation_messages.append(AIMessage(content=message))

    return dataset_items


def create_dataset_item(
    conversation_id: str,
    speaker: str,
    message: str,
    conversation_context: str,
    turn_id: str
) -> Dict[str, Any]:
    """
    Create a dataset item matching the noveum_dataset_items schema.

    Args:
        conversation_id: The conversation identifier
        speaker: "human" or "LLM"
        message: The message content
        conversation_context: JSON string of previous conversation
        turn_id: Turn position as string

    Returns:
        Dictionary matching the schema
    """
    # Generate UUID for item_key
    item_key = str(uuid.uuid4())

    # Numeric value for quality_score (using turn_id as requested)
    numeric_value = int(turn_id)

    return {
        # Primary key fields
        "item_id": "",  # String - empty
        "dataset_id": "sample",
        "item_key": item_key,
        "item_hash": "",  # String - empty
        "organization_id": "",  # String - empty
        "organization_slug": "noveum",
        "dataset_slug": "",  # String - empty

        # Version and deletion fields (defaults)
        "item_version": "",
        "deleted_at_version": "",
        "deleted_at_date": "1970-01-01T00:00:00.000000000",  # Default datetime

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
        "turn_id": turn_id,
        "ground_truth": "",
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

        # Conversation fields (REQUIRED)
        "conversation_id": conversation_id,
        "speaker": speaker,
        "message": message,
        "conversation_context": conversation_context,

        # Input/Output fields
        "input_text": "",
        "output_text": "",
        "expected_output": "",

        # Evaluation fields
        "evaluation_context": "{}",
        "criteria": "",
        "quality_score": float(numeric_value),  # Float64 - using turn_id value
        "validation_status": "",
        "validation_errors": "[]",

        # Tags and custom attributes
        "tags": "{}",
        "custom_attributes": "{}",

        # Timestamps (will use defaults in ClickHouse)
        "metadata": "{}"
    }


def main():
    """Main function to generate and save the conversation dataset."""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Configuration for conversations
    conversations_config = [
        {"turns": 10},  # Conversation 1: 10 turns
        {"turns": 10},  # Conversation 2: 10 turns
        {"turns": 10}   # Conversation 3: 10 turns
    ]

    all_dataset_items = []

    # Generate multiple conversations
    base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, config in enumerate(conversations_config, 1):
        conversation_id = f"conv_{base_timestamp}_{i}"

        print(f"Generating conversation {i}/{len(conversations_config)} with ID: {conversation_id} ({config['turns']} turns)")

        # Generate dataset for this conversation
        dataset_items = generate_conversation_dataset(
            conversation_id=conversation_id,
            num_turns=config['turns'],
            openai_api_key=api_key,
            conversation_index=i-1  # 0-based index for prompt selection
        )

        all_dataset_items.extend(dataset_items)

    # Save to JSON file
    output_file = "conversation_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_dataset_items, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(all_dataset_items)} total dataset items across {len(conversations_config)} conversations")
    print(f"Saved to {output_file}")

    # Print summary
    conversation_counts = {}
    for item in all_dataset_items:
        conv_id = item['conversation_id']
        conversation_counts[conv_id] = conversation_counts.get(conv_id, 0) + 1

    print("\nConversation breakdown:")
    for conv_id, count in conversation_counts.items():
        print(f"  {conv_id}: {count} items")

    # Print sample item
    if all_dataset_items:
        print("\nSample item structure:")
        sample = {k: v for k, v in all_dataset_items[0].items() if v not in ["", "[]", "{}", 0.0, "1970-01-01T00:00:00.000000000"]}
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
