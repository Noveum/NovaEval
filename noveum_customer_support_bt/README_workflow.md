# Noveum Platform Score Upload Workflow

## Prerequisites
- Virtual environment activated
- .env file configured with API credentials
- Processed dataset available
- Evaluation results CSV available

## Step-by-Step Commands

### Step 1: Setup Environment
```bash
cd /Users/mramanindia/work/NovaEval
source .venv/bin/activate
cd noveum_customer_support_bt
```

### Step 2: Create Dataset
```bash
python create_dataset.py --dataset-type agent --description "Customer Support Agent Evaluation Dataset" --pretty
```
**Note**: After this step, update your .env file with the returned dataset slug if different.

### Step 3: Create Dataset Version
```bash
python create_dataset_version.py --pretty
```

### Step 4: Upload Dataset Items
```bash
python upload_dataset.py --dataset-json processed_datasets/agent.rag_evaluation_metrics_dataset_processed_dataset.json --item-type conversation
```

### Step 5: Publish Dataset Version
```bash
python publish_dataset_version.py --pretty
```

### Step 6: Upload Evaluation Scores

#### Option A: Upload All Scores Separately
```bash
# Task Progression Scores
python upload_scores.py demo_results/agent.rag_evaluation_metrics_dataset/agent_evaluation_results.csv --item-key-col turn_id --score-col task_progression --reasoning-col task_progression_reasoning --scorer-id task_progression_scorer --scorer-version 1.0.0

# Context Relevancy Scores
python upload_scores.py demo_results/agent.rag_evaluation_metrics_dataset/agent_evaluation_results.csv --item-key-col turn_id --score-col context_relevancy --reasoning-col context_relevancy_reasoning --scorer-id context_relevancy_scorer --scorer-version 1.0.0

# Role Adherence Scores
python upload_scores.py demo_results/agent.rag_evaluation_metrics_dataset/agent_evaluation_results.csv --item-key-col turn_id --score-col role_adherence --reasoning-col role_adherence_reasoning --scorer-id role_adherence_scorer --scorer-version 1.0.0

# Tool Relevancy Scores
python upload_scores.py demo_results/agent.rag_evaluation_metrics_dataset/agent_evaluation_results.csv --item-key-col turn_id --score-col tool_relevancy --reasoning-col tool_relevancy_reasoning --scorer-id tool_relevancy_scorer --scorer-version 1.0.0

# Parameter Correctness Scores
python upload_scores.py demo_results/agent.rag_evaluation_metrics_dataset/agent_evaluation_results.csv --item-key-col turn_id --score-col parameter_correctness --reasoning-col parameter_correctness_reasoning --scorer-id parameter_correctness_scorer --scorer-version 1.0.0
```

#### Option B: Test with Dry Run First
Add `--dry-run` flag to any upload command to test without actually uploading:
```bash
python upload_scores.py demo_results/agent.rag_evaluation_metrics_dataset/agent_evaluation_results.csv --item-key-col turn_id --score-col task_progression --reasoning-col task_progression_reasoning --scorer-id task_progression_scorer --scorer-version 1.0.0 --dry-run
```

## Environment Variables Required
Make sure your .env file contains:
- NOVEUM_API_KEY
- NOVEUM_ORG_SLUG
- NOVEUM_DATASET_SLUG
- NOVEUM_DATASET_NAME
- LATEST_VERSION
- NOVEUM_PROJECT
- NOVEUM_ENVIRONMENT
- BETA (true/false)

## Troubleshooting
- If dataset creation fails, check if dataset already exists
- If upload fails, verify the JSON format matches expected schema
- Use --pretty flag for better formatted output
- Check API responses for specific error messages
