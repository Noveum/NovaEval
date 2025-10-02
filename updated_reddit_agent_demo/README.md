# Updated Reddit Agent Demo

This directory contains an updated demonstration of agent evaluation using Reddit data with enhanced datasets and improved evaluation capabilities.

## How to Run the Demo

1. **Run the Jupyter Notebook:** Open `final_agent_evaluation_demo.ipynb` and execute all cells (Shift+Enter)

2. **Execute all cells** in the notebook to run the complete evaluation demo.

## Output Files

### Demo Results
- **CSV files with scores and reasoning:** Located in `demo_results/` directory
  - Each dataset has its own subdirectory with evaluation results
  - Files include `agent_evaluation_results.csv` and `agent_evaluation.log`

### NovaPilot Recommendations
- **Analysis logs:** Located in `log/` directory
  - Contains detailed analysis and recommendations from NovaPilot
  - Files are timestamped for easy identification

## Directory Structure

- `final_agent_evaluation_demo.ipynb` - Main demo notebook
- `demo_results/` - Evaluation results and scores
- `log/` - NovaPilot analysis and recommendations
- `split_datasets/` - Preprocessed datasets for evaluation (includes additional agent and tool datasets)
- `processed_datasets/` - Additional processed data
- `langchain_agent/` - Agent implementation and utilities
- Other files are helper scripts and utilities for data processing

## Prerequisites

Make sure you have the required dependencies installed and create a `.env` file with the necessary environment variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
NOVEUM_PROJECT=your_noveum_project_name
NOVEUM_ENVIRONMENT=your_noveum_environment
NOVEUM_API_KEY=your_noveum_api_key
```

## Note

This is an updated version with expanded datasets including additional agent types and tool evaluations compared to the original reddit_agent_demo.
