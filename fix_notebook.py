#!/usr/bin/env python3
"""
Script to fix the agent_evaluation_demo.ipynb notebook.
"""

import json
import sys

def fix_notebook():
    # Read the original notebook
    with open('demo/agent_evaluation_demo.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Fix the first cell imports
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_lines = cell['source']
            
            # Find and fix the import line
            for j, line in enumerate(source_lines):
                if 'from novaeval.scorers.agent_scorers import AgentTaskCompletionScorer' in line:
                    # Replace with correct imports
                    source_lines[j] = "from novaeval.scorers.agent_scorers import (\n"
                    source_lines.insert(j+1, "    context_relevancy_scorer,\n")
                    source_lines.insert(j+2, "    role_adherence_scorer,\n")
                    source_lines.insert(j+3, "    task_progression_scorer,\n")
                    source_lines.insert(j+4, "    tool_relevancy_scorer,\n")
                    source_lines.insert(j+5, "    tool_correctness_scorer,\n")
                    source_lines.insert(j+6, "    parameter_correctness_scorer\n")
                    source_lines.insert(j+7, ")\n")
                    break
            
            # Fix scorer initialization
            for j, line in enumerate(source_lines):
                if 'AgentTaskCompletionScorer(' in line:
                    # Replace scorer initialization with function list
                    # Find the end of the scorers list
                    start_idx = j
                    bracket_count = 0
                    end_idx = j
                    for k in range(j, len(source_lines)):
                        if '[' in source_lines[k]:
                            bracket_count += source_lines[k].count('[')
                        if ']' in source_lines[k]:
                            bracket_count -= source_lines[k].count(']')
                            if bracket_count == 0:
                                end_idx = k
                                break
                    
                    # Replace the entire scorers section
                    new_scorers = [
                        "# Initialize scoring functions for evaluation\n",
                        "scoring_functions = [\n",
                        "    task_progression_scorer,\n",
                        "    context_relevancy_scorer,\n",
                        "    role_adherence_scorer,\n",
                        "    tool_relevancy_scorer,\n",
                        "    tool_correctness_scorer,\n",
                        "    parameter_correctness_scorer\n",
                        "]\n",
                        "\n",
                        "print(f\"✅ Initialized {len(scoring_functions)} scoring functions:\")\n",
                        "for func in scoring_functions:\n",
                        "    print(f\"  - {func.__name__}\")\n",
                        "\n",
                        "# Create AgentEvaluator\n",
                        "if gemini_model:\n",
                        "    evaluator = AgentEvaluator(\n",
                        "        agent_dataset=dataset,\n",
                        "        models=[gemini_model],\n",
                        "        scoring_functions=scoring_functions,\n",
                        "        output_dir=\"./demo_results\",\n",
                        "        stream=False,\n",
                        "        include_reasoning=True\n",
                        "    )\n",
                        "    print(\"\\n✅ AgentEvaluator created with Gemini model and scoring functions\")\n",
                        "else:\n",
                        "    print(\"\\n❌ Cannot create evaluator - Gemini model not available\")\n"
                    ]
                    
                    # Replace the lines
                    source_lines[start_idx:end_idx+1] = new_scorers
                    break
    
    # Write the fixed notebook
    with open('demo/agent_evaluation_demo_fixed.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Fixed notebook saved as demo/agent_evaluation_demo_fixed.ipynb")

if __name__ == "__main__":
    fix_notebook()
