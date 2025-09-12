#!/usr/bin/env python3
"""
Script to fix the specific broken cell in the notebook.
"""

import json

def fix_cell_9():
    # Read the notebook
    with open('demo/agent_evaluation_demo.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find and fix cell 9 (which has the broken scorer initialization)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_lines = cell['source']
            
            # Look for the cell with the syntax error (contains both scorers = [ and scoring_functions = [)
            source_text = ''.join(source_lines)
            if 'scorers = [' in source_text and 'scoring_functions = [' in source_text:
                print(f"Found broken cell at index {i}, fixing...")
                
                # Replace with clean content
                new_source = [
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
                
                cell['source'] = new_source
                # Clear the outputs since they contain errors
                cell['outputs'] = []
                cell['execution_count'] = None
                break
    
    # Write the fixed notebook
    with open('demo/agent_evaluation_demo.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Fixed cell 9 in the notebook")

if __name__ == "__main__":
    fix_cell_9()
