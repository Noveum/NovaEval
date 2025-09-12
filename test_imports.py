#!/usr/bin/env python3
"""
Test script to verify the notebook imports work correctly.
"""

try:
    import json
    import re
    from typing import Any, Dict, List, Optional, Union

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
        tool_correctness_scorer,
        parameter_correctness_scorer
    )

    print("✅ All imports successful!")
    print("✅ The notebook should now work correctly!")
    
    # Test that the functions are callable
    print("\n🔍 Checking scorer functions:")
    for func in [context_relevancy_scorer, role_adherence_scorer, task_progression_scorer,
                 tool_relevancy_scorer, tool_correctness_scorer, parameter_correctness_scorer]:
        print(f"  - {func.__name__}: {callable(func)}")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
