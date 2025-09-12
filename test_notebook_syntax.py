#!/usr/bin/env python3
"""
Test script to verify the notebook cells have valid syntax.
"""

import json
import ast

def test_notebook_syntax():
    with open('demo/agent_evaluation_demo.ipynb', 'r') as f:
        notebook = json.load(f)
    
    print("🔍 Testing notebook cell syntax...")
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'source' in cell:
            source_code = ''.join(cell['source'])
            
            if source_code.strip():  # Skip empty cells
                try:
                    # Try to parse the cell as valid Python
                    ast.parse(source_code)
                    print(f"  ✅ Cell {i}: Valid syntax")
                except SyntaxError as e:
                    print(f"  ❌ Cell {i}: Syntax error - {e}")
                    print(f"      First few lines: {source_code[:100]}...")
                    return False
    
    print("\n✅ All code cells have valid syntax!")
    return True

if __name__ == "__main__":
    test_notebook_syntax()
