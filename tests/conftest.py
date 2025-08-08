"""
Pytest configuration for NovaEval tests.

This file configures the test environment to ensure proper imports
and test discovery without manual sys.path manipulation.
"""

import sys
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure the src directory is in the path for development installations
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
