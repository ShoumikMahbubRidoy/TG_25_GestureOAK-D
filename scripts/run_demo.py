#!/usr/bin/env python3
"""
Direct demo runner for development
"""

import sys
from pathlib import Path

# Add src to Python path for development
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from gesture_oak.demos.hand_detection_demo import main as hand_demo_main

if __name__ == "__main__":
    hand_demo_main()