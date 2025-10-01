# run_hand_tracking.py
#!/usr/bin/env python3
from pathlib import Path
import sys

# Ensure we can import the package (src/ is inside project root)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your existing app entry
from src.gesture_oak.apps.hand_tracking_app import main as hand_app_main

if __name__ == "__main__":
    # This launches the hand-tracking pipeline directly (same as menu option 2)
    hand_app_main()
