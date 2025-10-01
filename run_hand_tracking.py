# run_hand_tracking.py
#!/usr/bin/env python3
from pathlib import Path
import sys, time, traceback

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gesture_oak.apps.hand_tracking_app import main as hand_app_main

if __name__ == "__main__":
    try:
        hand_app_main()
    except SystemExit:
        raise
    except Exception:
        log = (Path(sys.executable).parent if getattr(sys, "frozen", False) else ROOT) / "TG25_HandTracking_error.log"
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        print("\nERROR: Unhandled exception\n")
        traceback.print_exc()
        print(f"\nA log was written to: {log}\n")
        try:
            input("Press Enter to close...")
        except Exception:
            time.sleep(4)
        raise
