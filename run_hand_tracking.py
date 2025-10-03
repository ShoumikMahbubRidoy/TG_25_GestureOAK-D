# run_hand_tracking.py
#!/usr/bin/env python3
from pathlib import Path
import os, sys, time, traceback

def _exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

def _msgbox(title: str, text: str) -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x00000010)  # MB_ICONERROR
    except Exception:
        pass

# --- run from the EXE folder so relative paths work on double-click ---
os.chdir(_exe_dir())

# Make project importable both frozen and unfrozen
ROOT = _exe_dir()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gesture_oak.apps.hand_tracking_app import main as hand_app_main

if __name__ == "__main__":
    try:
        hand_app_main()
    except SystemExit:
        raise
    except Exception:
        log_path = _exe_dir() / "TG25_HandTracking_error.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        _msgbox(
            "TG25 Hand Tracking – Error",
            f"An unexpected error occurred.\n\n"
            f"A log was written to:\n{log_path}\n"
        )
        traceback.print_exc()
        time.sleep(0.3)
        raise
