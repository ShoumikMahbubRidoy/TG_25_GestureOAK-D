#!/usr/bin/env python3
# TG25_Launcher.py — Start/Stop GUI for Hand Tracking worker
import os, sys, time, signal, subprocess, tkinter as tk
from tkinter import messagebox
from pathlib import Path

APP_TITLE = "TG25 Hand Tracking – Launcher"
STOP_FILE_NAME = "TG25_STOP.flag"

def _exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

class LauncherApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title(APP_TITLE)
        self.master.geometry("380x160")
        self.proc: subprocess.Popen | None = None
        self.base = _exe_dir()
        self.stop_file = self.base / STOP_FILE_NAME

        # UI
        self.btn_start = tk.Button(master, text="Start Hand Tracking", width=24, command=self.start_worker)
        self.btn_stop  = tk.Button(master, text="Stop", width=24, state=tk.DISABLED, command=self.stop_worker)
        self.lbl_status = tk.Label(master, text="Status: idle", anchor="w", justify="left")

        self.btn_start.pack(pady=10)
        self.btn_stop.pack()
        self.lbl_status.pack(fill="x", padx=10, pady=10)

        master.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    def _worker_cmd(self) -> tuple[list[str], dict]:
        """
        Build the command + env for the worker.
        In frozen mode, we expect TG25_HandTracking.exe next to this launcher.
        In source mode, we run run_hand_tracking.py with current Python.
        """
        env = os.environ.copy()
        env["TG25_STOP_FILE"] = str(self.stop_file)

        if getattr(sys, "frozen", False):
            exe = self.base / "TG25_HandTracking.exe"
            if not exe.exists():
                messagebox.showerror(APP_TITLE, f"Could not find worker:\n{exe}")
                return [], env
            return [str(exe)], env
        else:
            # Source/dev mode
            script = self.base / "run_hand_tracking.py"
            if not script.exists():
                messagebox.showerror(APP_TITLE, f"Could not find:\n{script}")
                return [], env
            return [sys.executable, "-u", str(script)], env

    def start_worker(self):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo(APP_TITLE, "Already running.")
            return
        # ensure no stale stop flag
        try: self.stop_file.unlink()
        except: pass

        cmd, env = self._worker_cmd()
        if not cmd:
            return

        creationflags = 0
        # Give the child its own process group (so we can signal/terminate cleanly)
        if os.name == "nt":
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
            creationflags |= subprocess.CREATE_NEW_CONSOLE

        try:
            self.proc = subprocess.Popen(
                cmd, cwd=_exe_dir(), env=env,
                creationflags=creationflags
            )
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Failed to start worker:\n{e}")
            self.proc = None
            return

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._set_status("running (PID: %s)" % self.proc.pid)

    def stop_worker(self):
        if not self.proc or self.proc.poll() is not None:
            self._set_status("not running")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            return

        # 1) Ask politely: create STOP flag (the app checks it and exits)
        try:
            self.stop_file.write_text("stop", encoding="utf-8")
        except Exception:
            pass

        # 2) Wait a bit for graceful exit
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)

        # 3) If still alive, try gentle terminate
        if self.proc.poll() is None:
            try:
                if os.name == "nt":
                    # Send CTRL_BREAK_EVENT to the process group (if console process)
                    # Not all builds will have a console, so ignore errors.
                    os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    time.sleep(1.0)
                self.proc.terminate()
            except Exception:
                pass

        # 4) Hard kill as last resort
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass

        # Cleanup flag
        try: self.stop_file.unlink()
        except: pass

        self._set_status("stopped")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.proc = None

    def _poll(self):
        # Periodically update status if the worker died unexpectedly
        if self.proc and self.proc.poll() is not None:
            self._set_status("exited (code %s)" % self.proc.returncode)
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.proc = None
            try: self.stop_file.unlink()
            except: pass
        self.master.after(500, self._poll)

    def _set_status(self, text: str):
        self.lbl_status.config(text=f"Status: {text}")

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno(APP_TITLE, "Worker is running. Stop it and exit?"):
                return
            self.stop_worker()
        self.master.destroy()

def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
