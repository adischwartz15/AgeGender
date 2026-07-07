#!/usr/bin/env python
"""CLI: single command to launch the backend and frontend for a live course demonstration.

Runs the readiness check first (see check_demo_readiness.py) and refuses
to launch a demo that will silently fail on the first upload. If ready,
starts the FastAPI backend (uvicorn) and the Vite frontend dev server as
subprocesses, streams both to this terminal, and shuts both down cleanly
on Ctrl+C. Docker is intentionally not used here (removed from this
project as an unverified, never-actually-run path) -- this is plain
Python subprocess orchestration.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --skip-readiness-check   # launch anyway (not recommended)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_demo_readiness import check_demo_readiness  # noqa: E402

from src.utils.config import CONFIG_DIR, REPO_ROOT, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.run_demo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-readiness-check", action="store_true",
        help="Launch even if the checkpoint/calibration readiness check fails (not recommended)",
    )
    args = parser.parse_args()

    api_config = load_config(CONFIG_DIR / "api.yaml")["api"]
    ready, messages = check_demo_readiness(api_config)
    print("Demo readiness check")
    print("=" * 40)
    for message in messages:
        print(message)
    print("=" * 40)

    if not ready and not args.skip_readiness_check:
        print("NOT READY -- fix the [FAIL] item(s) above, or re-run with --skip-readiness-check.")
        return 1

    npm = shutil.which("npm")
    if npm is None:
        logger.error("npm not found on PATH; cannot launch the frontend dev server.")
        return 1

    backend_port = api_config.get("port", 8000)
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "src.api.main:app",
        "--host", "0.0.0.0", "--port", str(backend_port),
    ]
    frontend_cmd = [npm, "run", "dev"]

    print(f"Starting backend:  {' '.join(backend_cmd)}")
    backend_proc = subprocess.Popen(backend_cmd, cwd=REPO_ROOT)

    print(f"Starting frontend: {' '.join(frontend_cmd)} (cwd=frontend/)")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=REPO_ROOT / "frontend", shell=(sys.platform == "win32"))

    print("=" * 40)
    print(f"Backend:  http://localhost:{backend_port}/docs")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to stop both.")
    print("=" * 40)

    try:
        while backend_proc.poll() is None and frontend_proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down demo...")
    finally:
        for proc, name in ((backend_proc, "backend"), (frontend_proc, "frontend")):
            if proc.poll() is None:
                logger.info("Stopping %s (pid=%d)", name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
