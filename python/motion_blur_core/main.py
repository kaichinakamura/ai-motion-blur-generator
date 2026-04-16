"""
Entry point for the Motion Blur Core sidecar process.

Communication protocol:
  - Tauri (Rust) spawns this process as a sidecar.
  - This process reads JSON requests from stdin (one per line).
  - Responses and progress updates are written to stdout as JSON (one per line).
  - Errors are written to stderr.
"""

import json
import sys


def main() -> None:
    """Main loop: read JSON commands from stdin and respond via stdout."""
    # Signal readiness to the Tauri host
    _send({"status": "ready", "message": "Motion Blur Core initialized"})

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _send_error(f"Invalid JSON: {exc}")
            continue

        command = request.get("command")
        if command == "ping":
            _send({"status": "ok", "message": "pong"})
        elif command == "process":
            input_path = request.get("input")
            output_path = request.get("output", "output.mp4")
            shutter_angle = request.get("shutter_angle", 180.0)

            def progress_cb(percent, msg):
                _send({
                    "status": "progress",
                    "progress": percent,
                    "message": msg
                })

            try:
                from .pipeline import process_video
                process_video(input_path, output_path, shutter_angle, progress_cb)
                _send({"status": "done", "output": output_path})
            except Exception as e:
                _send_error(f"Processing failed: {e}")
        else:
            _send_error(f"Unknown command: {command}")


def _send(data: dict) -> None:
    """Send a JSON response to stdout (for Tauri to read)."""
    print(json.dumps(data), flush=True)


def _send_error(message: str) -> None:
    """Send an error message to stdout so the frontend can parse the JSON error event."""
    print(json.dumps({"status": "error", "message": message}), file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
