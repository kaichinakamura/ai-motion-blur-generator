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
import threading

def main() -> None:
    """Main loop: read JSON commands from stdin and respond via stdout."""
    _send({"status": "ready", "message": "Motion Blur Core initialized"})

    current_worker = None
    cancel_event = None

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
            if current_worker and current_worker.is_alive():
                _send_error("Already processing a video")
                continue
                
            input_path = request.get("input")
            output_path = request.get("output", "output.mp4")
            shutter_angle = request.get("shutter_angle", 180.0)
            use_rife = request.get("use_rife", False)
            flow_resolution = request.get("flow_resolution", 720)
            
            cancel_event = threading.Event()
            
            def worker_func(evt):
                def progress_cb(percent, msg):
                    _send({"status": "progress", "progress": percent, "message": msg})
                
                try:
                    from .pipeline import process_video
                    process_video(input_path, output_path, shutter_angle, use_rife, flow_resolution, progress_cb, evt)
                    if evt.is_set():
                        _send({"status": "done", "output": "cancelled"})
                    else:
                        _send({"status": "done", "output": output_path})
                except Exception as e:
                    _send_error(f"Processing failed: {e}")
                    
            current_worker = threading.Thread(target=worker_func, args=(cancel_event,), daemon=True)
            current_worker.start()
            
        elif command == "cancel":
            if cancel_event:
                cancel_event.set()
                _send({"status": "progress", "progress": 0, "message": "Cancelling process..."})
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
