import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "./App.css";

function App() {
  const [filePath, setFilePath] = useState<string | null>(null);
  const [outPath, setOutPath] = useState<string | null>(null);
  const [shutterAngle, setShutterAngle] = useState<number>(180);
  const [format, setFormat] = useState<string>(".mp4");

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");

  useEffect(() => {
    // Listen for progress from sidecar
    const unlistenOut = listen<string>("sidecar-output", (event) => {
      try {
        const data = JSON.parse(event.payload);
        if (data.status === "progress") {
          setProgress(data.progress || 0);
          setStatusMsg(data.message || "");
        } else if (data.status === "done") {
          setProgress(100);
          setStatusMsg(`Done! Saved to ${data.output}`);
          setIsProcessing(false);
        } else if (data.status === "error") {
          setStatusMsg(`Error: ${data.message}`);
          setIsProcessing(false);
        }
      } catch (e) {
        console.error("Failed to parse sidecar output", event.payload);
      }
    });

    const unlistenErr = listen<string>("sidecar-error", (event) => {
      console.error("Sidecar STDERR:", event.payload);
      // We do not abort processing simply because of stderr, 
      // as many libraries (e.g. ffmpeg, uv) output warnings to stderr.
    });

    // Native Drag and Drop via Tauri Window API
    const unlistenDrop = getCurrentWindow().onDragDropEvent((event) => {
      if (event.payload.type === 'drop') {
        const paths = event.payload.paths;
        if (paths && paths.length > 0) {
          setFilePath(paths[0]);
          setOutPath(null);
          setProgress(0);
          setStatusMsg("");
        }
      }
    });

    return () => {
      unlistenOut.then(fn => fn());
      unlistenErr.then(fn => fn());
      unlistenDrop.then(fn => fn());
    };
  }, []);

  async function handleFileSelect() {
    const selected = await open({
      multiple: false,
      filters: [{
        name: 'Video',
        extensions: ['mp4', 'mov', 'avi']
      }]
    });
    if (selected) {
      setFilePath(selected as string);
      setOutPath(null);
      setProgress(0);
      setStatusMsg("");
    }
  }

  async function handleOutSelect() {
    const defaultOut = filePath ? filePath.replace(/\.[^/.]+$/, `_blurred${format}`) : `output${format}`;
    const selected = await save({
      defaultPath: defaultOut,
      filters: [{
        name: 'Video',
        extensions: [format.replace('.', '')]
      }]
    });
    if (selected) {
      setOutPath(selected);
    }
  }

  async function handleProcess() {
    if (!filePath) return;
    setIsProcessing(true);
    setProgress(0);
    setStatusMsg("Starting process...");

    let finalOut = outPath || filePath.replace(/\.[^/.]+$/, `_blurred${format}`);

    try {
      const payload = JSON.stringify({
        command: "process",
        input: filePath,
        output: finalOut,
        shutter_angle: shutterAngle
      });

      await invoke("start_processing", { inputJson: payload });
    } catch (e) {
      console.error(e);
      setStatusMsg("Failed to start processing.");
      setIsProcessing(false);
    }
  }

  async function handleCancel() {
    try {
      await invoke("cancel_processing");
      setIsProcessing(false);
      setStatusMsg("Process cancelled.");
      setProgress(0);
    } catch (e) {
      console.error(e);
    }
  }

  function preventDefaults(e: React.DragEvent) {
    e.preventDefault();
  }

  return (
    <main onDrop={preventDefaults} onDragOver={preventDefaults} onDragEnter={preventDefaults}>
      <div className="title">AI Motion Blur Generator</div>

      <div
        className="drop-zone"
        onClick={handleFileSelect}
      >
        <h3>Drop video file here</h3>
        <span style={{ color: "rgba(255,255,255,0.6)" }}>or click to browse files</span>
      </div>

      {filePath && (
        <div className="file-info">
          <div>Selected: {filePath.split(/[\/\\]/).pop()}</div>
          <div style={{ marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ color: "rgba(255,255,255,0.6)", fontSize: "0.8rem" }}>
              Output: {outPath ? outPath.split(/[\/\\]/).pop() : filePath.split(/[\/\\]/).pop()?.replace(/\.[^/.]+$/, `_blurred${format}`)}
            </span>
            <button
              className="btn"
              style={{ padding: '0.2rem 0.5rem', fontSize: '0.8rem', width: 'auto', marginTop: 0 }}
              onClick={handleOutSelect}
              disabled={isProcessing}
            >
              Choose Output...
            </button>
          </div>
        </div>
      )}

      <div className="settings">
        <div className="setting-group">
          <label>Shutter Angle: {shutterAngle}°</label>
          <input
            type="range"
            min="10"
            max="360"
            step="10"
            value={shutterAngle}
            onChange={(e) => {
              setShutterAngle(parseInt(e.target.value));
              setOutPath(null);
            }}
            disabled={isProcessing}
          />
        </div>

        <div className="setting-group">
          <label>Output Format</label>
          <select
            value={format}
            onChange={(e) => {
              setFormat(e.target.value);
              setOutPath(null);
            }}
            disabled={isProcessing}
          >
            <option value=".mp4">.mp4</option>
            <option value=".mov">.mov</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
        <button className="btn primary" onClick={handleProcess} disabled={!filePath || isProcessing}>
          {isProcessing ? "Processing..." : "Generate Motion Blur"}
        </button>
        {isProcessing && (
          <button className="btn" style={{ background: '#FF6B6B', marginTop: '1rem', padding: '1rem', fontWeight: 'bold' }} onClick={handleCancel}>
            Cancel
          </button>
        )}
      </div>

      {(isProcessing || progress > 0) && (
        <div className="progress-container">
          <span className="status-message">{statusMsg}</span>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;
