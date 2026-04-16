use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct AppState {
    child_process: Arc<Mutex<Option<CommandChild>>>,
}

/// Spawn the Python sidecar, send a JSON request via stdin,
/// and return the first stdout line as the response.
#[tauri::command]
async fn start_processing(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    input_json: String,
) -> Result<String, String> {
    let sidecar = app
        .shell()
        .sidecar("motion-blur-core")
        .map_err(|e| e.to_string())?;

    let (mut rx, mut child) = sidecar.spawn().map_err(|e| e.to_string())?;

    // Send the JSON request to the Python process's stdin
    child
        .write(format!("{}\n", input_json).as_bytes())
        .map_err(|e| e.to_string())?;

    // Store for cancellation
    {
        let mut lock = state.child_process.lock().unwrap();
        *lock = Some(child);
    }

    // Read events from the sidecar
    let mut output = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let line_str = String::from_utf8_lossy(&line);
                // Forward progress events to the frontend
                app.emit("sidecar-output", &*line_str).ok();
                output = line_str.to_string();
            }
            CommandEvent::Stderr(line) => {
                let err = String::from_utf8_lossy(&line);
                app.emit("sidecar-error", &*err).ok();
            }
            CommandEvent::Terminated(_) => break,
            _ => {}
        }
    }

    // Clear from state
    {
        let mut lock = state.child_process.lock().unwrap();
        *lock = None;
    }

    Ok(output)
}

#[tauri::command]
async fn cancel_processing(state: State<'_, AppState>) -> Result<(), String> {
    let mut lock = state.child_process.lock().unwrap();
    if let Some(child) = lock.take() {
        let _ = child.kill().map_err(|e| e.to_string());
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            child_process: Arc::new(Mutex::new(None)),
        })
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![start_processing, cancel_processing])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
