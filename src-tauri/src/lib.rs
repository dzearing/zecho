#[macro_use]
extern crate objc;

mod accessibility;
mod audio;
mod bark;
pub mod cleanup;
mod history;
mod hotkey;
mod macos_panel;
mod models;
pub mod settings;
mod transcribe;

use audio::AudioRecorder;
use cleanup::TextCleaner;
use history::{HistoryItem, HistoryStore};
use models::ModelStatus;
use settings::Settings;
use std::sync::Mutex;
use tauri::{Emitter, Manager};
use transcribe::Transcriber;

struct AppState {
    recorder: Mutex<AudioRecorder>,
    transcriber: Mutex<Transcriber>,
    cleaner: Mutex<TextCleaner>,
    history: Mutex<HistoryStore>,
    settings: Mutex<Settings>,
    pending_update: Mutex<Option<tauri_plugin_updater::Update>>,
    update_check_in_progress: std::sync::atomic::AtomicBool,
    tray_update_item: Mutex<Option<tauri::menu::MenuItem<tauri::Wry>>>,
    tray_icon: Mutex<Option<tauri::tray::TrayIcon<tauri::Wry>>>,
}

#[derive(Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum UpdateState {
    Checking,
    UpToDate { current_version: String },
    Available {
        version: String,
        current_version: String,
        notes: Option<String>,
        date_unix: Option<i64>,
    },
    Downloading { version: String, downloaded: u64, total: Option<u64> },
    Installing { version: String },
    Error { message: String },
}

#[tauri::command]
fn start_recording(state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    let recorder = state.recorder.lock().map_err(|e| e.to_string())?;
    recorder.start()?;
    register_escape_shortcut(&app);
    Ok(())
}

#[tauri::command]
fn get_audio_level(state: tauri::State<'_, AppState>) -> f32 {
    state
        .recorder
        .lock()
        .map(|r| r.rms_level())
        .unwrap_or(0.0)
}

#[tauri::command]
fn stop_recording(state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    unregister_escape_shortcut(&app);
    let recorder = state.recorder.lock().map_err(|e| e.to_string())?;
    let samples = recorder.stop();
    drop(recorder);

    if samples.is_empty() {
        return Err("No audio recorded".to_string());
    }

    let app_handle = app.clone();
    std::thread::Builder::new()
        .name("zecho-process".into())
        .spawn(move || {
            #[cfg(target_os = "macos")]
            unsafe {
                // QOS_CLASS_UTILITY = 0x11 — lower priority than user-interactive/initiated
                extern "C" {
                    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
                }
                pthread_set_qos_class_self_np(0x11, 0);
            }

            let state: tauri::State<'_, AppState> = app_handle.state();
            match process_recording(&state, samples) {
                Ok(text) => {
                    app_handle.emit("transcription-complete", &text).ok();
                }
                Err(e) => {
                    eprintln!("Processing error: {}", e);
                    app_handle.emit("transcription-error", &e).ok();
                }
            }
        })
        .ok();

    Ok(())
}

#[tauri::command]
fn cancel_recording(state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    unregister_escape_shortcut(&app);
    let recorder = state.recorder.lock().map_err(|e| e.to_string())?;
    recorder.stop();
    Ok(())
}

fn process_recording(state: &AppState, samples: Vec<f32>) -> Result<String, String> {
    use std::time::Instant;

    let language = state.settings.lock().map(|s| s.language.clone()).unwrap_or_else(|_| "en".to_string());

    let t0 = Instant::now();
    let transcriber = state.transcriber.lock().map_err(|e| e.to_string())?;
    let raw_text = transcriber.transcribe(&samples, &language)?;
    let stt_model = transcriber.model_id().unwrap_or_default().to_string();
    drop(transcriber);
    let transcribe_ms = t0.elapsed().as_millis() as u64;

    if raw_text.is_empty() || is_silence_hallucination(&raw_text, audio::loudest_window_rms(&samples)) {
        return Err("No speech detected".to_string());
    }

    let t1 = Instant::now();
    let (style, level, custom_prompt) = {
        let s = state.settings.lock().map_err(|e| e.to_string())?;
        (s.writing_style.clone(), s.cleanup_level.clone(), s.custom_prompt.clone())
    };
    let cleaner = state.cleaner.lock().map_err(|e| e.to_string())?;
    let cleanup_model = cleaner.model_id().unwrap_or_default().to_string();
    let text = match cleaner.clean(
        &raw_text,
        &style,
        &level,
        custom_prompt.as_deref(),
        &language,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Cleanup error: {}, using raw text", e);
            raw_text.clone()
        }
    };
    drop(cleaner);
    let cleanup_ms = t1.elapsed().as_millis() as u64;

    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard.set_text(&text).map_err(|e| e.to_string())?;

    let mut history = state.history.lock().map_err(|e| e.to_string())?;
    history.add(text.clone(), raw_text, transcribe_ms, cleanup_ms, stt_model, cleanup_model);
    drop(history);

    let auto_paste = state.settings.lock().map(|s| s.auto_paste).unwrap_or(false);
    if auto_paste {
        simulate_paste();
    }

    Ok(text)
}

/// Speech models hear a lone "Yeah." or "Okay." in silence or room noise. Loudness alone can't
/// tell that apart from a softly spoken word (quiet speech measured below some silent
/// recordings), so only a quiet recording that produced nothing but a filler counts as silence.
fn is_silence_hallucination(text: &str, loudest_rms: f32) -> bool {
    const FILLERS: &[&str] = &[
        "yeah", "okay", "ok", "hmm", "hm", "mm", "mhm", "uh", "um", "ah", "oh", "you", "thank you", "thanks", "bye",
    ];
    // A normal sentence measured 0.018; silent and noisy recordings measured up to 0.011.
    const QUIET_RMS: f32 = 0.015;
    let words = text.trim().trim_end_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    loudest_rms < QUIET_RMS && FILLERS.contains(&words.as_str())
}

fn simulate_paste() {
    #[cfg(target_os = "macos")]
    {
        use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
        use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let source = match CGEventSource::new(CGEventSourceStateID::HIDSystemState) {
                Ok(s) => s,
                Err(_) => return,
            };
            // 'v' keycode = 9
            let key_v: CGKeyCode = 9;
            if let Ok(event) = CGEvent::new_keyboard_event(source.clone(), key_v, true) {
                event.set_flags(CGEventFlags::CGEventFlagCommand);
                event.post(core_graphics::event::CGEventTapLocation::HID);
            }
            if let Ok(event) = CGEvent::new_keyboard_event(source, key_v, false) {
                event.set_flags(CGEventFlags::CGEventFlagCommand);
                event.post(core_graphics::event::CGEventTapLocation::HID);
            }
        });
    }
}

#[tauri::command]
fn get_history(state: tauri::State<'_, AppState>) -> Result<Vec<HistoryItem>, String> {
    let history = state.history.lock().map_err(|e| e.to_string())?;
    Ok(history.items())
}

#[tauri::command]
fn copy_history_item(id: String, state: tauri::State<'_, AppState>) -> Result<String, String> {
    let history = state.history.lock().map_err(|e| e.to_string())?;
    let item = history.get(&id).ok_or("Item not found")?.clone();
    drop(history);
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    clipboard.set_text(&item.text).map_err(|e| e.to_string())?;
    Ok(item.text)
}

#[tauri::command]
fn delete_history_item(id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut history = state.history.lock().map_err(|e| e.to_string())?;
    history.delete(&id);
    Ok(())
}

#[tauri::command]
fn clear_history(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut history = state.history.lock().map_err(|e| e.to_string())?;
    history.clear();
    Ok(())
}

#[tauri::command]
fn open_settings(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("settings") {
        window.show().map_err(|e| e.to_string())?;
        window.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn get_settings(state: tauri::State<'_, AppState>) -> Result<Settings, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    Ok(settings.clone())
}

#[tauri::command]
fn update_settings(new_settings: Settings, state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    let path = settings.path.clone();
    *settings = new_settings;
    settings.path = path;
    hotkey::set_fn_needs_shift(settings.fn_key_needs_shift);
    settings.save()?;
    app.emit("settings-changed", &*settings).ok();
    Ok(())
}

#[tauri::command]
fn get_autostart_enabled(app: tauri::AppHandle) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch().is_enabled().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_autostart_enabled(enabled: bool, app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let manager = app.autolaunch();
    if enabled {
        manager.enable().map_err(|e| e.to_string())
    } else {
        manager.disable().map_err(|e| e.to_string())
    }
}

#[tauri::command]
fn list_models() -> Vec<ModelStatus> {
    models::list_models()
}

#[tauri::command]
fn download_model(model_id: String, app: tauri::AppHandle) -> Result<(), String> {
    let info = models::get_model(&model_id).ok_or("Unknown model")?;

    std::thread::spawn({
        let info = info.clone();
        let app = app.clone();
        move || {
            eprintln!("Starting download: {} ({})", info.name, info.url);
            app.emit("model-download-started", &info.id).ok();
            let app2 = app.clone();
            let id = info.id.to_string();
            let mut last_pct: u64 = 0;
            match models::download_model_blocking(&info, move |downloaded, total| {
                let pct = if total > 0 { downloaded * 100 / total } else { 0 };
                if pct != last_pct {
                    last_pct = pct;
                    app2.emit("model-download-progress", (&id, pct)).ok();
                }
            }) {
                Ok(path) => {
                    eprintln!("Download complete: {}", info.name);
                    if info.engine == models::Engine::Bark {
                        // The first load specializes the encoder for this Mac's Neural Engine, which
                        // takes minutes; do it now so selecting the model later is instant.
                        app.emit("model-download-preparing", &info.id).ok();
                        let t0 = std::time::Instant::now();
                        match bark::BarkModel::load(&path) {
                            Ok(_) => eprintln!("Bark prepared in {:.1}s", t0.elapsed().as_secs_f64()),
                            Err(e) => {
                                eprintln!("Bark prepare failed: {}", e);
                                app.emit("model-download-error", (&info.id, &e)).ok();
                                return;
                            }
                        }
                    }
                    if info.model_type == models::ModelType::Cleanup {
                        let state = app.state::<AppState>();
                        let active = state.settings.lock().map(|s| s.active_cleanup_model.clone()).unwrap_or_default();
                        if let Ok(mut c) = state.cleaner.lock() {
                            if active == info.id && c.model_id() != Some(info.id) {
                                c.start_worker(info.id, &path).ok();
                            }
                        };
                    }
                    app.emit("model-download-complete", &info.id).ok();
                }
                Err(e) => {
                    eprintln!("Download failed: {} - {}", info.name, e);
                    app.emit("model-download-error", (&info.id, &e)).ok();
                }
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn load_cleanup_model(model_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let info = models::get_model(&model_id).ok_or("Unknown model")?;
    let path = models::model_path(info);
    if !path.exists() {
        return Err(format!("Model not downloaded: {}", info.name));
    }
    let mut cleaner = state.cleaner.lock().map_err(|e| e.to_string())?;
    cleaner.start_worker(info.id, &path)
}

#[tauri::command]
fn load_whisper_model_cmd(model_id: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let info = models::get_model(&model_id).ok_or("Unknown model")?;
    let path = models::model_path(info);
    if !models::is_downloaded(info) {
        return Err(format!("Model not downloaded: {}", info.name));
    }
    let engine = transcribe::open_engine(&path)?;
    state.transcriber.lock().map_err(|e| e.to_string())?.set_engine(engine, info.id);
    Ok(())
}

#[tauri::command]
fn update_hotkey(hotkey: String, state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

    let old_hotkey = state.settings.lock().map(|s| s.hotkey.clone()).unwrap_or_default();
    if let Some((mods, code)) = parse_hotkey(&old_hotkey) {
        let old_shortcut = Shortcut::new(mods, code);
        app.global_shortcut().unregister(old_shortcut).ok();
    }

    let (mods, code) = parse_hotkey(&hotkey).ok_or("Invalid hotkey")?;
    let new_shortcut = Shortcut::new(mods, code);
    app.global_shortcut().on_shortcut(new_shortcut, |app_handle, _shortcut, event| {
        // The plugin forwards both key-down and key-up. Toggling on both turns the
        // shortcut into push-to-talk, so only act on the press.
        if event.state == ShortcutState::Pressed {
            let _ = app_handle.emit("toggle-recording", ());
        }
    }).map_err(|e| e.to_string())?;

    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.hotkey = hotkey;
    settings.save()
}

/// Bump when there's a new "What's new" dialog to show to people who update.
pub(crate) const WHATS_NEW_REVISION: u32 = 1;

#[derive(serde::Serialize)]
struct WhatsNewOffer {
    /// "speech" or "cleanup"
    kind: &'static str,
    model_id: &'static str,
    name: &'static str,
    size_mb: u32,
    downloaded: bool,
    current_name: String,
}

/// New default models the user isn't using yet and could switch to.
fn whats_new_offers(settings: &Settings) -> Vec<WhatsNewOffer> {
    let mut offers = Vec::new();
    let name_of = |id: &str| models::get_model(id).map(|m| m.name.to_string()).unwrap_or_else(|| id.to_string());

    let speech = models::default_whisper_model();
    if settings.language == "en" && speech.engine == models::Engine::Bark && settings.active_whisper_model != speech.id {
        offers.push(WhatsNewOffer {
            kind: "speech",
            model_id: speech.id,
            name: speech.name,
            size_mb: speech.size_mb,
            downloaded: models::is_downloaded(speech),
            current_name: name_of(&settings.active_whisper_model),
        });
    }

    let cleanup = models::default_cleanup_model(&settings.language);
    if cleanup.id.starts_with("spokenly") && settings.active_cleanup_model != cleanup.id {
        offers.push(WhatsNewOffer {
            kind: "cleanup",
            model_id: cleanup.id,
            name: cleanup.name,
            size_mb: cleanup.size_mb,
            downloaded: models::is_downloaded(cleanup),
            current_name: name_of(&settings.active_cleanup_model),
        });
    }
    offers
}

#[tauri::command]
fn get_whats_new(state: tauri::State<'_, AppState>) -> Result<Vec<WhatsNewOffer>, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    Ok(whats_new_offers(&settings))
}

fn mark_whats_new_seen(app: &tauri::AppHandle) {
    let state = app.state::<AppState>();
    if let Ok(mut s) = state.settings.lock() {
        if s.whats_new_seen < WHATS_NEW_REVISION {
            s.whats_new_seen = WHATS_NEW_REVISION;
            s.save().ok();
        }
    };
}

#[tauri::command]
fn dismiss_whats_new(app: tauri::AppHandle) {
    mark_whats_new_seen(&app);
    if let Some(win) = app.get_webview_window("whatsnew") {
        win.hide().ok();
    }
}

/// Makes a downloaded speech or cleanup model the active one and saves the choice.
#[tauri::command]
fn use_model(model_id: String, state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    let info = models::get_model(&model_id).ok_or("Unknown model")?;
    if !models::is_downloaded(info) {
        return Err(format!("Model not downloaded: {}", info.name));
    }
    let path = models::model_path(info);
    match info.model_type {
        models::ModelType::Whisper => {
            let engine = transcribe::open_engine(&path)?;
            state.transcriber.lock().map_err(|e| e.to_string())?.set_engine(engine, info.id);
        }
        models::ModelType::Cleanup => {
            state.cleaner.lock().map_err(|e| e.to_string())?.start_worker(info.id, &path)?;
        }
    }
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    match info.model_type {
        models::ModelType::Whisper => settings.active_whisper_model = info.id.to_string(),
        models::ModelType::Cleanup => settings.active_cleanup_model = info.id.to_string(),
    }
    settings.save()?;
    app.emit("settings-changed", &*settings).ok();
    Ok(())
}

#[tauri::command]
fn switch_language(language: String, state: tauri::State<'_, AppState>) -> Result<LanguageSwitch, String> {
    let (current_whisper, current_cleanup) = {
        let s = state.settings.lock().map_err(|e| e.to_string())?;
        (s.active_whisper_model.clone(), s.active_cleanup_model.clone())
    };

    let mut result = LanguageSwitch { whisper: None, cleanup: None };

    let target_id = models::whisper_model_for_language(&current_whisper, &language);
    if target_id != current_whisper {
        let info = models::get_model(&target_id).ok_or("Unknown model")?;
        if models::is_downloaded(info) {
            let engine = transcribe::open_engine(&models::model_path(info))?;
            state.transcriber.lock().map_err(|e| e.to_string())?.set_engine(engine, info.id);
        }
        result.whisper = Some(target_id);
    }

    // Keep the cleanup model if it handles the language; otherwise use a downloaded one that does,
    // or the language's default (which the UI then downloads; the worker switches when it lands).
    let current = models::get_model(&current_cleanup);
    if !current.map(|m| models::cleanup_supports(m, &language)).unwrap_or(false) {
        let candidates = models::cleanup_model_candidates(&current_cleanup, &language);
        let target = candidates
            .iter()
            .copied()
            .find(|m| models::is_downloaded(m))
            .unwrap_or_else(|| models::default_cleanup_model(&language));
        if models::is_downloaded(target) {
            state.cleaner.lock().map_err(|e| e.to_string())?.start_worker(target.id, &models::model_path(target))?;
        }
        state.settings.lock().map_err(|e| e.to_string())?.active_cleanup_model = target.id.to_string();
        result.cleanup = Some(target.id.to_string());
    }

    Ok(result)
}

#[derive(serde::Serialize)]
struct LanguageSwitch {
    whisper: Option<String>,
    cleanup: Option<String>,
}

#[tauri::command]
fn save_pill_position(x_pct: f64, y_pct: f64, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.pill_x_pct = Some(x_pct);
    settings.pill_y_pct = Some(y_pct);
    settings.save()
}

#[tauri::command]
fn start_drag(window: tauri::WebviewWindow) -> Result<(), String> {
    window.start_dragging().map_err(|e| e.to_string())
}

#[tauri::command]
fn persist_pill_position(app: tauri::AppHandle, state: tauri::State<'_, AppState>) {
    if let Some(window) = app.get_webview_window("pill") {
        save_pill_position_from_window(&window, &state);
    }
}

#[tauri::command]
fn check_accessibility() -> bool {
    accessibility::is_accessibility_enabled()
}

#[tauri::command]
fn open_accessibility_settings() {
    accessibility::prompt_accessibility();
}

#[tauri::command]
fn start_fn_listener(app: tauri::AppHandle) {
    if accessibility::is_accessibility_enabled() {
        hotkey::start_fn_key_listener(app);
    }
}

#[tauri::command]
fn check_microphone() -> bool {
    accessibility::is_microphone_enabled()
}

#[tauri::command]
fn open_mic_settings() {
    accessibility::open_mic_settings();
}

#[tauri::command]
fn request_microphone() {
    accessibility::request_microphone();
}

#[derive(serde::Serialize, Clone)]
struct SetupStatus {
    whisper_ready: bool,
    cleanup_ready: bool,
    whisper_downloading: bool,
    cleanup_downloading: bool,
}

#[tauri::command]
fn check_setup(state: tauri::State<'_, AppState>) -> SetupStatus {
    let active_id = state.settings.lock().map(|s| s.active_whisper_model.clone()).unwrap_or_default();
    let whisper_ready = models::speech_model_candidates(&active_id).into_iter().any(models::is_downloaded);
    let language = state.settings.lock().map(|s| s.language.clone()).unwrap_or_else(|_| "en".to_string());
    let cleanup_ready = models::cleanup_model_candidates("", &language).into_iter().any(models::is_downloaded);

    SetupStatus {
        whisper_ready,
        cleanup_ready,
        whisper_downloading: false,
        cleanup_downloading: false,
    }
}

#[tauri::command]
fn setup_download_models(app: tauri::AppHandle) {
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let active_id = state.settings.lock().map(|s| s.active_whisper_model.clone()).unwrap_or_default();
        if !models::speech_model_candidates(&active_id).into_iter().any(models::is_downloaded) {
            // Try the default (Bark where supported); if it can't be downloaded or prepared, use Whisper base.en.
            let mut fallbacks = vec![models::default_whisper_model()];
            fallbacks.extend(models::get_model("whisper-base-en").filter(|m| m.id != fallbacks[0].id));
            let mut last_err = String::new();
            let mut ready = false;
            for info in fallbacks {
                app.emit("setup-progress", "Downloading speech model...").ok();
                let path = match models::download_model_blocking(info, |_, _| {}) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("Setup: download of {} failed: {}", info.id, e);
                        last_err = e;
                        continue;
                    }
                };
                if info.engine == models::Engine::Bark {
                    app.emit("setup-progress", "Preparing speech model for this Mac (a few minutes)...").ok();
                }
                match transcribe::open_engine(&path) {
                    Ok(engine) => {
                        if let Ok(mut t) = state.transcriber.lock() {
                            t.set_engine(engine, info.id);
                        }
                        if let Ok(mut s) = state.settings.lock() {
                            s.active_whisper_model = info.id.to_string();
                            s.save().ok();
                        }
                        app.emit("setup-progress", "Speech model ready").ok();
                        ready = true;
                        break;
                    }
                    Err(e) => {
                        eprintln!("Setup: loading {} failed: {}", info.id, e);
                        last_err = e;
                    }
                }
            }
            if !ready {
                app.emit("setup-error", &format!("Failed to download speech model: {}", last_err)).ok();
                return;
            }
        }

        let (active_cleanup, language) = state
            .settings
            .lock()
            .map(|s| (s.active_cleanup_model.clone(), s.language.clone()))
            .unwrap_or_default();
        if !models::cleanup_model_candidates(&active_cleanup, &language).into_iter().any(models::is_downloaded) {
            let info = models::default_cleanup_model(&language);
            app.emit("setup-progress", "Downloading cleanup model...").ok();
            match models::download_model_blocking(info, |_, _| {}) {
                Ok(_) => {
                    app.emit("setup-progress", "Cleanup model ready").ok();
                }
                Err(e) => {
                    app.emit("setup-error", &format!("Failed to download cleanup model: {}", e)).ok();
                    return;
                }
            }
        }

        // Start the cleanup worker
        if let Some(info) = models::cleanup_model_candidates(&active_cleanup, &language).into_iter().find(|m| models::is_downloaded(m)) {
            let mut c = state.cleaner.lock().unwrap();
            c.start_worker(info.id, &models::model_path(info)).ok();
            drop(c);
            if let Ok(mut s) = state.settings.lock() {
                s.active_cleanup_model = info.id.to_string();
                s.save().ok();
            }
        }

        app.emit("setup-complete", ()).ok();
    });
}

#[tauri::command]
fn toggle_history(app: tauri::AppHandle) {
    if let Some(hist) = app.get_webview_window("history") {
        if hist.is_visible().unwrap_or(false) {
            hist.hide().ok();
            return;
        }

        if let Some(pill) = app.get_webview_window("pill") {
            if let (Ok(pill_pos), Ok(pill_size), Ok(hist_size)) = (
                pill.outer_position(),
                pill.outer_size(),
                hist.outer_size(),
            ) {
                let hist_w = hist_size.width as i32;
                let hist_h = hist_size.height as i32;
                let pill_cx = pill_pos.x + pill_size.width as i32 / 2;
                let mut x = pill_cx - hist_w / 2;
                let mut y = pill_pos.y - hist_h - 8;

                if let Ok(Some(monitor)) = pill.current_monitor() {
                    let screen = monitor.size();
                    let screen_pos = monitor.position();
                    let sw = screen.width as i32;
                    let sh = screen.height as i32;
                    if x < screen_pos.x { x = screen_pos.x; }
                    if x + hist_w > screen_pos.x + sw { x = screen_pos.x + sw - hist_w; }
                    if y < screen_pos.y { y = screen_pos.y; }
                    if y + hist_h > screen_pos.y + sh { y = screen_pos.y + sh - hist_h; }
                }

                hist.set_position(tauri::Position::Physical(
                    tauri::PhysicalPosition { x, y },
                )).ok();
            }
        }

        hist.show().ok();
        hist.set_focus().ok();
    }
}

#[tauri::command]
fn hide_setup(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("setup") {
        window.hide().map_err(|e| e.to_string())?;
    }
    if let Some(pill) = app.get_webview_window("pill") {
        pill.show().ok();
    }
    Ok(())
}

#[tauri::command]
fn complete_setup(state: tauri::State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.setup_complete = true;
    settings.save()?;
    drop(settings);

    if let Some(pill) = app.get_webview_window("pill") {
        pill.show().ok();
    }

    Ok(())
}

fn emit_update_state(app: &tauri::AppHandle, state: UpdateState) {
    app.emit_to("update", "update-state", state).ok();
}

#[tauri::command]
fn close_update_window(app: tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("update") {
        win.hide().ok();
    }
}

fn update_tray_label(app: &tauri::AppHandle) {
    let pending_version = app
        .state::<AppState>()
        .pending_update
        .lock()
        .ok()
        .and_then(|p| p.as_ref().map(|u| u.version.clone()));

    if let Ok(item_lock) = app.state::<AppState>().tray_update_item.lock() {
        if let Some(item) = item_lock.as_ref() {
            match &pending_version {
                Some(v) => { item.set_text(format!("Install Zecho v{}", v)).ok(); }
                None => { item.set_text("Check for Updates…").ok(); }
            }
        }
    }

    if let Ok(icon_lock) = app.state::<AppState>().tray_icon.lock() {
        if let Some(tray) = icon_lock.as_ref() {
            if pending_version.is_some() {
                tray.set_title(Some("↑")).ok();
            } else {
                tray.set_title(None::<&str>).ok();
            }
        }
    }
}

async fn perform_update_check(app: tauri::AppHandle, emit_to_dialog: bool) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    use tauri_plugin_updater::UpdaterExt;

    if app
        .state::<AppState>()
        .update_check_in_progress
        .swap(true, Ordering::Acquire)
    {
        return Ok(());
    }

    struct ResetGuard(tauri::AppHandle);
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            self.0
                .state::<AppState>()
                .update_check_in_progress
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }
    let _guard = ResetGuard(app.clone());

    if emit_to_dialog {
        emit_update_state(&app, UpdateState::Checking);
    }
    let current_version = app.package_info().version.to_string();
    let result = app
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await;
    match result {
        Ok(Some(update)) => {
            let version = update.version.clone();
            let notes = update.body.clone();
            let date_unix = update.date.map(|d| d.unix_timestamp());
            if let Ok(mut pending) = app.state::<AppState>().pending_update.lock() {
                *pending = Some(update);
            }
            update_tray_label(&app);
            if emit_to_dialog {
                emit_update_state(
                    &app,
                    UpdateState::Available { version, current_version, notes, date_unix },
                );
            }
        }
        Ok(None) => {
            if let Ok(mut pending) = app.state::<AppState>().pending_update.lock() {
                *pending = None;
            }
            update_tray_label(&app);
            if emit_to_dialog {
                emit_update_state(&app, UpdateState::UpToDate { current_version });
            }
        }
        Err(e) => {
            if emit_to_dialog {
                emit_update_state(&app, UpdateState::Error { message: e.to_string() });
            } else {
                eprintln!("Background update check failed: {}", e);
            }
        }
    }
    Ok(())
}

#[tauri::command]
async fn start_update_check(app: tauri::AppHandle) -> Result<(), String> {
    perform_update_check(app, true).await
}

#[tauri::command]
async fn open_update_dialog(app: tauri::AppHandle) -> Result<(), String> {
    let pending_info = {
        let state = app.state::<AppState>();
        let pending_lock = state.pending_update.lock().map_err(|e| e.to_string())?;
        pending_lock.as_ref().map(|u| {
            (
                u.version.clone(),
                u.body.clone(),
                u.date.map(|d| d.unix_timestamp()),
            )
        })
    };
    if let Some((version, notes, date_unix)) = pending_info {
        let current_version = app.package_info().version.to_string();
        emit_update_state(
            &app,
            UpdateState::Available { version, current_version, notes, date_unix },
        );
        Ok(())
    } else {
        perform_update_check(app, true).await
    }
}

async fn background_update_loop(app: tauri::AppHandle) {
    use std::time::Duration;
    tokio::time::sleep(Duration::from_secs(10)).await;
    loop {
        let _ = perform_update_check(app.clone(), false).await;
        tokio::time::sleep(Duration::from_secs(24 * 60 * 60)).await;
    }
}

#[tauri::command]
async fn install_pending_update(app: tauri::AppHandle) -> Result<(), String> {
    let update = {
        let state = app.state::<AppState>();
        let mut pending = state.pending_update.lock().map_err(|e| e.to_string())?;
        pending.take()
    };
    let Some(update) = update else {
        return Err("No pending update".into());
    };
    update_tray_label(&app);
    let version = update.version.clone();
    emit_update_state(
        &app,
        UpdateState::Downloading { version: version.clone(), downloaded: 0, total: None },
    );

    let downloaded = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let progress_app = app.clone();
    let progress_version = version.clone();
    let progress_downloaded = downloaded.clone();
    let finish_app = app.clone();
    let finish_version = version.clone();

    let result = update
        .download_and_install(
            move |chunk, total| {
                let current = progress_downloaded
                    .fetch_add(chunk as u64, std::sync::atomic::Ordering::Relaxed)
                    + chunk as u64;
                emit_update_state(
                    &progress_app,
                    UpdateState::Downloading {
                        version: progress_version.clone(),
                        downloaded: current,
                        total,
                    },
                );
            },
            move || {
                emit_update_state(
                    &finish_app,
                    UpdateState::Installing { version: finish_version.clone() },
                );
            },
        )
        .await;

    match result {
        Ok(_) => app.restart(),
        Err(e) => {
            emit_update_state(&app, UpdateState::Error { message: e.to_string() });
            Err(e.to_string())
        }
    }
}

fn create_tray_icon(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::TrayIconBuilder;

    let icon = {
        let png_data = include_bytes!("../icons/tray_icon.png");
        let decoder = png::Decoder::new(std::io::Cursor::new(png_data));
        let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
        buf.truncate(info.buffer_size());
        tauri::image::Image::new_owned(buf, info.width, info.height)
    };

    let history = MenuItem::with_id(app, "history", "Show History", true, None::<&str>)?;
    let settings_item = MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
    let check_updates = MenuItem::with_id(app, "check_updates", "Check for Updates…", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Zecho", true, None::<&str>)?;
    let sep1 = tauri::menu::PredefinedMenuItem::separator(app)?;
    let sep2 = tauri::menu::PredefinedMenuItem::separator(app)?;
    let menu = Menu::with_items(app, &[
        &history,
        &sep1,
        &settings_item,
        &check_updates,
        &sep2,
        &quit,
    ])?;

    if let Ok(mut slot) = app.state::<AppState>().tray_update_item.lock() {
        *slot = Some(check_updates.clone());
    }

    let tray = TrayIconBuilder::new()
        .icon(icon)
        .icon_as_template(true)
        .menu(&menu)
        .tooltip("Zecho")
        .on_menu_event(|app, event| match event.id().as_ref() {
            "history" => {
                if let Some(hist) = app.get_webview_window("history") {
                    if !hist.is_visible().unwrap_or(false) {
                        // Position above pill
                        if let Some(pill) = app.get_webview_window("pill") {
                            if let (Ok(pill_pos), Ok(pill_size), Ok(hist_size)) = (
                                pill.outer_position(), pill.outer_size(), hist.outer_size(),
                            ) {
                                let hist_w = hist_size.width as i32;
                                let hist_h = hist_size.height as i32;
                                let pill_cx = pill_pos.x + pill_size.width as i32 / 2;
                                let mut x = pill_cx - hist_w / 2;
                                let mut y = pill_pos.y - hist_h - 8;
                                if let Ok(Some(monitor)) = pill.current_monitor() {
                                    let screen = monitor.size();
                                    let sp = monitor.position();
                                    let sw = screen.width as i32;
                                    let sh = screen.height as i32;
                                    if x < sp.x { x = sp.x; }
                                    if x + hist_w > sp.x + sw { x = sp.x + sw - hist_w; }
                                    if y < sp.y { y = sp.y; }
                                    if y + hist_h > sp.y + sh { y = sp.y + sh - hist_h; }
                                }
                                hist.set_position(tauri::Position::Physical(
                                    tauri::PhysicalPosition { x, y },
                                )).ok();
                            }
                        }
                        hist.show().ok();
                    }
                    hist.set_focus().ok();
                }
            }
            "settings" => {
                if let Some(window) = app.get_webview_window("settings") {
                    window.show().ok();
                    window.set_focus().ok();
                }
            }
            "check_updates" => {
                if let Some(window) = app.get_webview_window("update") {
                    window.show().ok();
                    window.set_focus().ok();
                    app.emit_to("update", "update-window-opened", ()).ok();
                }
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;

    if let Ok(mut slot) = app.state::<AppState>().tray_icon.lock() {
        *slot = Some(tray);
    }

    Ok(())
}

fn save_pill_position_from_window(window: &tauri::WebviewWindow, state: &AppState) {
    let pos = match window.outer_position() {
        Ok(p) => p,
        _ => return,
    };
    let size = match window.outer_size() {
        Ok(s) => s,
        _ => return,
    };
    let monitor = match window.current_monitor() {
        Ok(Some(m)) => m,
        _ => return,
    };
    let screen = monitor.size();
    if screen.width == 0 || screen.height == 0 {
        return;
    }
    let x_pct = pos.x as f64 / screen.width as f64;
    let y_pct = (pos.y as f64 + size.height as f64) / screen.height as f64;

    if let Ok(mut settings) = state.settings.lock() {
        settings.pill_x_pct = Some(x_pct);
        settings.pill_y_pct = Some(y_pct);
        settings.save().ok();
    }
}

fn get_visible_frame() -> Option<(f64, f64, f64, f64)> {
    use cocoa::appkit::NSScreen;
    use cocoa::base::nil;
    use cocoa::foundation::NSRect;
    unsafe {
        let screen = NSScreen::mainScreen(nil);
        if screen == nil {
            return None;
        }
        let frame: NSRect = msg_send![screen, visibleFrame];
        Some((frame.origin.x, frame.origin.y, frame.size.width, frame.size.height))
    }
}

fn position_pill_window(window: &tauri::WebviewWindow, state: &AppState) {
    let monitor = match window.primary_monitor() {
        Ok(Some(m)) => m,
        _ => match window.current_monitor() {
            Ok(Some(m)) => m,
            _ => return,
        },
    };
    let screen = monitor.size();
    let scale = monitor.scale_factor();
    let win_h = window.outer_size().map(|s| s.height as i32).unwrap_or((46.0 * scale) as i32);
    let win_w = window.outer_size().map(|s| s.width as i32).unwrap_or((220.0 * scale) as i32);

    let settings = state.settings.lock().ok();
    let saved = settings.as_ref().and_then(|s| {
        match (s.pill_x_pct, s.pill_y_pct) {
            (Some(xp), Some(yp)) if xp >= 0.0 && xp <= 1.5 && yp >= 0.0 && yp <= 1.5 => {
                Some((xp, yp))
            }
            _ => None,
        }
    });

    let (x, y) = if let Some((x_pct, y_pct)) = saved {
        let x = (x_pct * screen.width as f64) as i32;
        let bottom_y = (y_pct * screen.height as f64) as i32;
        (x, bottom_y - win_h)
    } else if let Some((vf_x, vf_y, vf_w, _)) = get_visible_frame() {
        // visibleFrame is in AppKit coords (origin at bottom-left).
        // Convert to screen coords (origin at top-left) for Tauri positioning.
        let screen_h = screen.height as f64 / scale;
        let safe_bottom = vf_y;
        let margin = 8.0;
        let x = ((vf_x + vf_w * 0.75) * scale) as i32 - win_w / 2;
        let y = ((screen_h - safe_bottom - margin) * scale) as i32 - win_h;
        (x, y)
    } else {
        let x = (screen.width as i32 - win_w) / 2;
        let y = screen.height as i32 - win_h - (60.0 * scale) as i32;
        (x, y)
    };

    window
        .set_position(tauri::Position::Physical(tauri::PhysicalPosition { x, y }))
        .ok();
}

fn parse_hotkey(hotkey: &str) -> Option<(Option<tauri_plugin_global_shortcut::Modifiers>, tauri_plugin_global_shortcut::Code)> {
    use tauri_plugin_global_shortcut::{Code, Modifiers};

    // Split on '+' but keep a trailing literal '+' as the key itself (e.g. "cmd+shift++").
    let parts: Vec<String> = if hotkey.ends_with("+") && hotkey.len() > 1 {
        let mut p: Vec<String> = hotkey[..hotkey.len() - 1]
            .split('+')
            .map(|s| s.trim().to_lowercase())
            .collect();
        p.push("+".to_string());
        p
    } else {
        hotkey.split('+').map(|s| s.trim().to_lowercase()).collect()
    };

    let mut mods = Modifiers::empty();
    let mut key_part = String::new();

    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            key_part = part.clone();
        } else {
            match part.as_str() {
                "ctrl" | "control" => mods |= Modifiers::CONTROL,
                "alt" | "option" | "opt" => mods |= Modifiers::ALT,
                "shift" => mods |= Modifiers::SHIFT,
                "cmd" | "command" | "meta" | "super" => mods |= Modifiers::META,
                _ => return None,
            }
        }
    }

    let code = match key_part.as_str() {
        "space" => Code::Space,
        "a" => Code::KeyA, "b" => Code::KeyB, "c" => Code::KeyC, "d" => Code::KeyD,
        "e" => Code::KeyE, "f" => Code::KeyF, "g" => Code::KeyG, "h" => Code::KeyH,
        "i" => Code::KeyI, "j" => Code::KeyJ, "k" => Code::KeyK, "l" => Code::KeyL,
        "m" => Code::KeyM, "n" => Code::KeyN, "o" => Code::KeyO, "p" => Code::KeyP,
        "q" => Code::KeyQ, "r" => Code::KeyR, "s" => Code::KeyS, "t" => Code::KeyT,
        "u" => Code::KeyU, "v" => Code::KeyV, "w" => Code::KeyW, "x" => Code::KeyX,
        "y" => Code::KeyY, "z" => Code::KeyZ,
        "0" => Code::Digit0, "1" => Code::Digit1, "2" => Code::Digit2, "3" => Code::Digit3,
        "4" => Code::Digit4, "5" => Code::Digit5, "6" => Code::Digit6, "7" => Code::Digit7,
        "8" => Code::Digit8, "9" => Code::Digit9,
        "f1" => Code::F1, "f2" => Code::F2, "f3" => Code::F3, "f4" => Code::F4,
        "f5" => Code::F5, "f6" => Code::F6, "f7" => Code::F7, "f8" => Code::F8,
        "f9" => Code::F9, "f10" => Code::F10, "f11" => Code::F11, "f12" => Code::F12,

        // Punctuation, by KeyboardEvent.code name and by the unshifted literal.
        "backslash" | "\\" => Code::Backslash,
        "bracketleft" | "[" => Code::BracketLeft,
        "bracketright" | "]" => Code::BracketRight,
        "semicolon" | ";" => Code::Semicolon,
        "quote" | "'" => Code::Quote,
        "comma" | "," => Code::Comma,
        "period" | "." => Code::Period,
        "slash" | "/" => Code::Slash,
        "backquote" | "`" => Code::Backquote,
        "minus" | "-" => Code::Minus,
        "equal" | "=" | "+" => Code::Equal,

        // Named non-printing keys.
        "arrowup" | "up" => Code::ArrowUp,
        "arrowdown" | "down" => Code::ArrowDown,
        "arrowleft" | "left" => Code::ArrowLeft,
        "arrowright" | "right" => Code::ArrowRight,
        "enter" | "return" => Code::Enter,
        "tab" => Code::Tab,
        "backspace" => Code::Backspace,
        "delete" => Code::Delete,
        "home" => Code::Home,
        "end" => Code::End,
        "pageup" => Code::PageUp,
        "pagedown" => Code::PageDown,
        "insert" => Code::Insert,
        _ => return None,
    };

    let mod_opt = if mods.is_empty() { None } else { Some(mods) };
    Some((mod_opt, code))
}

fn register_global_shortcut(app: &tauri::App) {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

    let state: tauri::State<'_, AppState> = app.state();
    let hotkey = state.settings.lock().map(|s| s.hotkey.clone()).unwrap_or_else(|_| "alt+space".to_string());

    match parse_hotkey(&hotkey) {
        Some((mods, code)) => {
            let shortcut = Shortcut::new(mods, code);
            match app.global_shortcut().on_shortcut(shortcut, |app_handle, _shortcut, event| {
                // Only the press toggles; the matching release must be ignored or the
                // shortcut degrades into push-to-talk.
                // Only the press toggles; the matching release must be ignored or the
                // shortcut degrades into push-to-talk.
                if event.state == ShortcutState::Pressed {
                    let _ = app_handle.emit("toggle-recording", ());
                }
            }) {
                Ok(()) => eprintln!("zecho: registered global hotkey {hotkey}"),
                Err(e) => eprintln!("zecho: failed to register global hotkey {hotkey}: {e}"),
            }
        }
        None => eprintln!("zecho: unsupported hotkey in settings: {hotkey}"),
    }
}

fn register_escape_shortcut(app: &tauri::AppHandle) {
    use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};

    let esc = Shortcut::new(None, Code::Escape);
    if !app.global_shortcut().is_registered(esc) {
        app.global_shortcut().on_shortcut(esc, |app_handle, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                let _ = app_handle.emit("cancel-recording", ());
            }
        }).ok();
    }
}

fn unregister_escape_shortcut(app: &tauri::AppHandle) {
    use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut};

    let esc = Shortcut::new(None, Code::Escape);
    app.global_shortcut().unregister(esc).ok();
}

fn init_models_async(app_handle: tauri::AppHandle) {
    std::thread::spawn(move || {
        let state = app_handle.state::<AppState>();
        let active_id = state.settings.lock().map(|s| s.active_whisper_model.clone()).unwrap_or_default();
        let candidates: Vec<_> = models::speech_model_candidates(&active_id)
            .into_iter()
            .filter(|m| models::is_downloaded(m))
            .collect();

        // Core ML re-prepares Bark for the Neural Engine after every app update, which takes
        // minutes. Load a downloaded Whisper model first so dictation works in the meantime;
        // Bark replaces it below once it's ready.
        if candidates.first().map_or(false, |m| m.engine == models::Engine::Bark) {
            if let Some(info) = models::whisper_stand_in() {
                if let Ok(engine) = transcribe::open_engine(&models::model_path(info)) {
                    if let Ok(mut t) = state.transcriber.lock() {
                        t.set_engine(engine, info.id);
                    }
                    eprintln!("Loaded {} while Bark prepares", info.id);
                }
            }
        }

        // Load the speech model the user chose, falling back to the default, then Whisper base.en.
        for info in candidates {
            // Open outside the lock: if Core ML has dropped Bark's cache this takes minutes.
            match transcribe::open_engine(&models::model_path(info)) {
                Ok(engine) => {
                    if let Ok(mut t) = state.transcriber.lock() {
                        t.set_engine(engine, info.id);
                    }
                    eprintln!("Loaded speech model: {}", info.id);
                    break;
                }
                Err(e) => eprintln!("Failed to load speech model {}: {}", info.id, e),
            }
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let data_dir = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("zecho");
    std::fs::create_dir_all(&data_dir).ok();
    std::fs::create_dir_all(models::model_dir()).ok();

    let settings = Settings::load(&data_dir);
    hotkey::set_fn_needs_shift(settings.fn_key_needs_shift);

    let state = AppState {
        recorder: Mutex::new(AudioRecorder::new()),
        transcriber: Mutex::new(Transcriber::new()),
        cleaner: Mutex::new(TextCleaner::new()),
        history: Mutex::new(HistoryStore::load(&data_dir)),
        settings: Mutex::new(settings),
        pending_update: Mutex::new(None),
        update_check_in_progress: std::sync::atomic::AtomicBool::new(false),
        tray_update_item: Mutex::new(None),
        tray_icon: Mutex::new(None),
    };

    // Models loaded async in setup — see init_models_async

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_nspanel::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            cancel_recording,
            get_audio_level,
            start_drag,
            persist_pill_position,
            toggle_history,
            get_history,
            copy_history_item,
            delete_history_item,
            clear_history,
            open_settings,
            get_settings,
            update_settings,
            get_whats_new,
            dismiss_whats_new,
            use_model,
            get_autostart_enabled,
            set_autostart_enabled,
            list_models,
            download_model,
            load_cleanup_model,
            load_whisper_model_cmd,
            switch_language,
            update_hotkey,
            check_accessibility,
            open_accessibility_settings,
            start_fn_listener,
            check_microphone,
            open_mic_settings,
            request_microphone,
            check_setup,
            setup_download_models,
            complete_setup,
            hide_setup,
            start_update_check,
            open_update_dialog,
            install_pending_update,
            close_update_window,
        ])
        .setup(|app| {
            #[cfg(target_os = "macos")]
            {
                use cocoa::appkit::{NSApp, NSApplication, NSApplicationActivationPolicy};
                unsafe {
                    NSApp().setActivationPolicy_(
                        NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory,
                    );
                }
            }

            create_tray_icon(app).ok();
            register_global_shortcut(app);
            tauri::async_runtime::spawn(background_update_loop(app.handle().clone()));

            // Position pill BEFORE converting to NSPanel (set_position doesn't work on NSPanels)
            {
                let state: tauri::State<'_, AppState> = app.state();
                if let Some(window) = app.get_webview_window("pill") {
                    position_pill_window(&window, &state);
                }
            }

            macos_panel::make_panel(app);
            init_models_async(app.handle().clone());

            // After an update, introduce new default models once, if there's anything to switch to.
            {
                let state = app.state::<AppState>();
                let show = state
                    .settings
                    .lock()
                    .map(|s| s.setup_complete && s.whats_new_seen < WHATS_NEW_REVISION && !whats_new_offers(&s).is_empty())
                    .unwrap_or(false);
                if show {
                    if let Some(win) = app.get_webview_window("whatsnew") {
                        win.show().ok();
                        win.set_focus().ok();
                    }
                } else {
                    mark_whats_new_seen(app.handle());
                }
            }

            // Enable autostart by default on first launch. The flag prevents
            // re-enabling after the user explicitly disables it later.
            {
                use tauri_plugin_autostart::ManagerExt;
                let mut needs_init = false;
                if let Ok(s) = app.state::<AppState>().settings.lock() {
                    needs_init = !s.autostart_initialized;
                }
                if needs_init {
                    if let Err(e) = app.autolaunch().enable() {
                        eprintln!("Failed to enable autostart on first launch: {}", e);
                    }
                    if let Ok(mut s) = app.state::<AppState>().settings.lock() {
                        s.autostart_initialized = true;
                        s.save().ok();
                    }
                }
            }

            // Only start FN key listener if accessibility is already granted
            // Otherwise, FRE will guide the user to enable it
            if accessibility::is_accessibility_enabled() {
                hotkey::start_fn_key_listener(app.handle().clone());
            }

            // Show setup window only on first run
            {
                let setup_complete = app.state::<AppState>()
                    .settings.lock()
                    .map(|s| s.setup_complete)
                    .unwrap_or(false);
                if !setup_complete {
                    if let Some(pill) = app.get_webview_window("pill") {
                        pill.hide().ok();
                    }
                    if let Some(window) = app.get_webview_window("setup") {
                        window.show().ok();
                        window.set_focus().ok();
                    }
                }
            }

            // Start cleanup worker if model already downloaded
            let cleanup_handle = app.handle().clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(3));
                // Start the model the user chose; the cleanup prompt is built for that model.
                let state = cleanup_handle.state::<AppState>();
                let (active, language) = state
                    .settings
                    .lock()
                    .map(|s| (s.active_cleanup_model.clone(), s.language.clone()))
                    .unwrap_or_default();
                if let Some(info) = models::cleanup_model_candidates(&active, &language).into_iter().find(|m| models::is_downloaded(m)) {
                    if let Ok(mut c) = state.cleaner.lock() {
                        c.start_worker(info.id, &models::model_path(info)).ok();
                    }
                }
            });

            for label in &["history", "settings", "update", "whatsnew"] {
                if let Some(win) = app.get_webview_window(label) {
                    let handle = win.clone();
                    let is_whats_new = *label == "whatsnew";
                    win.on_window_event(move |event| {
                        if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                            api.prevent_close();
                            if is_whats_new {
                                mark_whats_new_seen(handle.app_handle());
                            }
                            handle.hide().ok();
                        }
                    });
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error running zecho");
}

#[cfg(test)]
mod silence_tests {
    use super::is_silence_hallucination;

    #[test]
    fn quiet_lone_fillers_are_silence() {
        assert!(is_silence_hallucination("Yeah.", 0.0045));
        assert!(is_silence_hallucination("Okay.", 0.0025));
        assert!(is_silence_hallucination("Hmm.", 0.0092));
    }

    #[test]
    fn quiet_real_words_are_kept() {
        assert!(!is_silence_hallucination("Yes.", 0.0062));
        assert!(!is_silence_hallucination("Okay, sounds good.", 0.005));
    }

    #[test]
    fn spoken_fillers_are_kept() {
        assert!(!is_silence_hallucination("Okay.", 0.02));
    }
}

#[cfg(test)]
mod hotkey_parse_tests {
    use super::parse_hotkey;
    use tauri_plugin_global_shortcut::{Code, Modifiers};

    #[test]
    fn parses_letter_hotkey() {
        let (mods, code) = parse_hotkey("cmd+shift+r").expect("cmd+shift+r should parse");
        assert_eq!(mods, Some(Modifiers::SHIFT | Modifiers::META));
        assert_eq!(code, Code::KeyR);
    }

    #[test]
    fn parses_backslash_hotkey() {
        let (mods, code) = parse_hotkey("cmd+shift+backslash").expect("backslash should parse");
        assert_eq!(mods, Some(Modifiers::SHIFT | Modifiers::META));
        assert_eq!(code, Code::Backslash);
    }

    #[test]
    fn parses_punctuation_literals() {
        assert_eq!(parse_hotkey("cmd+\\").map(|(_, c)| c), Some(Code::Backslash));
        assert_eq!(parse_hotkey("cmd+-").map(|(_, c)| c), Some(Code::Minus));
        assert_eq!(parse_hotkey("cmd+/").map(|(_, c)| c), Some(Code::Slash));
        assert_eq!(parse_hotkey("cmd+[").map(|(_, c)| c), Some(Code::BracketLeft));
        assert_eq!(parse_hotkey("cmd+`").map(|(_, c)| c), Some(Code::Backquote));
    }

    #[test]
    fn parses_named_keys() {
        assert_eq!(parse_hotkey("cmd+up").map(|(_, c)| c), Some(Code::ArrowUp));
        assert_eq!(parse_hotkey("cmd+enter").map(|(_, c)| c), Some(Code::Enter));
        assert_eq!(parse_hotkey("cmd+tab").map(|(_, c)| c), Some(Code::Tab));
    }

    #[test]
    fn modifier_order_does_not_matter() {
        assert_eq!(
            parse_hotkey("shift+cmd+backslash"),
            parse_hotkey("cmd+shift+backslash")
        );
    }

    #[test]
    fn rejects_unknown_key() {
        assert!(parse_hotkey("cmd+shift+nonsense").is_none());
    }
}

#[cfg(test)]
mod whats_new_tests {
    use super::*;

    fn offer_ids(language: &str, speech: &str, cleanup: &str) -> Vec<&'static str> {
        let mut s = Settings::default();
        s.language = language.to_string();
        s.active_whisper_model = speech.to_string();
        s.active_cleanup_model = cleanup.to_string();
        whats_new_offers(&s).iter().map(|o| o.model_id).collect()
    }

    #[test]
    fn offers_the_new_defaults_to_people_not_using_them() {
        let ids = offer_ids("en", "whisper-base-en", "gemma4-e4b");
        assert!(ids.contains(&"spokenly-cleanup-1"));
        assert_eq!(ids.contains(&"bark-3-en"), crate::bark::is_supported());
    }

    #[test]
    fn offers_nothing_once_on_the_defaults() {
        assert!(offer_ids("en", models::default_whisper_model().id, "spokenly-cleanup-1").is_empty());
    }

    #[test]
    fn offers_nothing_for_languages_the_new_models_do_not_handle() {
        assert!(offer_ids("ko", "whisper-base-multi", "gemma4-e4b").is_empty());
    }

    #[test]
    fn new_installs_start_with_whats_new_seen() {
        assert_eq!(Settings::default().whats_new_seen, WHATS_NEW_REVISION);
    }
}
