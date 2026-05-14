use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WritingStyle {
    Formal,
    Casual,
    VeryCasual,
}

impl Default for WritingStyle {
    fn default() -> Self {
        Self::Casual
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CleanupLevel {
    None,
    Light,
    Medium,
    High,
}

impl Default for CleanupLevel {
    fn default() -> Self {
        Self::Light
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    pub whisper_model: String,
    pub auto_paste: bool,
    pub writing_style: WritingStyle,
    pub cleanup_level: CleanupLevel,
    pub custom_prompt: Option<String>,
    pub active_whisper_model: String,
    pub active_cleanup_model: String,
    pub setup_complete: bool,
    pub pill_x_pct: Option<f64>,
    pub pill_y_pct: Option<f64>,
    #[serde(default = "default_true")]
    pub fn_key_enabled: bool,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub autostart_initialized: bool,
    #[serde(skip)]
    pub(crate) path: PathBuf,
}

fn default_true() -> bool {
    true
}

fn default_language() -> String {
    "en".to_string()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "ctrl+shift+r".to_string(),
            whisper_model: "base.en".to_string(),
            auto_paste: true,
            writing_style: WritingStyle::default(),
            cleanup_level: CleanupLevel::default(),
            custom_prompt: None,
            active_whisper_model: "whisper-base-en".to_string(),
            active_cleanup_model: "qwen25-1.5b".to_string(),
            setup_complete: false,
            pill_x_pct: None,
            pill_y_pct: None,
            fn_key_enabled: true,
            language: "en".to_string(),
            autostart_initialized: false,
            path: PathBuf::new(),
        }
    }
}

impl Settings {
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join("settings.json");
        let mut settings: Settings = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        settings.path = path;
        settings.validate_active_models();
        settings
    }

    fn validate_active_models(&mut self) {
        use crate::models;

        if let Some(info) = models::get_model(&self.active_cleanup_model) {
            if models::is_downloaded(info) {
                return;
            }
        }
        let candidates = ["qwen25-1.5b", "qwen25-3b", "gemma4-e2b", "gemma4-e4b"];
        for id in &candidates {
            if let Some(info) = models::get_model(id) {
                if models::is_downloaded(info) {
                    self.active_cleanup_model = id.to_string();
                    self.save().ok();
                    return;
                }
            }
        }
    }

    pub fn save(&self) -> Result<(), String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| e.to_string())
            .and_then(|json| std::fs::write(&self.path, json).map_err(|e| e.to_string()))
    }
}
