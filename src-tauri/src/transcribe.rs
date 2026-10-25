use crate::bark::BarkModel;
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub enum Engine {
    Whisper(WhisperContext),
    Bark(BarkModel),
}

pub struct Transcriber {
    engine: Option<Engine>,
}

unsafe impl Send for Transcriber {}
unsafe impl Sync for Transcriber {}

impl Transcriber {
    pub fn new() -> Self {
        Self { engine: None }
    }

    /// Swaps in an engine opened with `open_engine`, so slow loads don't need the transcriber lock.
    pub fn set_engine(&mut self, engine: Engine) {
        self.engine = Some(engine);
    }

    pub fn is_loaded(&self) -> bool {
        self.engine.is_some()
    }

    pub fn transcribe(&self, audio_samples: &[f32], language: &str) -> Result<String, String> {
        let engine = self
            .engine
            .as_ref()
            .ok_or("Speech model not loaded. Check Settings for model status.")?;

        if audio_samples.is_empty() {
            return Ok(String::new());
        }

        let ctx = match engine {
            Engine::Whisper(ctx) => ctx,
            Engine::Bark(model) => {
                if language != "en" {
                    return Err("Bark is English-only. Choose a multilingual speech model in Settings.".to_string());
                }
                return model.transcribe(audio_samples);
            }
        };

        let mut state = ctx.create_state().map_err(|e| format!("Whisper state error: {}", e))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(language));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        state
            .full(params, audio_samples)
            .map_err(|e| format!("Whisper transcription error: {}", e))?;

        let num_segments = state
            .full_n_segments()
            .map_err(|e| format!("Segment count error: {}", e))?;

        let mut text = String::new();
        for i in 0..num_segments {
            if let Ok(segment) = state.full_get_segment_text(i) {
                text.push_str(&segment);
            }
        }

        Ok(text.trim().to_string())
    }

    pub fn model_dir() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("zecho")
            .join("models")
    }

    pub fn default_model_path() -> PathBuf {
        Self::model_dir().join("ggml-base.en.bin")
    }
}

/// Opens a Whisper ggml file, or a Bark model directory.
pub fn open_engine(model_path: &std::path::Path) -> Result<Engine, String> {
    if model_path.is_dir() {
        return Ok(Engine::Bark(BarkModel::load(model_path)?));
    }
    let ctx = WhisperContext::new_with_params(
        model_path.to_str().ok_or("Invalid model path")?,
        WhisperContextParameters::default(),
    )
    .map_err(|e| format!("Failed to load whisper model: {}", e))?;
    Ok(Engine::Whisper(ctx))
}
