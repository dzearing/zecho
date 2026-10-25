use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub size_mb: u32,
    pub quality_score: u8,
    pub speed_score: u8,
    pub filename: &'static str,
    pub url: &'static str,
    pub model_type: ModelType,
    pub multilingual: bool,
    /// Speech models only: which engine runs it. `ModelType::Whisper` means "speech model" to the UI.
    pub engine: Engine,
    /// Multi-file models: `(path under url, path under the model's directory)`. When non-empty,
    /// `filename` names a directory and `url` is the base the remote paths are relative to.
    #[serde(skip)]
    pub files: &'static [(&'static str, &'static str)],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    Whisper,
    Cleanup,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Engine {
    Whisper,
    Bark,
    Llama,
}

const BARK_ENCODER_PKG: &str = "BarkEncoder.mlpackage";

/// Core ML package for the Neural Engine, plus the CPU decoder weights and attribution files.
const BARK_FILES: &[(&str, &str)] = &[
    ("ane/BarkEncoder.mlpackage/Manifest.json", "BarkEncoder.mlpackage/Manifest.json"),
    (
        "ane/BarkEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "BarkEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    ),
    (
        "ane/BarkEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "BarkEncoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    ),
    ("ane/decoder.npz", "decoder.npz"),
    ("ane/vocab.json", "vocab.json"),
    ("NOTICE", "NOTICE"),
    ("LICENSE", "LICENSE"),
];

pub const AVAILABLE_MODELS: &[ModelInfo] = &[
    // Whisper models — English-only (faster for English)
    ModelInfo {
        id: "whisper-tiny-en",
        name: "Fast",
        description: "Quickest transcription. Good for short, clear speech.",
        size_mb: 75,
        quality_score: 6,
        speed_score: 10,
        filename: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        model_type: ModelType::Whisper,
        multilingual: false,
        engine: Engine::Whisper,
        files: &[],
    },
    ModelInfo {
        id: "whisper-base-en",
        name: "Balanced",
        description: "Accurate and fast for everyday use.",
        size_mb: 142,
        quality_score: 8,
        speed_score: 8,
        filename: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        model_type: ModelType::Whisper,
        multilingual: false,
        engine: Engine::Whisper,
        files: &[],
    },
    ModelInfo {
        id: "whisper-small-en",
        name: "Accurate",
        description: "Best accuracy. Uses more memory and is slower.",
        size_mb: 466,
        quality_score: 9,
        speed_score: 5,
        filename: "ggml-small.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        model_type: ModelType::Whisper,
        multilingual: false,
        engine: Engine::Whisper,
        files: &[],
    },
    // Underdog Bark 3.0 (NVIDIA Parakeet TDT 0.6B v3 fine-tune), English only, macOS 15+ on Apple silicon.
    // Pinned to a repo revision so the files can't change under us. CC-BY-4.0; NOTICE ships with it.
    ModelInfo {
        id: "bark-3-en",
        name: "Bark 3.0",
        description: "Fastest and most accurate for English. Runs on the Neural Engine. Prepares for a few minutes after download. By Conway Research, CC-BY-4.0.",
        size_mb: 193,
        quality_score: 10,
        speed_score: 10,
        filename: "bark-3.0-ane",
        url: "https://huggingface.co/ConwayResearch/Underdog-Bark-3.0/resolve/85db2ad18ab41e534a222ee253c29191d77ed919/coreml",
        model_type: ModelType::Whisper,
        multilingual: false,
        engine: Engine::Bark,
        files: BARK_FILES,
    },
    // Whisper models — Multilingual
    ModelInfo {
        id: "whisper-tiny-multi",
        name: "Fast",
        description: "Quickest transcription. Multilingual support.",
        size_mb: 75,
        quality_score: 5,
        speed_score: 10,
        filename: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        model_type: ModelType::Whisper,
        multilingual: true,
        engine: Engine::Whisper,
        files: &[],
    },
    ModelInfo {
        id: "whisper-base-multi",
        name: "Balanced",
        description: "Recommended — accurate and fast. Multilingual support.",
        size_mb: 142,
        quality_score: 7,
        speed_score: 8,
        filename: "ggml-base.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        model_type: ModelType::Whisper,
        multilingual: true,
        engine: Engine::Whisper,
        files: &[],
    },
    ModelInfo {
        id: "whisper-small-multi",
        name: "Accurate",
        description: "Best accuracy. Multilingual support.",
        size_mb: 466,
        quality_score: 8,
        speed_score: 5,
        filename: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        model_type: ModelType::Whisper,
        multilingual: true,
        engine: Engine::Whisper,
        files: &[],
    },
    // Cleanup models
    // Spokenly Cleanup 1: a Qwen3-0.6B fine-tune for dictation cleanup. Apache-2.0. Pinned to a revision.
    // It takes its own fixed prompt, so writing style, cleanup level and custom instructions don't apply.
    ModelInfo {
        id: "spokenly-cleanup-1",
        name: "Spokenly Cleanup",
        description: "Fastest. Trained only for dictation cleanup; ignores writing style and custom instructions.",
        size_mb: 610,
        quality_score: 8,
        speed_score: 10,
        filename: "spokenly-cleanup-1-q8_0.gguf",
        url: "https://huggingface.co/spokenly/cleanup-1-gguf/resolve/79936c95e5c0ea9162ae7a435e64cb3127020e36/cleanup-1-q8_0.gguf",
        model_type: ModelType::Cleanup,
        multilingual: false,
        engine: Engine::Llama,
        files: &[],
    },
    ModelInfo {
        id: "qwen25-1.5b",
        name: "Balanced",
        description: "Recommended — reliable cleanup with good speed.",
        size_mb: 986,
        quality_score: 8,
        speed_score: 8,
        filename: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        model_type: ModelType::Cleanup,
        multilingual: false,
        engine: Engine::Llama,
        files: &[],
    },
    ModelInfo {
        id: "qwen25-3b",
        name: "Quality",
        description: "Best at handling corrections and nuance. Slower.",
        size_mb: 1940,
        quality_score: 9,
        speed_score: 6,
        filename: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        model_type: ModelType::Cleanup,
        multilingual: false,
        engine: Engine::Llama,
        files: &[],
    },
    ModelInfo {
        id: "gemma4-e2b",
        name: "Gemma 4 E2B",
        description: "Google Gemma 4 — fast, high-quality cleanup.",
        size_mb: 2963,
        quality_score: 9,
        speed_score: 7,
        filename: "gemma-4-E2B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
        model_type: ModelType::Cleanup,
        multilingual: false,
        engine: Engine::Llama,
        files: &[],
    },
    ModelInfo {
        id: "gemma4-e4b",
        name: "Gemma 4 E4B",
        description: "Google Gemma 4 — best quality, larger download.",
        size_mb: 4747,
        quality_score: 10,
        speed_score: 5,
        filename: "gemma-4-E4B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
        model_type: ModelType::Cleanup,
        multilingual: false,
        engine: Engine::Llama,
        files: &[],
    },
];

pub fn model_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("zecho")
        .join("models")
}

pub fn get_model(id: &str) -> Option<&'static ModelInfo> {
    AVAILABLE_MODELS.iter().find(|m| m.id == id)
}

pub fn model_path(info: &ModelInfo) -> PathBuf {
    model_dir().join(info.filename)
}

pub fn is_downloaded(info: &ModelInfo) -> bool {
    match info.engine {
        Engine::Bark => crate::bark::is_model_dir(&model_path(info)),
        _ => model_path(info).exists(),
    }
}

/// Whether this machine can run the model at all.
pub fn is_supported(info: &ModelInfo) -> bool {
    match info.engine {
        Engine::Bark => crate::bark::is_supported(),
        _ => true,
    }
}

/// The English speech model for new installs: Bark where the Mac can run it, else Whisper base.en.
pub fn default_whisper_model() -> &'static ModelInfo {
    let bark = get_model("bark-3-en").unwrap();
    if is_supported(bark) {
        bark
    } else {
        get_model("whisper-base-en").unwrap()
    }
}

/// Speech models to try loading, best first: the user's choice, the default, then Whisper base.en,
/// which existing installs already have, so upgrading never forces a download.
pub fn speech_model_candidates(active_id: &str) -> Vec<&'static ModelInfo> {
    let mut out: Vec<&'static ModelInfo> = Vec::new();
    for info in [get_model(active_id), Some(default_whisper_model()), get_model("whisper-base-en")]
        .into_iter()
        .flatten()
    {
        if is_supported(info) && !out.iter().any(|m| m.id == info.id) {
            out.push(info);
        }
    }
    out
}

pub fn whisper_model_for_language(current_id: &str, language: &str) -> String {
    let need_multilingual = language != "en";
    let current = get_model(current_id);
    let current_is_multilingual = current.map(|m| m.multilingual).unwrap_or(false);

    if need_multilingual == current_is_multilingual {
        return current_id.to_string();
    }

    let tier = if current_id.contains("tiny") {
        "tiny"
    } else if current_id.contains("small") {
        "small"
    } else {
        "base"
    };

    let suffix = if need_multilingual { "multi" } else { "en" };
    let target_id = format!("whisper-{}-{}", tier, suffix);

    if get_model(&target_id).is_some() {
        target_id
    } else {
        current_id.to_string()
    }
}

pub fn default_cleanup_model() -> &'static ModelInfo {
    get_model("qwen25-1.5b").unwrap()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    pub name: String,
    pub description: String,
    pub size_mb: u32,
    pub quality_score: u8,
    pub speed_score: u8,
    pub downloaded: bool,
    pub model_type: ModelType,
    pub multilingual: bool,
}

pub fn list_models() -> Vec<ModelStatus> {
    AVAILABLE_MODELS
        .iter()
        .filter(|m| is_supported(m))
        .map(|m| ModelStatus {
            id: m.id.to_string(),
            name: m.name.to_string(),
            description: if m.id == default_whisper_model().id {
                format!("Recommended — {}", m.description)
            } else {
                m.description.to_string()
            },
            size_mb: m.size_mb,
            quality_score: m.quality_score,
            speed_score: m.speed_score,
            downloaded: is_downloaded(m),
            model_type: m.model_type.clone(),
            multilingual: m.multilingual,
        })
        .collect()
}

pub fn download_model_blocking(
    info: &ModelInfo,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<PathBuf, String> {
    if !info.files.is_empty() {
        return download_dir_blocking(info, on_progress);
    }

    let path = model_path(info);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let tmp_path = path.with_extension("tmp");
    let total = info.size_mb as u64 * 1024 * 1024;
    download_file(info.url, &tmp_path, |n, len| on_progress(n, len.unwrap_or(total)))?;
    std::fs::rename(&tmp_path, &path).map_err(|e| format!("Failed to rename: {}", e))?;

    Ok(path)
}

/// Downloads every file of a multi-file model into `<name>.tmp/`, finishes it (Bark: compiles the
/// Core ML encoder), then moves it into place so a half-done download never looks installed.
fn download_dir_blocking(info: &ModelInfo, mut on_progress: impl FnMut(u64, u64)) -> Result<PathBuf, String> {
    let path = model_path(info);
    let tmp = path.with_extension("tmp");
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).map_err(|e| e.to_string())?;
    }
    let total = info.size_mb as u64 * 1024 * 1024;
    let mut done: u64 = 0;
    for (remote, local) in info.files {
        let dest = tmp.join(local);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let url = format!("{}/{}", info.url, remote);
        let mut last = 0;
        download_file(&url, &dest, |n, _| {
            last = n;
            on_progress((done + n).min(total), total);
        })?;
        done += last;
    }

    if info.engine == Engine::Bark {
        let pkg = tmp.join(BARK_ENCODER_PKG);
        crate::bark::compile_encoder(&pkg, &tmp.join(crate::bark::ENCODER_DIR))?;
        std::fs::remove_dir_all(&pkg).ok();
    }

    if path.exists() {
        std::fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, &path).map_err(|e| format!("Failed to rename: {}", e))?;
    on_progress(total, total);
    Ok(path)
}

/// Streams `url` to `dest`, reporting `(bytes so far, content length)`.
fn download_file(url: &str, dest: &std::path::Path, mut on_progress: impl FnMut(u64, Option<u64>)) -> Result<(), String> {
    use std::io::{Read, Write};

    let response = reqwest::blocking::get(url).map_err(|e| format!("Download failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }

    let total = response.content_length();
    let mut file = std::fs::File::create(dest).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut downloaded: u64 = 0;
    let mut reader = response;
    let mut buf = [0u8; 65536];

    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("Read error: {}", e))?;
        if n == 0 { break; }
        file.write_all(&buf[..n]).map_err(|e| format!("Write error: {}", e))?;
        downloaded += n as u64;
        on_progress(downloaded, total);
    }

    file.flush().map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bark_is_the_default_where_supported() {
        let expected = if crate::bark::is_supported() { "bark-3-en" } else { "whisper-base-en" };
        assert_eq!(default_whisper_model().id, expected);
        let recommended: Vec<_> = list_models()
            .into_iter()
            .filter(|m| m.model_type == ModelType::Whisper && !m.multilingual && m.description.starts_with("Recommended"))
            .map(|m| m.id)
            .collect();
        assert_eq!(recommended, vec![expected.to_string()]);
    }

    #[test]
    fn speech_candidates_fall_back_to_base_en() {
        let ids = |active: &str| speech_model_candidates(active).iter().map(|m| m.id).collect::<Vec<_>>();
        let default = default_whisper_model().id;
        assert_eq!(ids("whisper-small-en")[..2], ["whisper-small-en", default]);
        assert_eq!(ids("whisper-small-en").last(), Some(&"whisper-base-en"));
        // Unknown ids (e.g. a model from another build) are skipped, and nothing is listed twice.
        assert_eq!(ids("whisper-crisperwhisper-small")[0], default);
        assert_eq!(ids("whisper-base-en").iter().filter(|id| **id == "whisper-base-en").count(), 1);
    }

    #[test]
    fn bark_is_kept_for_english() {
        assert_eq!(whisper_model_for_language("bark-3-en", "en"), "bark-3-en");
    }

    #[test]
    fn bark_falls_back_to_multilingual_whisper() {
        // Bark is English-only, so another language needs a multilingual Whisper model.
        assert_eq!(whisper_model_for_language("bark-3-en", "fr"), "whisper-base-multi");
    }

    #[test]
    fn whisper_tiers_still_switch() {
        assert_eq!(whisper_model_for_language("whisper-small-en", "de"), "whisper-small-multi");
        assert_eq!(whisper_model_for_language("whisper-tiny-multi", "en"), "whisper-tiny-en");
    }

    #[test]
    fn bark_downloads_into_a_directory() {
        let bark = get_model("bark-3-en").unwrap();
        assert_eq!(bark.engine, Engine::Bark);
        assert!(!bark.multilingual);
        assert!(bark.url.contains("/resolve/") && !bark.url.contains("/main/"), "pin a revision");
        let locals: Vec<&str> = bark.files.iter().map(|(_, l)| *l).collect();
        for f in [crate::bark::DECODER_FILE, crate::bark::VOCAB_FILE, "NOTICE"] {
            assert!(locals.contains(&f), "missing {}", f);
        }
        assert!(locals.iter().any(|l| l.starts_with(BARK_ENCODER_PKG)));
    }

    /// Downloads and installs Bark into the app's model directory, as the Settings download does.
    /// Network-heavy (~200 MB): `cargo test bark_download -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bark_download() {
        let bark = get_model("bark-3-en").unwrap();
        let mut last = 0;
        let path = download_model_blocking(bark, |n, total| {
            let pct = n * 100 / total;
            if pct >= last + 20 {
                last = pct;
                eprintln!("{}%", pct);
            }
        })
        .unwrap();
        assert!(is_downloaded(bark), "{} incomplete", path.display());
        assert!(!path.join(BARK_ENCODER_PKG).exists(), "the .mlpackage should be removed after compiling");
    }
}
