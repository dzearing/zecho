# Zecho

**Voice to clipboard, instantly.** Speak naturally, get clean text -- no subscriptions, no cloud, no ads. Everything runs locally on your machine.

Hold the **FN key** to record, release to get polished text copied to your clipboard. That's it.

## How It Works

1. **Speak** -- Hold FN and talk naturally. Corrections, pauses, and filler words are all fine.
2. **AI cleans** -- Local speech recognition (Bark or Whisper) transcribes your speech. A small local model removes filler words, fixes grammar, and resolves self-corrections ("I want red, uh, no actually blue" becomes "I want blue").
3. **Paste** -- Clean text lands in your clipboard. Paste anywhere.

## Features

- **Fully local** -- No data leaves your machine. No accounts, no network calls, no telemetry.
- **FN key recording** -- Hold to record, release to transcribe. Double-tap to lock recording.
- **AI text cleanup** -- Removes "um," "uh," filler words, and resolves self-corrections using a small local model.
- **Writing styles** -- Formal, Casual, or Very Casual. Custom prompts for fine-tuning.
- **Cleanup levels** -- None, Light, Medium, or High -- control how aggressively AI edits your text.
- **Clipboard history** -- Every transcription is saved with before/after comparison (raw transcription vs. cleaned text).
- **macOS native floating panel** -- Hover and interact without stealing focus from your active window.
- **Lightweight** -- Built with Rust and Tauri v2. ~3 MB app overhead.
- **Auto updates** -- New versions are detected and applied automatically.

## Getting Started

### Prerequisites

- macOS 12+
- Rust toolchain (`rustup`)
- Tauri CLI v2 (`cargo install tauri-cli --version "^2"`)

### Setup

```bash
# Clone the repo
git clone https://github.com/dzearing/zecho.git
cd zecho

# Download the default models (Whisper + Spokenly Cleanup)
./scripts/download-models.sh

# Run in development mode
cargo tauri dev
```

The model download script places files in `~/Library/Application Support/zecho/models/`. The app will also download models on demand through its settings UI.

### Project Structure

```
ui/              # Frontend (HTML/CSS/JS)
src-tauri/       # Rust backend (Tauri v2)
  src/
    audio.rs       # Audio capture (cpal)
    transcribe.rs  # Speech-to-text engine selection (Whisper / Bark)
    bark.rs        # Bark 3.0 on Core ML (Neural Engine)
    cleanup.rs     # LLM text cleanup
    hotkey.rs      # FN key listener (macOS)
    history.rs     # Clipboard history storage
    settings.rs    # User preferences
    models.rs      # Model management & downloads
    macos_panel.rs # Native floating panel
scripts/         # Setup scripts
docs/            # Landing page (GitHub Pages)
```

## Download

Get the latest release: [github.com/dzearing/zecho/releases/latest](https://github.com/dzearing/zecho/releases/latest)

## Speech models

Zecho uses [Whisper](https://github.com/ggerganov/whisper.cpp) by default. On macOS 15+ with Apple silicon, English users can opt into **Bark 3.0** in Settings. It runs on the Neural Engine and is faster and more accurate than Whisper base.en, but it's English-only, and after download it takes a few minutes to prepare for your Mac.

Bark 3.0 is [Underdog Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) by Conway Research, derived from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (© NVIDIA Corporation) and [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) (© Moondream). It's used under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); Zecho downloads the model unmodified, along with its NOTICE and LICENSE.

Text cleanup defaults to [Spokenly Cleanup 1](https://huggingface.co/spokenly/cleanup-1) (Apache-2.0), a small model trained only for dictation cleanup. It follows the writing style but not the cleanup level or custom instructions; pick a Gemma model in Settings for those.

## License

Apache 2.0
