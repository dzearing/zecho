window.onerror = function(msg, src, line, col, err) {
  console.error("ZECHO ERROR:", msg, src, line, col, err?.stack);
};
window.onunhandledrejection = function(e) {
  console.error("ZECHO ASYNC ERROR:", e.reason, e.reason?.stack);
};

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

let isRecording = false;
let recordingLocked = false;
let waveformInterval = null;
let fnHoldTimer = null;
let doneTimer = null;
let lastFnDown = 0;

const DOUBLE_TAP_MS = 400;

const $ = (sel) => document.querySelector(sel);

function setState(state) {
  const pill = $("#pill");
  const states = ["idle", "recording", "processing", "done", "setup"];

  states.forEach((s) => {
    const el = $(`#state-${s}`);
    if (el) el.classList.toggle("hidden", s !== state);
  });

  pill.className = "";
  if (state !== "idle") {
    pill.classList.add(state);
  }
}

let barLevels = new Array(16).fill(0);

function startWaveform() {
  const canvas = $("#waveform");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const bars = 16;
  const barW = 3;
  const totalWidth = bars * barW + (bars - 1) * 2;
  const offsetX = (w - totalWidth) / 2;

  waveformInterval = setInterval(async () => {
    let level = 0;
    try {
      level = await invoke("get_audio_level");
    } catch (_) {}

    // Aggressive normalization — make speech visually prominent
    const normalized = Math.min(1, Math.pow(level * 15, 0.7));

    barLevels.shift();
    barLevels.push(normalized);

    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < bars; i++) {
      const amp = 0.08 + barLevels[i] * 0.92;
      const barH = amp * h * 0.9;
      const x = offsetX + i * (barW + 2);
      const y = (h - barH) / 2;
      const alpha = 0.5 + barLevels[i] * 0.5;
      // Tint bars slightly purple when active
      const r = Math.round(200 + barLevels[i] * 55);
      const g = Math.round(200 + barLevels[i] * 30);
      const b = 255;
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, 1.5);
      ctx.fill();
    }
  }, 50);
}

function stopWaveform() {
  if (waveformInterval) {
    clearInterval(waveformInterval);
    waveformInterval = null;
  }
}

async function startRecording() {
  if (isRecording) return;
  if (doneTimer) { clearTimeout(doneTimer); doneTimer = null; }
  try {
    await invoke("start_recording");
    isRecording = true;
    setState("recording");
    startWaveform();
  } catch (err) {
    console.error("Start error:", err);
  }
}

async function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  recordingLocked = false;
  stopWaveform();
  setState("processing");
  try {
    await invoke("stop_recording");
    // UI stays in "processing" — transcription-complete event will trigger "done"
  } catch (err) {
    console.error("Stop error:", err);
    setState("idle");
  }
}

async function cancelRecording() {
  if (!isRecording) return;
  isRecording = false;
  recordingLocked = false;
  stopWaveform();
  try {
    await invoke("cancel_recording");
  } catch (err) {
    console.error("Cancel error:", err);
  }
  setState("idle");
}

// ── FN key: hold-to-record + double-tap-to-lock ──

function handleFnDown() {
  const now = Date.now();

  if (isRecording && recordingLocked) {
    // FN pressed while locked recording — stop it
    stopRecording();
    return;
  }

  if (!isRecording) {
    // Check for double-tap
    if (now - lastFnDown < DOUBLE_TAP_MS) {
      // Double-tap: start and lock
      recordingLocked = true;
      startRecording();
    } else {
      // Single press: start (will stop on release unless locked)
      recordingLocked = false;
      startRecording();
    }
  }

  lastFnDown = now;
}

function handleFnUp() {
  if (isRecording && !recordingLocked) {
    // Release after hold — stop recording
    stopRecording();
  }
}

function toggleHistory() {
  invoke("toggle_history").catch(() => {});
}

// ── Event listeners ──

$("#btn-history").addEventListener("click", (e) => {
  e.stopPropagation();
  toggleHistory();
});

$("#btn-settings").addEventListener("click", async (e) => {
  e.stopPropagation();
  try {
    await invoke("open_settings");
  } catch (err) {
    console.error("Settings error:", err);
  }
});

$("#btn-cancel").addEventListener("click", (e) => {
  e.stopPropagation();
  cancelRecording();
});

$("#btn-stop").addEventListener("click", (e) => {
  e.stopPropagation();
  stopRecording();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && isRecording) {
    cancelRecording();
  }
});

// Backend events
listen("pill-hover", (event) => {
  const pill = $("#pill");
  if (event.payload) {
    pill.classList.add("hover");
  } else {
    pill.classList.remove("hover");
  }
});
let fnKeyEnabled = true;
invoke("get_settings").then((s) => { fnKeyEnabled = s.fn_key_enabled !== false; }).catch(() => {});
listen("settings-changed", (event) => { fnKeyEnabled = event.payload.fn_key_enabled !== false; });
listen("fn-key-down", () => { if (fnKeyEnabled) handleFnDown(); });
listen("fn-key-up", () => { if (fnKeyEnabled) handleFnUp(); });
listen("toggle-recording", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});
listen("cancel-recording", () => {
  if (isRecording) cancelRecording();
});
listen("transcription-complete", () => {
  setState("done");
  doneTimer = setTimeout(() => setState("idle"), 1200);
});
listen("transcription-error", (event) => {
  console.error("Transcription error:", event.payload);
  setState("idle");
});


// ── Drag ──

$("#pill").addEventListener("mousedown", async (e) => {
  if (e.target.closest("button") || e.target.closest("canvas")) return;
  try {
    await invoke("start_drag");
  } catch (_) {}
  invoke("persist_pill_position").catch(() => {});
});

// ── First-run setup ──

async function checkSetup() {
  try {
    const status = await invoke("check_setup");
    if (!status.whisper_ready || !status.cleanup_ready) {
      setState("setup");
      $("#setup-label").textContent = "Downloading models...";
      await invoke("setup_download_models");
    } else {
      setState("idle");
    }
  } catch (err) {
    setState("idle");
  }
}

listen("setup-progress", (event) => {
  const label = $("#setup-label");
  if (label) label.textContent = event.payload;
});

listen("setup-complete", () => {
  setState("idle");
});

listen("setup-error", (event) => {
  const label = $("#setup-label");
  if (label) label.textContent = "Setup failed";
  setTimeout(() => setState("idle"), 3000);
});

checkSetup();
