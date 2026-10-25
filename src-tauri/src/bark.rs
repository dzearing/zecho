//! Underdog Bark 3.0 speech-to-text (English only, macOS 15+ on Apple silicon).
//!
//! A Rust port of the model's reference `coreml/ane/bark_coreml.py`: the FastConformer encoder runs as a
//! Core ML model on the Neural Engine, while the audio frontend and the greedy TDT decoder (2-layer LSTM +
//! joint head, int8 weights with per-row scales from `decoder.npz`) run on the CPU. Keep this file in step
//! with the reference implementation; `bark_parity` in the tests compares the two.

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSProcessInfo, NSString, NSURL};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

const SR: usize = 16000;
const HOP: usize = 160;
const NFFT: usize = 512;
const WIN: usize = 400;
const N_MELS: usize = 128;
const N_BINS: usize = NFFT / 2 + 1;
/// Encoder windows in seconds; each is a function (`w8`, `w15`) in the multifunction Core ML package.
const WINDOWS: [usize; 2] = [8, 15];
/// Silence added in front of each piece of audio (the setting the model was evaluated with).
const PAD_SAMPLES: usize = SR / 4;

pub const ENCODER_DIR: &str = "BarkEncoder.mlmodelc";
pub const DECODER_FILE: &str = "decoder.npz";
pub const VOCAB_FILE: &str = "vocab.json";

/// Bark's encoder needs Core ML multifunction models (macOS 15) and is only validated on Apple silicon.
pub fn is_supported() -> bool {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return false;
    }
    let v = NSProcessInfo::processInfo().operatingSystemVersion();
    v.majorVersion >= 15
}

/// True when `dir` holds a ready-to-load Bark model (compiled encoder plus decoder files).
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join(ENCODER_DIR).is_dir() && dir.join(DECODER_FILE).is_file() && dir.join(VOCAB_FILE).is_file()
}

/// Compiles a downloaded `.mlpackage` into `dest` (a `.mlmodelc` directory). Core ML caches its
/// device-specific build of a compiled model by path, so the result must live at a stable location.
#[allow(deprecated)] // the async variant needs a run loop; this runs on a worker thread anyway
pub fn compile_encoder(mlpackage: &Path, dest: &Path) -> Result<(), String> {
    let url = file_url(mlpackage)?;
    let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
        .map_err(|e| format!("Core ML compile failed: {}", e.localizedDescription()))?;
    let compiled_path = compiled
        .path()
        .map(|p| p.to_string())
        .ok_or("Core ML compile returned no path")?;
    if dest.exists() {
        std::fs::remove_dir_all(dest).map_err(|e| e.to_string())?;
    }
    // The compiler writes to a temp dir that may sit on another volume, so fall back to copying.
    if std::fs::rename(&compiled_path, dest).is_err() {
        copy_dir(Path::new(&compiled_path), dest)?;
        std::fs::remove_dir_all(&compiled_path).ok();
    }
    Ok(())
}

fn copy_dir(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    for entry in std::fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let to = dest.join(entry.file_name());
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), to).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn file_url(path: &Path) -> Result<Retained<NSURL>, String> {
    let s = path.to_str().ok_or("Invalid model path")?;
    Ok(NSURL::fileURLWithPath(&NSString::from_str(s)))
}

pub struct BarkModel {
    encoders: Vec<(usize, Retained<MLModel>)>,
    frontend: Frontend,
    decoder: Decoder,
}

// MLModel prediction is thread-safe, and the transcriber is used behind a Mutex anyway.
unsafe impl Send for BarkModel {}
unsafe impl Sync for BarkModel {}

impl BarkModel {
    /// Loads a model directory. The first load on a machine specializes the encoder for the Neural
    /// Engine, which takes minutes; later loads of the same path take well under a second.
    pub fn load(dir: &Path) -> Result<Self, String> {
        if !is_supported() {
            return Err("Bark needs macOS 15 or later on Apple silicon".to_string());
        }
        let vocab: Vocab = serde_json::from_str(
            &std::fs::read_to_string(dir.join(VOCAB_FILE)).map_err(|e| format!("Failed to read vocab: {}", e))?,
        )
        .map_err(|e| format!("Invalid vocab: {}", e))?;
        let arrays = read_npz(&dir.join(DECODER_FILE))?;
        let decoder = Decoder::new(&arrays, vocab)?;
        let mel = arrays.get("mel_filters").ok_or("decoder.npz is missing mel_filters")?;
        let frontend = Frontend::new(mel.data.clone());

        let url = file_url(&dir.join(ENCODER_DIR))?;
        let mut encoders = Vec::new();
        for w in WINDOWS {
            let model = unsafe {
                let config = MLModelConfiguration::new();
                config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
                config.setFunctionName(Some(&NSString::from_str(&format!("w{}", w))));
                MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
            }
            .map_err(|e| format!("Failed to load Bark encoder: {}", e.localizedDescription()))?;
            encoders.push((w, model));
        }
        Ok(Self { encoders, frontend, decoder })
    }

    /// `audio`: mono 16 kHz samples.
    pub fn transcribe(&self, audio: &[f32]) -> Result<String, String> {
        let max_piece = WINDOWS[WINDOWS.len() - 1] as f32 - PAD_SAMPLES as f32 / SR as f32;
        let mut parts = Vec::new();
        for piece in segments(audio, max_piece, 5.0) {
            let ids = self.piece(piece)?;
            parts.push(self.decoder.detok(&ids));
        }
        Ok(parts.join(" ").trim().to_string())
    }

    fn piece(&self, audio: &[f32]) -> Result<Vec<usize>, String> {
        let mut x = vec![0.0f32; PAD_SAMPLES];
        x.extend_from_slice(audio);
        let (feats, valid) = self.frontend.features(&x);
        let (w, model) = self
            .encoders
            .iter()
            .find(|(w, _)| x.len() <= w * SR)
            .ok_or("Audio piece is longer than the encoder window")?;
        let t = w * 100 + 1;
        let enc = unsafe { run_encoder(model, &feats, valid, t)? };
        let n = enc_len(valid).min(enc.len() / ENC_DIM);
        Ok(self.decoder.decode(&enc[..n * ENC_DIM]))
    }
}

const ENC_DIM: usize = 640;

fn enc_len(mut n: usize) -> usize {
    // Three stride-2 convolutions (kernel 3, padding 1).
    for _ in 0..3 {
        n = (n + 1) / 2;
    }
    n
}

/// Runs the encoder on `feats` ([frames][128], zero-padded to `t` frames) and returns `[T'][640]` as f32.
unsafe fn run_encoder(model: &MLModel, feats: &[f32], valid: usize, t: usize) -> Result<Vec<f32>, String> {
    let ns_err = |e: Retained<objc2_foundation::NSError>| e.localizedDescription().to_string();
    let n = |v: usize| NSNumber::new_isize(v as isize);

    let features = MLMultiArray::initWithShape_dataType_error(
        MLMultiArray::alloc(),
        &NSArray::from_retained_slice(&[n(1), n(t), n(N_MELS)]),
        MLMultiArrayDataType::Float16,
    )
    .map_err(ns_err)?;
    let mask = MLMultiArray::initWithShape_dataType_error(
        MLMultiArray::alloc(),
        &NSArray::from_retained_slice(&[n(1), n(t)]),
        MLMultiArrayDataType::Int32,
    )
    .map_err(ns_err)?;

    let fs = strides(&features);
    #[allow(deprecated)]
    let fp = features.dataPointer().as_ptr() as *mut u16;
    let frames = feats.len() / N_MELS;
    for i in 0..t {
        for j in 0..N_MELS {
            let v = if i < frames { feats[i * N_MELS + j] } else { 0.0 };
            *fp.add(i * fs[1] + j * fs[2]) = f32_to_f16(v);
        }
    }
    let ms = strides(&mask);
    #[allow(deprecated)]
    let mp = mask.dataPointer().as_ptr() as *mut i32;
    for i in 0..t {
        *mp.add(i * ms[1]) = (i < valid) as i32;
    }

    let keys = [NSString::from_str("features"), NSString::from_str("mask")];
    let values = [
        MLFeatureValue::featureValueWithMultiArray(&features),
        MLFeatureValue::featureValueWithMultiArray(&mask),
    ];
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let value_refs: Vec<&AnyObject> = values.iter().map(|v| v.as_ref() as &AnyObject).collect();
    let dict = NSDictionary::from_slices(&key_refs, &value_refs);
    let input = MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict)
        .map_err(ns_err)?;
    let output = model
        .predictionFromFeatures_error(ProtocolObject::from_ref(&*input))
        .map_err(|e| format!("Bark encoder failed: {}", ns_err(e)))?;
    let encoded = output
        .featureValueForName(&NSString::from_str("encoded"))
        .and_then(|v| v.multiArrayValue())
        .ok_or("Bark encoder returned no 'encoded' output")?;

    let shape: Vec<usize> = encoded.shape().iter().map(|v| v.as_isize() as usize).collect();
    if shape.len() != 3 || shape[2] != ENC_DIM {
        return Err(format!("Unexpected Bark encoder output shape {:?}", shape));
    }
    let es = strides(&encoded);
    #[allow(deprecated)]
    let ep = encoded.dataPointer().as_ptr();
    let data_type = encoded.dataType();
    let mut out = vec![0.0f32; shape[1] * ENC_DIM];
    for i in 0..shape[1] {
        for j in 0..ENC_DIM {
            let k = i * es[1] + j * es[2];
            out[i * ENC_DIM + j] = if data_type == MLMultiArrayDataType::Float16 {
                f16_to_f32(*(ep as *const u16).add(k))
            } else if data_type == MLMultiArrayDataType::Float32 {
                *(ep as *const f32).add(k)
            } else {
                return Err(format!("Unexpected Bark encoder output type {:?}", data_type));
            };
        }
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err("Bark encoder produced non-finite output".to_string());
    }
    Ok(out)
}

unsafe fn strides(a: &MLMultiArray) -> Vec<usize> {
    a.strides().iter().map(|v| v.as_isize() as usize).collect()
}

// ── Audio frontend ──
// A port of transformers' ParakeetFeatureExtractor: pre-emphasis, centred STFT, slaney mel, log,
// per-utterance normalisation.

struct Frontend {
    mel: Vec<f32>, // [128][257]
    window: Vec<f32>,
    fft: Fft,
}

impl Frontend {
    fn new(mel: Vec<f32>) -> Self {
        // Symmetric Hann (np.hanning), centred in the 512-sample FFT frame.
        let mut window = vec![0.0f32; NFFT];
        let off = (NFFT - WIN) / 2;
        for i in 0..WIN {
            window[off + i] =
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64).cos()) as f32;
        }
        Self { mel, window, fft: Fft::new(NFFT) }
    }

    /// Returns `([frames][128] features, number of valid frames)`.
    fn features(&self, x: &[f32]) -> (Vec<f32>, usize) {
        let len = x.len();
        let mut xp = vec![0.0f32; len + NFFT];
        if len > 0 {
            xp[NFFT / 2] = x[0];
            for i in 1..len {
                xp[NFFT / 2 + i] = x[i] - 0.97 * x[i - 1];
            }
        }
        let frames = 1 + (xp.len() - NFFT) / HOP;
        let mut feats = vec![0.0f32; frames * N_MELS];
        let mut re = vec![0.0f32; NFFT];
        let mut im = vec![0.0f32; NFFT];
        let mut power = vec![0.0f32; N_BINS];
        for f in 0..frames {
            let start = f * HOP;
            for i in 0..NFFT {
                re[i] = xp[start + i] * self.window[i];
                im[i] = 0.0;
            }
            self.fft.run(&mut re, &mut im);
            for k in 0..N_BINS {
                power[k] = re[k] * re[k] + im[k] * im[k];
            }
            let row = &mut feats[f * N_MELS..(f + 1) * N_MELS];
            for (m, out) in row.iter_mut().enumerate() {
                let filt = &self.mel[m * N_BINS..(m + 1) * N_BINS];
                let e: f32 = filt.iter().zip(&power).map(|(a, b)| a * b).sum();
                *out = (e + 2f32.powi(-24)).ln();
            }
        }

        let valid = (len / HOP).min(frames);
        for m in 0..N_MELS {
            let mean = (0..valid).map(|f| feats[f * N_MELS + m] as f64).sum::<f64>() / valid.max(1) as f64;
            let var = (0..valid).map(|f| (feats[f * N_MELS + m] as f64 - mean).powi(2)).sum::<f64>()
                / (valid.max(2) - 1) as f64;
            let std = var.sqrt() as f32;
            for f in 0..frames {
                let v = &mut feats[f * N_MELS + m];
                *v = if f < valid { (*v - mean as f32) / (std + 1e-5) } else { 0.0 };
            }
        }
        (feats, valid)
    }
}

/// Radix-2 complex FFT, in place.
struct Fft {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let (cos, sin) = (0..n / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .unzip();
        Self { n, cos, sin }
    }

    fn run(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n;
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= n {
            let half = size / 2;
            let step = n / size;
            for start in (0..n).step_by(size) {
                for k in 0..half {
                    let (wr, wi) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + half);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            size *= 2;
        }
    }
}

/// Cuts audio longer than `max_s` at pauses: within each `[min_s, max_s]` stretch, cut in the middle
/// of the quietest 200 ms span (smoothed 10 ms energy), so no word is split.
fn segments(x: &[f32], max_s: f32, min_s: f32) -> Vec<&[f32]> {
    let max_len = (max_s * SR as f32) as usize;
    if x.len() <= max_len {
        return vec![x];
    }
    let fr = SR / 100;
    let frame_energy: Vec<f32> = x
        .chunks_exact(fr)
        .map(|c| c.iter().map(|v| v * v).sum::<f32>() / fr as f32)
        .collect();
    // np.convolve(e, ones(20)/20, mode="same")
    let n = frame_energy.len();
    let energy: Vec<f32> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(10);
            let hi = (i + 9).min(n - 1);
            frame_energy[lo..=hi].iter().sum::<f32>() / 20.0
        })
        .collect();
    let mut out = Vec::new();
    let mut start = 0;
    while x.len() - start > max_len {
        let lo = (start + (min_s * SR as f32) as usize) / fr;
        let hi = ((start + max_len) / fr).min(n);
        let mut best = lo;
        for i in lo..hi {
            if energy[i] < energy[best] {
                best = i;
            }
        }
        let cut = best * fr + fr / 2;
        out.push(&x[start..cut]);
        start = cut;
    }
    out.push(&x[start..]);
    out
}

// ── TDT decoder ──

#[derive(serde::Deserialize)]
struct Vocab {
    pieces: Vec<String>,
    special_ids: Vec<usize>,
    blank_id: usize,
    vocab_size: usize,
    durations: Vec<usize>,
    max_symbols_per_step: usize,
}

struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// y += self · x
    fn mul_add(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!((x.len(), y.len()), (self.cols, self.rows));
        unsafe {
            accelerate::cblas_sgemv(
                accelerate::ROW_MAJOR,
                accelerate::NO_TRANS,
                self.rows as i32,
                self.cols as i32,
                1.0,
                self.data.as_ptr(),
                self.cols as i32,
                x.as_ptr(),
                1,
                1.0,
                y.as_mut_ptr(),
                1,
            );
        }
    }
}

struct LstmLayer {
    w_ih: Matrix,
    w_hh: Matrix,
    bias: Vec<f32>,
}

struct Decoder {
    emb: Matrix,
    lstm: Vec<LstmLayer>,
    proj_w: Matrix,
    proj_b: Vec<f32>,
    head_w: Matrix,
    head_b: Vec<f32>,
    vocab: Vocab,
}

type LstmState = Vec<(Vec<f32>, Vec<f32>)>;

impl Decoder {
    fn new(z: &HashMap<String, NpyArray>, vocab: Vocab) -> Result<Self, String> {
        let get = |k: &str| z.get(k).ok_or_else(|| format!("decoder.npz is missing {}", k));
        // Int8 weights `k.q` scaled per row by `k.s`, or a plain float array `k`.
        let weight = |k: &str| -> Result<Matrix, String> {
            let (data, shape) = match (z.get(&format!("{}.q", k)), z.get(&format!("{}.s", k))) {
                (Some(q), Some(s)) => {
                    let cols = q.data.len() / s.data.len();
                    let data = q.data.iter().enumerate().map(|(i, v)| v * s.data[i / cols]).collect();
                    (data, q.shape.clone())
                }
                _ => {
                    let a = get(k)?;
                    (a.data.clone(), a.shape.clone())
                }
            };
            let rows = shape.first().copied().unwrap_or(1);
            Ok(Matrix { rows, cols: data.len() / rows.max(1), data })
        };
        let vector = |k: &str| get(k).map(|a| a.data.clone());

        let mut lstm = Vec::new();
        for i in 0..2 {
            let b_ih = vector(&format!("decoder.lstm.bias_ih_l{}", i))?;
            let b_hh = vector(&format!("decoder.lstm.bias_hh_l{}", i))?;
            lstm.push(LstmLayer {
                w_ih: weight(&format!("decoder.lstm.weight_ih_l{}", i))?,
                w_hh: weight(&format!("decoder.lstm.weight_hh_l{}", i))?,
                bias: b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect(),
            });
        }
        Ok(Self {
            emb: weight("decoder.embedding.weight")?,
            lstm,
            proj_w: weight("decoder.decoder_projector.weight")?,
            proj_b: vector("decoder.decoder_projector.bias")?,
            head_w: weight("joint.head.weight")?,
            head_b: vector("joint.head.bias")?,
            vocab,
        })
    }

    fn step(&self, tok: usize, state: &mut LstmState) -> Vec<f32> {
        let h = self.emb.cols;
        let mut x = self.emb.data[tok * h..(tok + 1) * h].to_vec();
        let mut gates = vec![0.0f32; 4 * h];
        for (layer, (hs, cs)) in self.lstm.iter().zip(state.iter_mut()) {
            gates.copy_from_slice(&layer.bias);
            layer.w_ih.mul_add(&x, &mut gates);
            layer.w_hh.mul_add(hs, &mut gates);
            // PyTorch gate order: input, forget, cell, output.
            for j in 0..h {
                let i = sigmoid(gates[j]);
                let f = sigmoid(gates[h + j]);
                let g = gates[2 * h + j].tanh();
                let o = sigmoid(gates[3 * h + j]);
                cs[j] = f * cs[j] + i * g;
                hs[j] = o * cs[j].tanh();
            }
            x.copy_from_slice(hs);
        }
        let mut out = self.proj_b.clone();
        self.proj_w.mul_add(&x, &mut out);
        out
    }

    /// Greedy TDT decoding of `enc` (`[T][640]`, already projected), as in transformers'
    /// `ParakeetForTDT.generate` with batch size 1.
    fn decode(&self, enc: &[f32]) -> Vec<usize> {
        let v = &self.vocab;
        let t_len = enc.len() / ENC_DIM;
        let h = self.emb.cols;
        let mut state: LstmState = (0..self.lstm.len()).map(|_| (vec![0.0; h], vec![0.0; h])).collect();
        let mut dec = self.step(v.blank_id, &mut state);
        let mut logits = vec![0.0f32; self.head_w.rows];
        let mut hidden = vec![0.0f32; ENC_DIM];
        let (mut t, mut sym, mut out) = (0usize, 0usize, Vec::new());
        while t < t_len {
            for j in 0..ENC_DIM {
                hidden[j] = (enc[t * ENC_DIM + j] + dec[j]).max(0.0);
            }
            logits.copy_from_slice(&self.head_b);
            self.head_w.mul_add(&hidden, &mut logits);
            let tok = argmax(&logits[..v.vocab_size]);
            let mut d = v.durations[argmax(&logits[v.vocab_size..])];
            if tok == v.blank_id {
                t += d.max(1);
                sym = 0;
            } else {
                out.push(tok);
                dec = self.step(tok, &mut state);
                sym += 1;
                if d == 0 && sym >= v.max_symbols_per_step {
                    d = 1;
                }
                if d > 0 {
                    sym = 0;
                }
                t += d;
            }
        }
        out
    }

    fn detok(&self, ids: &[usize]) -> String {
        let v = &self.vocab;
        ids.iter()
            .filter(|i| !v.special_ids.contains(i) && **i < v.pieces.len())
            .map(|&i| v.pieces[i].as_str())
            .collect::<String>()
            .replace('▁', " ")
            .trim()
            .to_string()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// First index of the maximum, like np.argmax.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

mod accelerate {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemv(
            order: i32,
            trans: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            incx: i32,
            beta: f32,
            y: *mut f32,
            incy: i32,
        );
    }
}

// ── .npz reading ──

struct NpyArray {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Reads every array in an .npz (a zip of .npy files) as f32. Supports the dtypes decoder.npz uses.
fn read_npz(path: &Path) -> Result<HashMap<String, NpyArray>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("Invalid npz: {}", e))?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| format!("Invalid npz: {}", e))?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| format!("Invalid npz: {}", e))?;
        out.insert(name.clone(), parse_npy(&bytes).map_err(|e| format!("{}: {}", name, e))?);
    }
    Ok(out)
}

fn parse_npy(b: &[u8]) -> Result<NpyArray, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("not an .npy file".to_string());
    }
    let (header_len, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => return Err(format!("unsupported .npy version {}", v)),
    };
    let header = std::str::from_utf8(&b[start..start + header_len]).map_err(|e| e.to_string())?;
    let field = |key: &str| -> Result<&str, String> {
        let i = header.find(key).ok_or_else(|| format!("header has no {}", key))? + key.len();
        Ok(header[i..].trim_start_matches([':', ' ']))
    };
    if field("'fortran_order'")?.starts_with("True") {
        return Err("fortran-ordered arrays are not supported".to_string());
    }
    let descr = field("'descr'")?.trim_start_matches('\'');
    let descr = &descr[..descr.find('\'').ok_or("bad descr")?];
    let shape_str = field("'shape'")?;
    let shape: Vec<usize> = shape_str[1..shape_str.find(')').ok_or("bad shape")?]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let raw = &b[start + header_len..];
    let data: Vec<f32> = match descr {
        "<f4" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "<f2" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "|i1" => raw.iter().map(|&v| v as i8 as f32).collect(),
        d => return Err(format!("unsupported dtype {}", d)),
    };
    if data.len() != shape.iter().product::<usize>() {
        return Err("data length does not match shape".to_string());
    }
    Ok(NpyArray { shape, data })
}

// ── Half-precision conversion ──

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: renormalise.
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << shift) & 0x3ff) << 13
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half with round-to-nearest-even.
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1; // may carry into the exponent, which is the correct rounding
    }
    sign | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trips() {
        for v in [0.0f32, 1.0, -2.5, 0.333, 65504.0, 6.1e-5, 1e-7, -1e-6] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{} -> {:#x} -> {}", v, h, back);
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn fft_matches_dft() {
        let n = 16;
        let fft = Fft::new(n);
        let input: Vec<f32> = (0..n).map(|i| ((i * 7) % 5) as f32 - 2.0).collect();
        let (mut re, mut im) = (input.clone(), vec![0.0; n]);
        fft.run(&mut re, &mut im);
        for k in 0..n {
            let (mut r, mut i) = (0.0f64, 0.0f64);
            for (t, x) in input.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                r += *x as f64 * a.cos();
                i += *x as f64 * a.sin();
            }
            assert!((re[k] as f64 - r).abs() < 1e-4 && (im[k] as f64 - i).abs() < 1e-4);
        }
    }

    #[test]
    fn encoder_length_matches_subsampling() {
        assert_eq!(enc_len(1501), 188);
        assert_eq!(enc_len(801), 101);
        assert_eq!(enc_len(8), 1);
    }

    #[test]
    fn short_audio_is_one_segment() {
        let x = vec![0.1f32; SR * 10];
        assert_eq!(segments(&x, 14.75, 5.0).len(), 1);
    }

    #[test]
    fn long_audio_is_cut_at_the_quiet_part() {
        let mut x = vec![0.5f32; SR * 20];
        for v in &mut x[SR * 9..SR * 9 + SR / 2] {
            *v = 0.0;
        }
        let segs = segments(&x, 14.75, 5.0);
        assert_eq!(segs.len(), 2);
        let cut = segs[0].len() as f32 / SR as f32;
        assert!((9.0..9.5).contains(&cut), "cut at {}", cut);
        assert_eq!(segs.iter().map(|s| s.len()).sum::<usize>(), x.len());
    }

    /// Compares against transcripts from the Python reference. Needs a downloaded model, so it is
    /// ignored by default: `BARK_DIR=... BARK_MANIFEST=... cargo test bark_parity -- --ignored --nocapture`
    /// where the manifest is a JSON list of `{"wav": path, "hyp": reference transcript}`.
    #[test]
    #[ignore]
    fn bark_parity() {
        let dir = std::env::var("BARK_DIR").expect("BARK_DIR");
        let manifest: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(std::env::var("BARK_MANIFEST").unwrap()).unwrap()).unwrap();
        let t0 = std::time::Instant::now();
        let model = BarkModel::load(Path::new(&dir)).unwrap();
        eprintln!("load {:.2}s", t0.elapsed().as_secs_f64());
        let (mut same, mut total_ms) = (0, 0.0);
        let mut out = Vec::new();
        for m in &manifest {
            let mut r = hound::WavReader::open(m["wav"].as_str().unwrap()).unwrap();
            let s: Vec<f32> = r.samples::<i16>().map(|x| x.unwrap() as f32 / 32768.0).collect();
            let t = std::time::Instant::now();
            let text = model.transcribe(&s).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            total_ms += ms;
            if text == m["hyp"].as_str().unwrap() {
                same += 1;
            } else {
                eprintln!("DIFF {}\n  py: {}\n  rs: {}", m["wav"], m["hyp"], text);
            }
            out.push(serde_json::json!({"id": m["id"], "hyp": text, "ms": ms}));
        }
        eprintln!("{}/{} identical, {:.0} ms total", same, manifest.len(), total_ms);
        if let Ok(p) = std::env::var("BARK_OUT") {
            std::fs::write(p, serde_json::json!({"results": out}).to_string()).unwrap();
        }
    }
}
