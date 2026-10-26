use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};

const WHISPER_SAMPLE_RATE: u32 = 16000;

pub struct AudioRecorder {
    samples: Arc<Mutex<Vec<f32>>>,
    device_rate: Arc<Mutex<u32>>,
    recording: Arc<AtomicBool>,
    stream_handle: Mutex<Option<cpal::Stream>>,
    current_rms: Arc<AtomicU32>,
}

unsafe impl Send for AudioRecorder {}
unsafe impl Sync for AudioRecorder {}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            device_rate: Arc::new(Mutex::new(WHISPER_SAMPLE_RATE)),
            recording: Arc::new(AtomicBool::new(false)),
            stream_handle: Mutex::new(None),
            current_rms: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn rms_level(&self) -> f32 {
        f32::from_bits(self.current_rms.load(Ordering::Relaxed))
    }

    pub fn start(&self) -> Result<(), String> {
        if self.recording.load(Ordering::SeqCst) {
            return Ok(());
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        *self.device_rate.lock().unwrap() = rate;
        self.samples.lock().unwrap().clear();
        self.recording.store(true, Ordering::SeqCst);
        self.current_rms.store(0, Ordering::SeqCst);

        let samples = self.samples.clone();
        let recording = self.recording.clone();
        let rms = self.current_rms.clone();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device
                .build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !recording.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut buf = samples.lock().unwrap();
                        let mut sum_sq: f32 = 0.0;
                        let mut count: usize = 0;
                        for chunk in data.chunks(channels) {
                            let s = chunk[0];
                            buf.push(s);
                            sum_sq += s * s;
                            count += 1;
                        }
                        if count > 0 {
                            let level = (sum_sq / count as f32).sqrt();
                            rms.store(level.to_bits(), Ordering::Relaxed);
                        }
                    },
                    |err| eprintln!("Audio stream error: {}", err),
                    None,
                )
                .map_err(|e| e.to_string())?,
            cpal::SampleFormat::I16 => {
                let samples = self.samples.clone();
                let recording = self.recording.clone();
                let rms = self.current_rms.clone();
                device
                    .build_input_stream(
                        &config.into(),
                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                            if !recording.load(Ordering::Relaxed) {
                                return;
                            }
                            let mut buf = samples.lock().unwrap();
                            let mut sum_sq: f32 = 0.0;
                            let mut count: usize = 0;
                            for chunk in data.chunks(channels) {
                                let s = chunk[0] as f32 / i16::MAX as f32;
                                buf.push(s);
                                sum_sq += s * s;
                                count += 1;
                            }
                            if count > 0 {
                                let level = (sum_sq / count as f32).sqrt();
                                rms.store(level.to_bits(), Ordering::Relaxed);
                            }
                        },
                        |err| eprintln!("Audio stream error: {}", err),
                        None,
                    )
                    .map_err(|e| e.to_string())?
            }
            fmt => return Err(format!("Unsupported sample format: {:?}", fmt)),
        };

        stream.play().map_err(|e| e.to_string())?;
        *self.stream_handle.lock().unwrap() = Some(stream);

        Ok(())
    }

    pub fn stop(&self) -> Vec<f32> {
        self.recording.store(false, Ordering::SeqCst);
        *self.stream_handle.lock().unwrap() = None;

        let raw_samples = self.samples.lock().unwrap().clone();
        let device_rate = *self.device_rate.lock().unwrap();

        if raw_samples.is_empty() {
            return Vec::new();
        }

        resample(&raw_samples, device_rate, WHISPER_SAMPLE_RATE)
    }

    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;
        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };
        output.push(sample as f32);
    }
    output
}

/// RMS of the loudest 100 ms of 16 kHz audio. Speech is sustained, so a click or bump can't
/// raise it the way it raises the peak sample.
pub fn loudest_window_rms(samples: &[f32]) -> f32 {
    samples
        .chunks(1600)
        .map(|w| (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt())
        .fold(0.0, f32::max)
}
