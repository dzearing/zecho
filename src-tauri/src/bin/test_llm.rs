use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::io::{self, BufRead, Write};
use std::num::NonZeroU32;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: test_llm <model_path> [prompt]");
        eprintln!("  With prompt: single-shot mode, prints result and exits");
        eprintln!("  Without prompt: worker mode, reads prompts from stdin (one per line, \\n escaped as \\\\n)");
        std::process::exit(1);
    }

    let model_path = &args[1];

    let backend = LlamaBackend::init().expect("backend init");
    let model_params = LlamaModelParams::default().with_n_gpu_layers(0);
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .expect("model load");

    if args.len() >= 3 {
        // Single-shot mode
        let result = run_inference(&model, &backend, &args[2]);
        print!("{}", result);
    } else {
        // Worker mode: read prompts from stdin, write results to stdout
        eprintln!("test_llm: worker mode ready");
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            if line.is_empty() {
                continue;
            }

            let prompt = line.replace("\\n", "\n");
            let result = run_inference(&model, &backend, &prompt);
            let escaped = result.replace('\n', "\\n");
            writeln!(stdout, "{}", escaped).ok();
            stdout.flush().ok();
        }
    }
}

fn run_inference(model: &LlamaModel, backend: &LlamaBackend, prompt: &str) -> String {
    let tokens = match model.str_to_token(prompt, AddBos::Always) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Tokenize error: {}", e);
            return String::new();
        }
    };

    let n_ctx = std::cmp::max(tokens.len() as u32 + 256, 1024);
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
    let mut ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Context error: {}", e);
            return String::new();
        }
    };

    let mut batch = LlamaBatch::new(n_ctx as usize, 1);
    for (i, token) in tokens.iter().enumerate() {
        if batch.add(*token, i as i32, &[0], i == tokens.len() - 1).is_err() {
            return String::new();
        }
    }
    if ctx.decode(&mut batch).is_err() {
        return String::new();
    }

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.1),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::dist(42),
    ]);

    let mut output = String::new();
    let mut n_cur = tokens.len() as i32;

    for _ in 0..256 {
        let token = sampler.sample(&ctx, -1);
        if model.is_eog_token(token) {
            break;
        }

        #[allow(deprecated)]
        match model.token_to_str(token, Special::Tokenize) {
            Ok(piece) => output.push_str(&piece),
            Err(_) => break,
        }

        let stop_markers = ["<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<start_of_turn>"];
        if let Some(pos) = stop_markers.iter().filter_map(|m| output.find(m)).min() {
            output.truncate(pos);
            break;
        }

        batch.clear();
        if batch.add(token, n_cur, &[0], true).is_err() {
            break;
        }
        n_cur += 1;
        if ctx.decode(&mut batch).is_err() {
            break;
        }
    }

    output.trim().to_string()
}
