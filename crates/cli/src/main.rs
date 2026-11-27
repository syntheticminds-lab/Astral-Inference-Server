// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral` command-line tool.
//!
//!   astral hw [--bench | --bench-fresh]   print detected hardware as JSON; --bench reuses a
//!                         cached bandwidth result for this GPU/driver/clocks, --bench-fresh re-measures
//!   astral selftest       run every GPU kernel against its CPU reference
//!   astral gguf-info <model> [--tensors]   summarise model weights (metadata, types, layout)
//!   astral verify-embd <file> <eval_callback.log>   match token_embd rows against a llama.cpp dump
//!   astral verify-model <file> <eval_callback.log>   run the prompt tokens and compare every layer slot
//!   astral generate <model> <n> <token ids...>   greedy decode n tokens, report tok/s
//!   astral serve <model> [--checkpoint <dir>] [options]   the OpenAI-compatible server
//!
//! `<model>` is a `.gguf` file or a safetensors checkpoint directory.
//!   astral kbench <mxfp4-gemv | mxfp4-mma | mxfp4-mma-f16> <n> <k> <b> [iters]   time one kernel (for Nsight Compute)
//!   astral version        print version and build information

#![forbid(unsafe_code)]

use astral_core::Result;
use std::process::ExitCode;

const USAGE: &str = "usage: astral <hw [--bench | --bench-fresh] | selftest | gguf-info <model> [--tensors] | verify-embd <file> <log> | verify-model <file> <log> | generate <model> <n> <ids...> | batch-check <model> <n> <steps> | spec-generate <model> <checkpoint dir> <k> <n> <ids...|@file> | serve <model> [--checkpoint <dir>] [--no-draft] [--k 6] [--host 0.0.0.0] [--port 8000] [--ctx 8192] [--name inkling-2-mini] [--queue 64] [--max-seqs N] | version>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("hw") => cmd_hw(if args.iter().any(|a| a == "--bench-fresh") {
            astral_hw::BenchMode::Fresh
        } else if args.iter().any(|a| a == "--bench") {
            astral_hw::BenchMode::Cached
        } else {
            astral_hw::BenchMode::Skip
        }),
        Some("selftest") => cmd_selftest(),
        Some("gguf-info") if args.len() >= 2 => cmd_gguf_info(&args[1], args.iter().any(|a| a == "--tensors")),
        Some("verify-embd") if args.len() >= 3 => cmd_verify_embd(&args[1], &args[2]),
        Some("verify-model") if args.len() >= 3 => cmd_verify_model(&args[1], &args[2]),
        Some("score") if args.len() >= 3 => cmd_score(&args[1], &args[2..]),
        Some("qwen-generate") if args.len() >= 4 => cmd_qwen_generate(&args[1], &args[2], &args[3..]),
        Some("generate") if args.len() >= 4 => cmd_generate(&args[1], &args[2], &args[3..]),
        Some("serve") => cmd_serve(&args[1..]),
        Some("batch-check") if args.len() >= 4 => cmd_batch_check(&args[1], &args[2], &args[3]),
        Some("cmp-prefill") if args.len() >= 3 => cmd_cmp_prefill(&args[1], &args[2..]),
        Some("probe") if args.len() >= 5 => cmd_probe(&args[1], &args[2], &args[3], &args[4]),
        Some("spec-generate") if args.len() >= 6 => cmd_spec_generate(&args[1], &args[2], &args[3], &args[4], &args[5..]),
        Some("kbench") if args.len() >= 5 && args[1] == "mxfp4-gemv" => cmd_kbench_mxfp4(&args[2..], false),
        Some("kbench") if args.len() >= 5 && args[1] == "mxfp4-mma" => cmd_kbench_mxfp4(&args[2..], true),
        Some("kbench") if args.len() >= 5 && args[1] == "mxfp4-mma-f16" => cmd_kbench_mxfp4_f16(&args[2..]),
        Some("kbench") if args.len() >= 2 && args[1] == "f16-tma" => cmd_kbench_f16_tma(&args[2..]),
        Some("version") | Some("--version") | Some("-V") => {
            cmd_version();
            Ok(true)
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn cmd_version() {
    println!("astral {}", env!("CARGO_PKG_VERSION"));
    println!("kernels built for: {:?}", astral_kernels::BUILT_ARCHS);
    println!("nvcc: {}", astral_kernels::NVCC_VERSION);
}

fn cmd_hw(mode: astral_hw::BenchMode) -> Result<bool> {
    let report = astral_hw::probe(mode)?;
    println!("{}", serde_json::to_string_pretty(&report).expect("report serialises"));
    Ok(true)
}

fn cmd_selftest() -> Result<bool> {
    let n = astral_gpu::Device::count()?;
    if n == 0 {
        eprintln!("no CUDA devices found");
        return Ok(false);
    }
    let dev = astral_gpu::Device::get(0)?;
    println!("device 0: {} (cc {:?})", dev.name()?, dev.compute_capability()?);
    let ctx = astral_gpu::Context::new(dev)?;
    let results = astral_testing::selftest::run(&ctx)?;
    let mut all = true;
    for r in &results {
        all &= r.pass;
        println!("{:<5} {:<20} {:<20} max_scaled_err={:.3e} (tol {:.1e})", if r.pass { "PASS" } else { "FAIL" }, r.kernel, r.case, r.error.max_scaled, r.tolerance);
        if let Some(info) = &r.info {
            println!("      {info}");
        }
    }
    println!("{} / {} cases passed", results.iter().filter(|r| r.pass).count(), results.len());
    Ok(all)
}

fn cmd_gguf_info(path: &str, list_tensors: bool) -> Result<bool> {
    let g = astral_loader::open_model(path)?;
    let arch = g.architecture().unwrap_or("?").to_string();
    println!("file        {}", g.path.display());
    println!("gguf        v{}  alignment {}  data offset {}  file {} bytes", g.version, g.alignment, g.data_offset, g.file_len);
    println!("arch        {arch}   name {}", g.get("general.name").map(|v| v.summary()).unwrap_or_default());
    println!("tensors     {}   metadata keys {}", g.tensors.len(), g.metadata.len());
    let (gap, data) = g.coverage();
    println!("coverage    {} of {} data bytes are alignment padding", gap, data);
    println!("types:");
    for (ty, (n, bytes)) in g.type_tally() {
        println!("  {ty:<8} {n:>5} tensors  {:>9.3} GB", bytes as f64 / 1e9);
    }
    println!("{arch}.* keys:");
    for (k, v) in &g.metadata {
        if k.starts_with(&format!("{arch}.")) {
            println!("  {k} = {}", v.summary());
        }
    }
    if list_tensors {
        for t in &g.tensors {
            println!("  {:<48} {:<7} {:?}", t.name, t.ty.name(), t.dims);
        }
    }
    Ok(true)
}

fn cmd_verify_embd(path: &str, log_path: &str) -> Result<bool> {
    let g = astral_loader::Gguf::open(path)?;
    let log = std::fs::read_to_string(log_path)?;
    let printed = astral_testing::golden::parse_first(&log, "embd").ok_or_else(|| astral_core::AstralError::Invalid("no `embd` tensor in the dump".into()))?;
    let m = astral_testing::golden::match_embedding_rows(&g, "token_embd.weight", &printed)?;
    println!("printed rows {}  matched token ids {:?}", printed.rows.len(), m.token_ids);
    println!("sum: astral {:.6}  llama.cpp {:.6}  diff {:.3e}", m.our_sum, m.printed_sum, m.our_sum - m.printed_sum);
    let ok = m.pass(1e-5);
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

/// The prompt of the llama.cpp eval-callback dump (identified by `verify-embd`).
const GOLDEN_PROMPT: [u32; 5] = [371, 151451, 149879, 151194, 149884];

fn load_session(path: &str, capacity: u32) -> Result<astral_model::inkling::Session> {
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let g = astral_loader::open_model(path)?;
    let t0 = std::time::Instant::now();
    let w = astral_model::inkling::Weights::load(&ctx, &g)?;
    eprintln!("loaded {:.2} GB of weights in {:.1} s", w.bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
    astral_model::inkling::Session::new(&ctx, w, capacity)
}

/// `batch-check <gguf> <n> <steps>`: decodes `n` sequences together and compares every row's
/// logits and greedy pick with the same sequence decoded alone (teacher-forced on the
/// single-sequence tokens, so every step compares the same inputs).
fn cmd_batch_check(path: &str, n: &str, steps: &str) -> Result<bool> {
    use astral_model::inkling::{DecodeRow, Session, Weights};
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let n: usize = n.parse().map_err(|_| bad("batch-check: n must be a number".into()))?;
    let steps: usize = steps.parse().map_err(|_| bad("batch-check: steps must be a number".into()))?;
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let g = astral_loader::open_model(path)?;
    let w = Weights::load(&ctx, &g)?;
    let mut s = Session::with_seqs(&ctx, w, 2048, n, n)?;
    let prompts: Vec<Vec<u32>> = (0..n).map(|i| (0..(6 + i % 5)).map(|j| ((i * 7919 + j * 104729) % 150000 + 1000) as u32).collect()).collect();
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
    // ASTRAL_CHECK_PREFILL=1: prefill logits of prompt 4 on slot 0 twice, then on every slot.
    if std::env::var_os("ASTRAL_CHECK_PREFILL").is_some() {
        let p = &prompts[4.min(n - 1)];
        let mut base = None;
        for q in [0usize, 0].into_iter().chain(1..n) {
            s.reset_seq(q)?;
            s.prefill_seq(q, 0, p)?;
            let l = s.logits()?;
            let b = base.get_or_insert_with(|| l.clone());
            let d = l.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("prefill on slot {q}: max |diff| {d:.3e}");
        }
        return Ok(true);
    }
    // Reference: each sequence alone (sequence 0, one-token graph).
    let (mut rt, mut rl) = (Vec::new(), Vec::new());
    for p in &prompts {
        s.reset()?;
        let mut t = s.prefill(p)?;
        let (mut toks, mut lg) = (vec![t], vec![s.logits()?]);
        for _ in 1..steps {
            t = s.step(t)?;
            toks.push(t);
            lg.push(s.logits()?);
        }
        rt.push(toks);
        rl.push(lg);
    }
    s.reset()?;
    let mut ok = true;
    // All prompts prefilled together (packed passes): each must match its own prefill.
    for i in 0..n {
        s.reset_seq(i)?;
    }
    let jobs: Vec<(usize, &[u32])> = prompts.iter().enumerate().map(|(i, p)| (i, p.as_slice())).collect();
    let firsts = s.prefill_many(&jobs)?;
    for (i, &t) in firsts.iter().enumerate() {
        if t != rt[i][0] {
            println!("seq {i}: prefill pick {t} != {}", rt[i][0]);
            ok = false;
        }
    }
    let vocab = rl[0][0].len();
    for k in 0..steps - 1 {
        let rows: Vec<DecodeRow> = (0..n).map(|i| DecodeRow { seq: i, pos: (prompts[i].len() + k) as u32, token: rt[i][k] }).collect();
        let next = s.decode_batch(&rows)?;
        let lg = s.batch_logits(n)?;
        let (mut maxd, mut picks) = (0.0f32, 0usize);
        for i in 0..n {
            let row = &lg[i * vocab..(i + 1) * vocab];
            let r = &rl[i][k + 1];
            let rd = row.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            if rd > 0.0 && std::env::var_os("ASTRAL_CHECK_ROWS").is_some() {
                println!("  row {i} (prompt {} tokens): {rd:.3e}", prompts[i].len());
            }
            maxd = maxd.max(rd);
            if next[i] != rt[i][k + 1] || argmax(row) != next[i] {
                picks += 1;
            }
        }
        println!("step {k:3}: max |logit diff| {maxd:.3e}, {picks} / {n} picks differ");
        ok &= picks == 0 && maxd == 0.0;
    }
    // Timing: one batched step at this batch size.
    let rows: Vec<DecodeRow> = (0..n).map(|i| DecodeRow { seq: i, pos: (prompts[i].len() + steps) as u32, token: (1000 + i * 7919 % 150000) as u32 }).collect();
    s.decode_batch(&rows)?;
    if std::env::var_os("ASTRAL_PROFILE").is_some() {
        astral_gpu::profiler(true)?;
        s.decode_batch(&rows)?;
        astral_gpu::profiler(false)?;
        return Ok(ok);
    }
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        s.decode_batch(&rows)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("batch {n}: {ms:.2} ms per step, {:.0} tok/s", n as f64 * 1e3 / ms);
    let long: Vec<u32> = (0..512).map(|j| (j * 7919 % 150000 + 1000) as u32).collect();
    s.reset_seq(0)?;
    s.prefill_seq(0, 0, &long)?;
    if std::env::var_os("ASTRAL_PROFILE_PREFILL").is_some() {
        s.reset_seq(0)?;
        astral_gpu::profiler(true)?;
        s.prefill_seq(0, 0, &long)?;
        astral_gpu::profiler(false)?;
        return Ok(ok);
    }
    let t0 = std::time::Instant::now();
    s.reset_seq(0)?;
    s.prefill_seq(0, 0, &long)?;
    println!("prefill 512 tokens: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    if n >= 4 {
        let k = n.min(16);
        for i in 0..k {
            s.reset_seq(i)?;
        }
        let jobs: Vec<(usize, &[u32])> = (0..k).map(|i| (i, long.as_slice())).collect();
        let t0 = std::time::Instant::now();
        let prof = std::env::var_os("ASTRAL_PROFILE_PACKED").is_some();
        if prof {
            astral_gpu::profiler(true)?;
        }
        s.prefill_many(&jobs)?;
        if prof {
            astral_gpu::profiler(false)?;
        }
        println!("prefill {k} x 512 tokens packed: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(ok)
}

/// Diagnostic load generator: `probe <model> decode <n> <secs>` runs batched decode steps of n
/// rows, `probe <model> prefill <k> <secs>` packed prefills of k x 512 tokens, for `secs`
/// seconds, printing the time per step / pass every second.
fn cmd_probe(path: &str, what: &str, n: &str, secs: &str) -> Result<bool> {
    use astral_model::inkling::{DecodeRow, Session, Weights};
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    let n: usize = n.parse().map_err(|_| bad("probe: n must be a number".into()))?;
    let secs: f64 = secs.parse().map_err(|_| bad("probe: secs must be a number".into()))?;
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let g = astral_loader::open_model(path)?;
    let w = Weights::load(&ctx, &g)?;
    let cap = std::env::var("ASTRAL_PROBE_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let mut s = Session::with_seqs(&ctx, w, cap, n, n)?;
    let long: Vec<u32> = (0..512).map(|j| (j * 7919 % 150000 + 1000) as u32).collect();
    let t_end = std::time::Instant::now() + std::time::Duration::from_secs_f64(secs);
    let (mut cnt, mut tick) = (0usize, std::time::Instant::now());
    match what {
        "decode" => {
            for q in 0..n {
                s.reset_seq(q)?;
                s.prefill_seq(q, 0, &long[..64])?;
            }
            let rows: Vec<DecodeRow> = (0..n).map(|q| DecodeRow { seq: q, pos: 600, token: (1000 + q * 7919 % 150000) as u32 }).collect();
            // ASTRAL_PROFILE=1: one warmed-up step under the profiler, then exit.
            if std::env::var_os("ASTRAL_PROFILE").is_some() {
                for _ in 0..5 {
                    s.decode_batch(&rows)?;
                }
                astral_gpu::profiler(true)?;
                s.decode_batch(&rows)?;
                astral_gpu::profiler(false)?;
                return Ok(true);
            }
            while std::time::Instant::now() < t_end {
                s.decode_batch(&rows)?;
                cnt += 1;
                if tick.elapsed().as_secs_f64() >= 1.0 {
                    println!("decode {n}: {:.2} ms per step", tick.elapsed().as_secs_f64() * 1e3 / cnt as f64);
                    (cnt, tick) = (0, std::time::Instant::now());
                }
            }
        }
        "interleave" => {
            // Server-like: a mixed pass (n - 1 decoding rows + a prompt chunk) every 10 decode
            // steps of the n - 1 rows; prints the decode-step time.
            for q in 0..n {
                s.reset_seq(q)?;
                s.prefill_seq(q, 0, &long[..64])?;
            }
            let chunk: Vec<u32> = (0..512).map(|j| (j * 7919 % 150000 + 1000) as u32).collect();
            let one = [1000u32];
            let rows: Vec<DecodeRow> = (0..n - 1).map(|q| DecodeRow { seq: q, pos: 600, token: (1000 + q * 7919 % 150000) as u32 }).collect();
            let mut dt = 0f64;
            let mut it = 0usize;
            while std::time::Instant::now() < t_end {
                if std::env::var_os("ASTRAL_NO_MIX").is_none() {
                    s.reset_seq(n - 1)?;
                    let mut parts: Vec<(usize, u32, &[u32])> = (0..n - 1).map(|q| (q, 600u32, &one[..])).collect();
                    parts.push((n - 1, 0, &chunk[..]));
                    s.step_mixed(&parts)?;
                }
                for _ in 0..10 {
                    let t0 = std::time::Instant::now();
                    s.decode_batch(&rows)?;
                    dt += t0.elapsed().as_secs_f64();
                    cnt += 1;
                }
                it += 1;
                if tick.elapsed().as_secs_f64() >= 2.0 {
                    println!("interleave {}: decode {:.2} ms per step ({it} mixed passes)", n - 1, dt * 1e3 / cnt as f64);
                    (cnt, dt, tick) = (0, 0.0, std::time::Instant::now());
                }
            }
        }
        "speccheck" => {
            // Batched speculative decoding over n sequences with distinct prompts: 96 tokens each,
            // against plain batched greedy decode (same prompts).
            let k: usize = std::env::var("ASTRAL_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
            s.enable_spec(std::path::Path::new(path), k)?;
            s.enable_batch_spec(n)?;
            let mut rng = 0x2545f4914f6cdd1du64;
            let prompts: Vec<Vec<u32>> = (0..n)
                .map(|_| {
                    (0..256)
                        .map(|_| {
                            rng ^= rng << 13;
                            rng ^= rng >> 7;
                            rng ^= rng << 17;
                            (1000 + rng % 150000) as u32
                        })
                        .collect()
                })
                .collect();
            let len = 96usize;
            // Plain greedy reference.
            let mut plain = vec![Vec::new(); n];
            let mut last = Vec::new();
            for q in 0..n {
                s.reset_seq(q)?;
                last.push(s.prefill_seq(q, 0, &prompts[q])?);
                plain[q].push(last[q]);
            }
            let t0 = std::time::Instant::now();
            for step in 0..len - 1 {
                let rows: Vec<DecodeRow> = (0..n).map(|q| DecodeRow { seq: q, pos: (256 + step) as u32, token: last[q] }).collect();
                last = s.decode_batch(&rows)?;
                for q in 0..n {
                    plain[q].push(last[q]);
                }
            }
            let tp = t0.elapsed().as_secs_f64();
            // Speculative.
            let mut out = vec![Vec::new(); n];
            let mut st: Vec<(usize, u32, u32, Vec<u32>)> = Vec::new();
            for q in 0..n {
                s.reset_seq(q)?;
                let nx = s.prefill_seq_spec(q, 0, &prompts[q])?;
                out[q].push(nx);
                let d = s.spec_first_draft(q, 256, nx)?;
                st.push((q, 256, nx, d));
            }
            let (t1, mut rounds) = (std::time::Instant::now(), 0usize);
            while out.iter().any(|o| o.len() < len) {
                let res = s.spec_round(&st)?;
                rounds += 1;
                for (e, (acc, dr)) in res.into_iter().enumerate() {
                    let q = st[e].0;
                    out[q].extend_from_slice(&acc);
                    st[e].1 += acc.len() as u32;
                    st[e].2 = *acc.last().expect("non-empty");
                    st[e].3 = dr;
                }
            }
            let ts = t1.elapsed().as_secs_f64();
            let made: usize = out.iter().map(|o| o.len() - 1).sum();
            let same: usize = out.iter().zip(&plain).map(|(a, b)| a.iter().zip(b).take_while(|(x, y)| x == y).count().min(len)).sum();
            println!(
                "speccheck {n} x k{k}: {rounds} rounds, {:.2} tokens/round/seq, {:.2} ms/round; spec {:.0} tok/s vs plain {:.0} tok/s; prefix equal to plain {same} / {}",
                made as f64 / (rounds * n) as f64,
                ts * 1e3 / rounds as f64,
                made as f64 / ts,
                (n * (len - 1)) as f64 / tp,
                n * len
            );
            return Ok(true);
        }
        "chaincheck" => {
            // Chained verification on the decode kernels: 3 tokens per sequence must pick what
            // three decode steps pick; after keeping only the first, the next step must match a
            // sequence that only saw the first. Then times chains of 1..=4 tokens.
            let toks: Vec<Vec<u32>> = (0..n).map(|q| (0..4).map(|j| (1000 + (q * 4 + j) * 7919 % 150000) as u32).collect()).collect();
            let fresh = |s: &mut Session| -> Result<()> {
                for q in 0..n {
                    s.reset_seq(q)?;
                    s.prefill_seq(q, 0, &long[..64])?;
                }
                Ok(())
            };
            fresh(&mut s)?;
            let rows: Vec<DecodeRow> = (0..n).flat_map(|q| (0..3).map(move |j| (q, j))).map(|(q, j)| DecodeRow { seq: q, pos: 64 + j as u32, token: toks[q][j] }).collect();
            let a = s.decode_verify(&rows)?;
            let keep: Vec<(usize, usize)> = (0..n).map(|q| (q, 3 * q)).collect();
            s.commit_verify(&keep)?;
            let after: Vec<DecodeRow> = (0..n).map(|q| DecodeRow { seq: q, pos: 65, token: toks[q][3] }).collect();
            let a2 = s.decode_batch(&after)?;
            fresh(&mut s)?;
            let mut b = vec![Vec::new(); n];
            for j in 0..3 {
                let r: Vec<DecodeRow> = (0..n).map(|q| DecodeRow { seq: q, pos: 64 + j as u32, token: toks[q][j] }).collect();
                for (q, t) in s.decode_batch(&r)?.into_iter().enumerate() {
                    b[q].push(t);
                }
            }
            fresh(&mut s)?;
            s.decode_batch(&(0..n).map(|q| DecodeRow { seq: q, pos: 64, token: toks[q][0] }).collect::<Vec<_>>())?;
            let b2 = s.decode_batch(&after)?;
            let by: Vec<usize> = (0..3).map(|j| (0..n).filter(|&q| a[3 * q + j] == b[q][j]).count()).collect();
            let roll = a2.iter().zip(&b2).filter(|(x, y)| x == y).count();
            println!("chaincheck: picks equal decode's by position {by:?} of {n}; after rollback {roll} / {n}");
            for v in 1..=4usize {
                let rows: Vec<DecodeRow> = (0..n).flat_map(|q| (0..v).map(move |j| (q, j))).map(|(q, j)| DecodeRow { seq: q, pos: 600 + j as u32, token: toks[q][j] }).collect();
                s.decode_verify(&rows)?;
                let t0 = std::time::Instant::now();
                let mut c = 0;
                while t0.elapsed().as_secs_f64() < 2.0 {
                    s.decode_verify(&rows)?;
                    c += 1;
                }
                println!("chain {n} x {v}: {:.2} ms per pass", t0.elapsed().as_secs_f64() * 1e3 / c as f64);
            }
            return Ok(true);
        }
        "verifycheck" => {
            // A packed pass of 3 tokens per sequence must pick what three decode steps pick.
            let toks: Vec<Vec<u32>> = (0..n).map(|q| (0..3).map(|j| (1000 + (q * 3 + j) * 7919 % 150000) as u32).collect()).collect();
            for q in 0..n {
                s.reset_seq(q)?;
                s.prefill_seq(q, 0, &long[..64])?;
            }
            let parts: Vec<(usize, u32, &[u32])> = (0..n).map(|q| (q, 64u32, &toks[q][..])).collect();
            let a = s.step_verify(&parts)?;
            for q in 0..n {
                s.reset_seq(q)?;
                s.prefill_seq(q, 0, &long[..64])?;
            }
            let mut b = vec![Vec::new(); n];
            for j in 0..3 {
                let rows: Vec<DecodeRow> = (0..n).map(|q| DecodeRow { seq: q, pos: 64 + j as u32, token: toks[q][j] }).collect();
                for (q, t) in s.decode_batch(&rows)?.into_iter().enumerate() {
                    b[q].push(t);
                }
            }
            let same: usize = a.iter().zip(&b).map(|(x, y)| x.iter().zip(y).filter(|(u, v)| u == v).count()).sum();
            let by: Vec<usize> = (0..3).map(|j| a.iter().zip(&b).filter(|(x, y)| x[j] == y[j]).count()).collect();
            // The same rows through a plain prompt pass (the kernels verification uses).
            let mut pf = vec![Vec::new(); n];
            for j in 0..3 {
                for q in 0..n {
                    s.reset_seq(q)?;
                    let mut p = long[..64].to_vec();
                    p.extend_from_slice(&toks[q][..=j]);
                    pf[q].push(s.prefill_seq(q, 0, &p)?);
                }
            }
            let byp: Vec<usize> = (0..3).map(|j| a.iter().zip(&pf).filter(|(x, y)| x[j] == y[j]).count()).collect();
            println!("verifycheck: verify vs prompt pass by position {byp:?}");
            println!("verifycheck: {same} / {} picks equal decode's (by position {by:?})", 3 * n);
            return Ok(same * 10 >= 3 * n * 9);
        }
        "verify" => {
            // Cost of a pass where each of n sequences feeds v tokens (v = 1..=4), the shape
            // of a batched speculative verification.
            for q in 0..n {
                s.reset_seq(q)?;
                s.prefill_seq(q, 0, &long[..64])?;
            }
            let toks: Vec<Vec<u32>> = (0..n).map(|q| (0..4).map(|j| (1000 + (q * 4 + j) * 7919 % 150000) as u32).collect()).collect();
            let per = (t_end - std::time::Instant::now()).as_secs_f64() / 4.0;
            for v in 1..=4usize {
                let parts: Vec<(usize, u32, &[u32])> = (0..n).map(|q| (q, 600u32, &toks[q][..v])).collect();
                let step = |s: &mut Session, p: &[(usize, u32, &[u32])]| -> Result<()> { if v > 1 { s.step_verify(p).map(|_| ()) } else { s.step_mixed(p).map(|_| ()) } };
                step(&mut s, &parts)?;
                if std::env::var("ASTRAL_PROFILE").ok().and_then(|x| x.parse::<usize>().ok()) == Some(v) {
                    astral_gpu::profiler(true)?;
                    step(&mut s, &parts)?;
                    astral_gpu::profiler(false)?;
                    return Ok(true);
                }
                let (t0, mut c) = (std::time::Instant::now(), 0usize);
                while t0.elapsed().as_secs_f64() < per {
                    step(&mut s, &parts)?;
                    c += 1;
                }
                println!("verify {n} x {v}: {:.2} ms per pass", t0.elapsed().as_secs_f64() * 1e3 / c as f64);
            }
        }
        "mixcheck" => {
            // Greedy picks of n - 1 sequences decoded 8 steps by decode_batch, then again by
            // step_mixed passes that also prefill a prompt into sequence n - 1.
            let prompts: Vec<Vec<u32>> = (0..n - 1).map(|q| (0..40 + 7 * q).map(|j| ((j * 7919 + q * 104729) % 150000 + 1000) as u32).collect()).collect();
            let run = |s: &mut Session, mixed: bool| -> Result<Vec<Vec<u32>>> {
                let mut last = Vec::new();
                for (q, p) in prompts.iter().enumerate() {
                    s.reset_seq(q)?;
                    last.push(s.prefill_seq(q, 0, p)?);
                }
                let mut picks = vec![Vec::new(); n - 1];
                let extra: Vec<u32> = (0..300).map(|j| (j * 31 % 150000 + 2000) as u32).collect();
                s.reset_seq(n - 1)?;
                for k in 0..8u32 {
                    let next = if mixed {
                        let toks: Vec<[u32; 1]> = last.iter().map(|&t| [t]).collect();
                        let mut parts: Vec<(usize, u32, &[u32])> = (0..n - 1).map(|q| (q, prompts[q].len() as u32 + k, &toks[q][..])).collect();
                        parts.push((n - 1, 30 * k, &extra[(30 * k) as usize..(30 * k + 30) as usize]));
                        let mut v = s.step_mixed(&parts)?;
                        v.truncate(n - 1);
                        v
                    } else {
                        let rows: Vec<DecodeRow> = (0..n - 1).map(|q| DecodeRow { seq: q, pos: prompts[q].len() as u32 + k, token: last[q] }).collect();
                        s.decode_batch(&rows)?
                    };
                    for q in 0..n - 1 {
                        picks[q].push(next[q]);
                    }
                    last = next;
                }
                Ok(picks)
            };
            let a = run(&mut s, false)?;
            let b = run(&mut s, true)?;
            let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
            println!("mixcheck: {same} / {} sequences give the same 8 tokens", n - 1);
            for (q, (x, y)) in a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y).take(4) {
                println!("  seq {q}: batch {x:?} mixed {y:?}");
            }
            return Ok(same == n - 1);
        }
        "mixed" => {
            // n - 1 decoding sequences (one token each) plus one prompt chunk filling the budget.
            for q in 0..n {
                s.reset_seq(q)?;
                s.prefill_seq(q, 0, &long[..64])?;
            }
            let budget = s.token_budget();
            let clen = std::env::var("ASTRAL_MIX_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(budget - (n - 1));
            let chunk: Vec<u32> = (0..clen).map(|j| (j * 7919 % 150000 + 1000) as u32).collect();
            let one = [1000u32];
            while std::time::Instant::now() < t_end {
                s.reset_seq(n - 1)?;
                let mut parts: Vec<(usize, u32, &[u32])> = (0..n - 1).map(|q| (q, 600u32, &one[..])).collect();
                parts.push((n - 1, 0, &chunk[..]));
                let prof = cnt == 0 && std::env::var_os("ASTRAL_PROFILE").is_some();
                if prof {
                    astral_gpu::profiler(true)?;
                }
                s.step_mixed(&parts)?;
                if prof {
                    astral_gpu::profiler(false)?;
                    return Ok(true);
                }
                cnt += 1;
                if tick.elapsed().as_secs_f64() >= 1.0 {
                    println!("mixed {} + {}: {:.1} ms per pass", n - 1, chunk.len(), tick.elapsed().as_secs_f64() * 1e3 / cnt as f64);
                    (cnt, tick) = (0, std::time::Instant::now());
                }
            }
        }
        _ => {
            let jobs: Vec<(usize, &[u32])> = (0..n).map(|i| (i, long.as_slice())).collect();
            while std::time::Instant::now() < t_end {
                for i in 0..n {
                    s.reset_seq(i)?;
                }
                s.prefill_many(&jobs)?;
                cnt += 1;
                if tick.elapsed().as_secs_f64() >= 1.0 {
                    let ms = tick.elapsed().as_secs_f64() * 1e3 / cnt as f64;
                    println!("prefill {n} x 512: {ms:.1} ms per pass, {:.0} tok/s", (n * 512) as f64 * 1e3 / ms);
                    (cnt, tick) = (0, std::time::Instant::now());
                }
            }
        }
    }
    Ok(true)
}

/// Max |a - b| over the printed first/last three values of each row.
fn row_diff(rows: &[Vec<f32>], printed: &astral_testing::golden::PrintedTensor) -> f64 {
    let mut m = 0.0f64;
    for (r, (head, tail)) in rows.iter().zip(&printed.rows) {
        let n = r.len();
        for i in 0..3 {
            m = m.max((r[i] as f64 - head[i]).abs()).max((r[n - 3 + i] as f64 - tail[i]).abs());
        }
    }
    m
}

fn cmd_verify_model(path: &str, log_path: &str) -> Result<bool> {
    let log = std::fs::read_to_string(log_path)?;
    let mut s = load_session(path, 64)?;
    s.capture = true;
    s.set_debug(true);
    let mut dbg: std::collections::BTreeMap<&'static str, Vec<Vec<f32>>> = Default::default();
    let mut order: Vec<&'static str> = Vec::new();
    let mut per_slot: std::collections::BTreeMap<usize, Vec<Vec<f32>>> = Default::default();
    let mut next = 0;
    for &t in &GOLDEN_PROMPT {
        next = s.step(t)?;
        for (slot, v) in s.captured.drain(..) {
            per_slot.entry(slot).or_default().push(v);
        }
        for (name, v) in s.take_debug() {
            if !order.contains(&name) {
                order.push(name);
            }
            dbg.entry(name).or_default().push(v);
        }
    }
    for name in &order {
        let rows = &dbg[name];
        let ours: f64 = rows.iter().flatten().map(|&x| x as f64).sum();
        match astral_testing::golden::parse_first(&log, &format!("{name}-{}", astral_model::inkling::tap_layer())) {
            Some(p) if p.rows.len() == rows.len() => {
                println!("  {name:<16} max|d| {:.2e}  sum astral {ours:>12.4} llama.cpp {:>12.4}", row_diff(rows, &p), p.sum)
            }
            Some(p) => println!("  {name:<16} sum astral {ours:>12.4} llama.cpp {:>12.4} (printed as {} rows)", p.sum, p.rows.len()),
            None => println!("  {name:<16} sum astral {ours:>12.4}  first {:?}", &rows[0][..3]),
        }
    }
    let mut ok = true;
    for (slot, rows) in &per_slot {
        let Some(p) = astral_testing::golden::parse_first(&log, &format!("l_out-{slot}")) else {
            println!("l_out-{slot:<2}  (not in the dump)");
            continue;
        };
        let sum: f64 = rows.iter().flatten().map(|&x| x as f64).sum();
        let diff = row_diff(rows, &p);
        let rel = (sum - p.sum).abs() / p.sum.abs().max(1.0);
        let pass = diff <= 0.05 * rows.iter().flatten().fold(1.0f32, |m, &x| m.max(x.abs())) as f64;
        ok &= pass;
        println!("l_out-{slot:<2}  max|d| {diff:.2e}  sum astral {sum:>12.4} llama.cpp {:>12.4} (rel {rel:.1e})  {}", p.sum, if pass { "ok" } else { "DIFF" });
    }
    let logits = s.logits()?;
    if let Some(p) = astral_testing::golden::parse_first(&log, "result_output") {
        let sum: f64 = logits.iter().map(|&x| x as f64).sum();
        let diff = row_diff(std::slice::from_ref(&logits), &p);
        println!("decode  result_output  max|d| {diff:.2e}  sum astral {sum:.4} llama.cpp {:.4}", p.sum);
    }

    // The same prompt as one batched prefill (llama.cpp's dump is itself a prefill).
    s.reset()?;
    s.set_debug(false);
    let next_p = s.prefill(&GOLDEN_PROMPT)?;
    println!("prefill:");
    let d = s.w.cfg.dim;
    for (slot, v) in s.captured.clone() {
        let Some(p) = astral_testing::golden::parse_first(&log, &format!("l_out-{slot}")) else { continue };
        let rows: Vec<Vec<f32>> = v.chunks(d).map(|c| c.to_vec()).collect();
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let diff = row_diff(&rows, &p);
        let pass = diff <= 0.05 * rows.iter().flatten().fold(1.0f32, |m, &x| m.max(x.abs())) as f64;
        ok &= pass;
        println!("l_out-{slot:<2}  max|d| {diff:.2e}  sum astral {sum:>12.4} llama.cpp {:>12.4}  {}", p.sum, if pass { "ok" } else { "DIFF" });
    }
    println!("prefill greedy next token: {next_p} (decode path: {next})");
    ok &= next_p == next;
    let logits = s.logits()?;
    if let Some(p) = astral_testing::golden::parse_first(&log, "result_output") {
        let sum: f64 = logits.iter().map(|&x| x as f64).sum();
        let diff = row_diff(std::slice::from_ref(&logits), &p);
        println!("result_output  max|d| {diff:.2e}  sum astral {sum:.4} llama.cpp {:.4}", p.sum);
        ok &= diff < 0.25;
    }
    println!("greedy next token after the prompt: {next}");
    println!("{}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

/// Teacher-forced: after each token, our greedy pick and the top-2 logit margin.
fn cmd_score(path: &str, ids: &[String]) -> Result<bool> {
    let toks: Vec<u32> = ids.iter().filter_map(|t| t.parse().ok()).collect();
    let mut s = load_session(path, (toks.len() + 1) as u32)?;
    let mut agree = 0;
    for (i, &t) in toks.iter().enumerate() {
        let next = s.step(t)?;
        let l = s.logits()?;
        let mut best = (f32::MIN, 0usize, f32::MIN);
        for (j, &v) in l.iter().enumerate() {
            if v > best.0 {
                best = (v, j, best.0);
            } else if v > best.2 {
                best.2 = v;
            }
        }
        if let Some(&want) = toks.get(i + 1) {
            let ok = next == want;
            agree += ok as usize;
            println!("pos {i:>3}  next {next:>6}  want {want:>6}  {}  margin {:.3}  logit[want] {:.3}", if ok { "  " } else { "XX" }, best.0 - best.2, l[want as usize]);
        }
    }
    println!("agreement {agree}/{}", toks.len() - 1);
    Ok(true)
}

fn cmd_qwen_generate(dir: &str, n: &str, ids: &[String]) -> Result<bool> {
    let n: usize = n.parse().map_err(|_| astral_core::AstralError::Invalid("n must be a number".into()))?;
    let prompt: Vec<u32> = ids.iter().filter_map(|t| t.parse().ok()).collect();
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let t0 = std::time::Instant::now();
    let w = astral_model::qwen35::load(&ctx, std::path::Path::new(dir))?;
    eprintln!("loaded {:.2} GB in {:.1} s ({} MXFP4 blocks re-encoded)", w.bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), w.reencoded_blocks);
    let warm: usize = std::env::var("ASTRAL_WARM_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut s = astral_model::qwen35::Session::new(&ctx, w, (prompt.len() + n + 1).max(warm).min(8192) as u32)?;
    if warm > 0 {
        let toks: Vec<u32> = prompt.iter().cycle().take(warm).copied().collect();
        s.prefill(&toks)?;
        s.reset()?;
        let t = std::time::Instant::now();
        s.prefill(&toks)?;
        println!("prefill {warm} tokens: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        s.reset()?;
    }
    let t1 = std::time::Instant::now();
    let mut next = if std::env::var_os("ASTRAL_NO_PREFILL").is_some() {
        let mut x = 0;
        for &t in &prompt {
            x = s.step(t)?;
        }
        x
    } else {
        s.prefill(&prompt)?
    };
    let tp = t1.elapsed().as_secs_f64();
    let mut out = vec![next];
    let t2 = std::time::Instant::now();
    for _ in 1..n {
        next = s.step(next)?;
        out.push(next);
    }
    let td = t2.elapsed().as_secs_f64();
    println!("tokens: {out:?}");
    println!("prompt {} tok in {:.3} s; decode {} tok at {:.1} tok/s", prompt.len(), tp, n.saturating_sub(1), (n.saturating_sub(1)) as f64 / td.max(1e-9));
    Ok(true)
}

fn cmd_generate(path: &str, n: &str, ids: &[String]) -> Result<bool> {
    let n: usize = n.parse().map_err(|_| astral_core::AstralError::Invalid("n must be a number".into()))?;
    let prompt: Vec<u32> = ids.iter().filter_map(|t| t.parse().ok()).collect();
    if prompt.is_empty() {
        return Err(astral_core::AstralError::Invalid("no token ids".into()));
    }
    let warm_len: usize = std::env::var("ASTRAL_WARM_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let cap = std::env::var("ASTRAL_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or((prompt.len() + n + 1).max(warm_len).min(8192) as u32);
    let mut s = load_session(path, cap)?;
    if let Ok(w) = std::env::var("ASTRAL_TOUCH") {
        s.touch(&w)?;
    }
    if let Ok(warm) = std::env::var("ASTRAL_WARM_PREFILL") {
        // Time a prefill of `warm` tokens of the prompt repeated (cold start excluded).
        let k: usize = warm.parse().unwrap_or(512);
        let toks: Vec<u32> = prompt.iter().cycle().take(k).copied().collect();
        s.prefill(&toks)?;
        s.reset()?;
        let t = std::time::Instant::now();
        s.prefill(&toks)?;
        println!("prefill {k} tokens: {:.1} ms ({:.0} tok/s)", t.elapsed().as_secs_f64() * 1e3, k as f64 / t.elapsed().as_secs_f64());
        s.reset()?;
    }
    let t0 = std::time::Instant::now();
    let mut next = if std::env::var_os("ASTRAL_NO_PREFILL").is_some() {
        let mut n = 0;
        for &t in &prompt {
            n = s.step(t)?;
        }
        n
    } else {
        s.prefill(&prompt)?
    };
    let tp = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    let t1 = std::time::Instant::now();
    for _ in 1..n {
        next = s.step(next)?;
        out.push(next);
    }
    let td = t1.elapsed().as_secs_f64();
    println!("tokens: {out:?}");
    println!("prompt {} tok in {:.3} s; decode {} tok at {:.1} tok/s", prompt.len(), tp, n - 1, (n - 1) as f64 / td.max(1e-9));
    Ok(true)
}

/// `cmp-prefill <gguf> <ids...|@file>`: last-token logits of a batched prefill vs feeding the
/// same tokens one decode step at a time (max |diff|, argmax of each).
fn cmd_cmp_prefill(path: &str, ids: &[String]) -> Result<bool> {
    let prompt = read_ids(ids)?;
    let mut s = load_session(path, (prompt.len() + 8) as u32)?;
    for &t in &prompt {
        s.step(t)?;
    }
    let a = s.logits()?;
    s.reset()?;
    s.prefill(&prompt)?;
    let b = s.logits()?;
    let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
    let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("{} tokens: max |logit diff| {diff:.4}; argmax step {:?} prefill {:?}", prompt.len(), am(&a), am(&b));
    Ok(true)
}

/// `serve <model> [--checkpoint <dir>] [--no-draft] [--k N] [--host H] [--port P] [--ctx N]
/// [--name S] [--queue N] [--max-seqs N]`: the OpenAI-compatible server. `<model>` is a `.gguf` file or a
/// safetensors checkpoint directory. The checkpoint directory provides the tokenizer and the
/// DSpark draft (used unless `--no-draft`); it defaults to `<model>` when that is a directory.
const SERVE_USAGE: &str = "\
usage: astral serve <model> [options]

<model> is a local .gguf file, a local checkpoint directory (safetensors), a Hugging Face
repository id (org/name) or one GGUF file of a repository (org/name/file.gguf). Repositories
are downloaded on first use (set HF_TOKEN for gated or private ones).

Model:
  --tokenizer <dir|org/name>        tokenizer and chat template (default: the model's own)
  --revision <rev>                  branch, tag or commit of a repository (default: main)
  --download-dir <dir>              where repositories are downloaded (default: ~/.cache/astral/hub)
  --served-model-name <name>        model name in the API (default: the model's name)
Server:
  --host <addr>                     listen address (default: 0.0.0.0)
  --port <port>                     listen port (default: 8000)
Memory and batching:
  --max-model-len <n>               context length per request (default: 8192)
  --max-num-seqs <n>                requests decoded together (default: as many as fit, up to 64)
  --max-num-batched-tokens <n>      tokens per forward pass, prompt chunks included (default: 2048)
  --gpu-memory-utilization <f>      fraction of GPU memory Astral may use (default: all but a margin)
  --kv-cache-dtype <auto|fp16>      KV cache precision (default: auto = fp16)
  --max-queue <n>                   requests waiting beyond the running ones (default: 64)
Speculative decoding:
  --no-draft                        no draft model (default: the checkpoint's draft when it has one)
  --num-speculative-tokens <k>      drafted tokens per round (default: 6)
";

fn cmd_serve(a: &[String]) -> Result<bool> {
    let bad = |m: String| astral_core::AstralError::Invalid(m);
    if a.is_empty() || a.iter().any(|x| x == "-h" || x == "--help") {
        print!("{SERVE_USAGE}");
        return Ok(!a.is_empty());
    }
    let model = a[0].clone();
    let mut opts = astral_loader::hub::HubOptions::default();
    let (mut tokenizer, mut name, mut draft, mut k, mut host, mut port, mut ctx, mut queue, mut max_seqs) = (None::<String>, None::<String>, true, 6usize, "0.0.0.0".to_string(), 8000u16, 8192u32, 64usize, 0usize);
    let (mut util, mut batched) = (None::<f32>, None::<usize>);
    let mut i = 1;
    while i < a.len() {
        // Both "--flag value" and "--flag=value".
        let (flag, inline) = match a[i].split_once('=') {
            Some((f, v)) if a[i].starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a[i].clone(), None),
        };
        let mut val = || -> Result<String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            i += 1;
            a.get(i).cloned().ok_or_else(|| bad(format!("serve: {flag} needs a value")))
        };
        fn num<T: std::str::FromStr>(flag: &str, v: String) -> Result<T> {
            v.parse().map_err(|_| astral_core::AstralError::Invalid(format!("serve: {flag} must be a number, got {v:?}")))
        }
        match flag.as_str() {
            "--tokenizer" | "--checkpoint" => tokenizer = Some(val()?),
            "--revision" => opts.revision = Some(val()?),
            "--download-dir" => opts.download_dir = Some(val()?.into()),
            "--served-model-name" | "--name" => name = Some(val()?),
            "--host" => host = val()?,
            "--port" => port = num(&flag, val()?)?,
            "--max-model-len" | "--ctx" => ctx = num(&flag, val()?)?,
            "--max-num-seqs" | "--max-seqs" => max_seqs = num(&flag, val()?)?,
            "--max-num-batched-tokens" => batched = Some(num(&flag, val()?)?),
            "--max-queue" | "--queue" => queue = num(&flag, val()?)?,
            "--gpu-memory-utilization" => {
                let u: f32 = num(&flag, val()?)?;
                if !(0.05..=1.0).contains(&u) {
                    return Err(bad(format!("serve: --gpu-memory-utilization must be in (0, 1], got {u}")));
                }
                util = Some(u);
            }
            "--kv-cache-dtype" => match val()?.to_ascii_lowercase().as_str() {
                "auto" | "fp16" | "float16" | "half" => {}
                o => return Err(bad(format!("serve: --kv-cache-dtype {o} is not supported yet (auto or fp16)"))),
            },
            "--no-draft" => draft = false,
            "--num-speculative-tokens" | "--k" => k = num(&flag, val()?)?,
            o => return Err(bad(format!("serve: unknown option {o} (astral serve --help lists them)"))),
        }
        i += 1;
    }
    if let Some(t) = batched
        && !(64..=astral_model::inkling::MAX_PASS_ROWS).contains(&t)
    {
        return Err(bad(format!("serve: --max-num-batched-tokens must be between 64 and {}", astral_model::inkling::MAX_PASS_ROWS)));
    }
    let m = astral_loader::hub::resolve(&model, &opts)?;
    let tok_dir = match tokenizer {
        Some(t) => astral_loader::hub::resolve_companion(&t, &opts)?.weights,
        None => m.tokenizer_dir.clone().ok_or_else(|| bad(format!("serve: no tokenizer.json next to {}; pass --tokenizer <dir or org/name>", m.weights.display())))?,
    };
    if !tok_dir.join("tokenizer.json").exists() {
        return Err(bad(format!("serve: {} has no tokenizer.json", tok_dir.display())));
    }
    // The draft lives in the native checkpoint (mtp.safetensors); without it, decode plainly.
    let draft_dir = [m.weights.clone(), tok_dir.clone()].into_iter().find(|d| d.join("mtp.safetensors").exists());
    if draft && draft_dir.is_none() {
        eprintln!("astral: no DSpark draft (mtp.safetensors) found; serving without speculative decoding");
    }
    let draft = draft_dir.filter(|_| draft);
    let engine = astral_api::engine::EngineConfig { gguf: m.weights.clone(), draft, k, ctx, queue, max_seqs, gpu_memory_utilization: util, max_batched_tokens: batched };
    let model_name = name.unwrap_or_else(|| m.name.to_ascii_lowercase());
    astral_api::serve(astral_api::ServerConfig { engine, tokenizer_dir: tok_dir, host, port, model_name })?;
    Ok(true)
}

/// Token ids from the arguments, or from a file of whitespace-separated ids (`@path`).
fn read_ids(ids: &[String]) -> Result<Vec<u32>> {
    let words: Vec<String> = match ids {
        [one] if one.starts_with('@') => std::fs::read_to_string(&one[1..])?.split_whitespace().map(str::to_string).collect(),
        _ => ids.to_vec(),
    };
    let v: Vec<u32> = words.iter().filter_map(|t| t.trim_matches(|c| c == ',' || c == '[' || c == ']').parse().ok()).collect();
    if v.is_empty() {
        return Err(astral_core::AstralError::Invalid("no token ids".into()));
    }
    Ok(v)
}

/// `spec-generate <gguf> <native checkpoint dir> <k> <n> <ids... | @file>`: greedy generation
/// with DSpark speculative decoding (k drafts verified per round), then the same with plain
/// decoding for comparison (skipped with ASTRAL_SPEC_NO_CHECK=1).
fn cmd_spec_generate(path: &str, dir: &str, k: &str, n: &str, ids: &[String]) -> Result<bool> {
    let bad = |m: &str| astral_core::AstralError::Invalid(m.into());
    let k: usize = k.parse().map_err(|_| bad("k must be a number"))?;
    let n: usize = n.parse().map_err(|_| bad("n must be a number"))?;
    let prompt = read_ids(ids)?;
    let cap = std::env::var("ASTRAL_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or((prompt.len() + n + 64).min(8192) as u32);
    let mut s = load_session(path, cap)?;
    let t = std::time::Instant::now();
    s.enable_spec(std::path::Path::new(dir), k)?;
    // Diagnostics: accept against a fixed continuation (e.g. another engine's output).
    if let Ok(f) = std::env::var("ASTRAL_SPEC_FORCE") {
        let cont = read_ids(&[f])?;
        let mut seq = prompt.clone();
        seq.extend(&cont);
        s.spec_force(&seq)?;
        eprintln!("forced continuation: {} tokens", cont.len());
    }
    eprintln!("draft loaded in {:.1} s", t.elapsed().as_secs_f64());
    let reps: usize = std::env::var("ASTRAL_SPEC_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut spec_out = Vec::new();
    for rep in 0..reps {
        s.reset()?;
        let t0 = std::time::Instant::now();
        let first = s.prefill(&prompt)?;
        let tp = t0.elapsed().as_secs_f64();
        s.reset()?;
        let before = s.spec_stats().unwrap_or_default();
        let t1 = std::time::Instant::now();
        spec_out = s.spec_generate(&prompt, n)?;
        let total = t1.elapsed().as_secs_f64();
        let st = s.spec_stats().unwrap_or_default();
        let (rounds, acc) = (st.rounds - before.rounds, st.accepted - before.accepted);
        let decode = (total - tp).max(1e-9);
        println!(
            "run {rep}: prefill {} tok {:.1} ms; decode {} tok in {:.3} s = {:.1} tok/s; {rounds} rounds, {:.2} tokens/round, acceptance {:.1}% of {k} drafts",
            prompt.len(),
            tp * 1e3,
            spec_out.len() - 1,
            decode,
            (spec_out.len() - 1) as f64 / decode,
            (spec_out.len() - 1) as f64 / rounds.max(1) as f64,
            100.0 * acc as f64 / (rounds.max(1) * k as u64) as f64
        );
        let _ = first;
    }
    println!("spec tokens: {spec_out:?}");
    if std::env::var_os("ASTRAL_SPEC_NO_CHECK").is_none() {
        s.reset()?;
        let mut next = s.prefill(&prompt)?;
        let mut plain = vec![next];
        let t1 = std::time::Instant::now();
        for _ in 1..n {
            next = s.step(next)?;
            plain.push(next);
        }
        let td = t1.elapsed().as_secs_f64();
        let same = spec_out.iter().zip(&plain).take_while(|(a, b)| a == b).count();
        println!("plain decode {} tok at {:.1} tok/s; spec output equals plain greedy for the first {same} / {} tokens", n - 1, (n - 1) as f64 / td.max(1e-9), plain.len());
        // Teacher-forced check: plain decode along the spec tokens; at every position the spec
        // token must be the plain argmax, up to near-ties (logit gap below the batched-vs-decode
        // numerical noise).
        s.reset()?;
        let mut pred = s.prefill(&prompt)?;
        let mut logits = s.logits()?;
        let (mut miss, mut worst) = (0usize, 0f32);
        for (i, &t) in spec_out.iter().enumerate() {
            if t != pred {
                miss += 1;
                worst = worst.max(logits[pred as usize] - logits[t as usize]);
            }
            if i + 1 < spec_out.len() {
                pred = s.step(t)?;
                logits = s.logits()?;
            }
        }
        println!("teacher-forced: spec token != plain argmax at {miss} / {} positions, largest logit gap {worst:.3}", spec_out.len());
    }
    Ok(true)
}

fn cmd_kbench_mxfp4(a: &[String], mma: bool) -> Result<bool> {
    let p = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, k, b, iters) = (p(0, 16384), p(1, 8192), p(2, 1), p(3, 50));
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let (name, r) =
        if mma { ("mxfp4 mma", astral_testing::kbench::gemm_mxfp4_mma(&ctx, n, k, b, iters as u32, false)?) } else { ("mxfp4 gemv", astral_testing::kbench::gemv_mxfp4(&ctx, n, k, b, iters as u32)?) };
    println!("{name} n{n} k{k} b{b}: {:.1} us/call, {:.0} GB/s of weights", r.us_per_call, r.weight_gbps);
    Ok(true)
}

fn cmd_kbench_mxfp4_f16(a: &[String]) -> Result<bool> {
    let p = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, k, b, iters) = (p(0, 16384), p(1, 8192), p(2, 1), p(3, 50));
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let r = astral_testing::kbench::gemm_mxfp4_mma(&ctx, n, k, b, iters as u32, true)?;
    println!("mxfp4 mma f16x n{n} k{k} b{b}: {:.1} us/call, {:.0} GB/s of weights", r.us_per_call, r.weight_gbps);
    Ok(true)
}

/// `kbench f16-tma [n] [tokens] [k] [iters]`: the TMA f16 GEMM on random data, checked against
/// a CPU reference on a few tokens, and timed.
fn cmd_kbench_f16_tma(a: &[String]) -> Result<bool> {
    let p = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, t, k, iters, tile) = (p(0, 8576).next_multiple_of(128), p(1, 2048), p(2, 2304), p(3, 50), p(4, 128));
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let reg = astral_kernels::KernelRegistry::load(&ctx)?;
    let gemv = astral_kernels::ops::Gemv::new(&reg)?;
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..n * k).map(|_| rnd() * 0.05).collect();
    let x: Vec<f32> = (0..t * k).map(|_| rnd()).collect();
    let h = |v: &[f32]| -> Vec<u16> { v.iter().map(|&f| astral_quant::f32_to_f16(f)).collect() };
    let (wh, xh) = (h(&w), h(&x));
    let wd = astral_gpu::DeviceBuffer::from_slice(&ctx, &wh)?;
    let xd = astral_gpu::DeviceBuffer::from_slice(&ctx, &xh)?;
    let mut yd = astral_gpu::DeviceBuffer::<f32>::zeroed(&ctx, n * t)?;
    let tm = astral_gpu::TensorMap::f16_2d_sw128(wd.device_ptr(), n as u64, k as u64, 128, 64)?;
    let mut work = astral_kernels::ops::MmaWork::default();
    work.push_chunked(0, &(0..t as u32).map(|r| (r, r)).collect::<Vec<_>>(), tile.max(128));
    let items = astral_gpu::DeviceBuffer::from_slice(&ctx, &work.items)?;
    let xi = astral_gpu::DeviceBuffer::from_slice(&ctx, &work.xidx)?;
    let yi = astral_gpu::DeviceBuffer::from_slice(&ctx, &work.yidx)?;
    let st = astral_gpu::Stream::new(&ctx)?;
    // Tile 0: the two-block-cluster GEMM; 1: the persistent dense one.
    let run = |yd: &mut astral_gpu::DeviceBuffer<f32>| {
        if tile == 1 {
            gemv.launch_gemm_f16_dense(&st, &tm, n as u32, k as u32, xd.device_ptr(), t as u32, yd)
        } else if tile == 0 {
            gemv.launch_gemm_f16_mc(&st, &tm, n as u32, k as u32, xd.device_ptr(), t as u32, yd)
        } else {
            gemv.launch_gemm_f16_tma(&st, tile as u32, &tm, n as u32, k as u32, 1, None, xd.device_ptr(), t as u32, yd, t as u32, &items, work.n_items(), &xi, &yi, t as u32)
        }
    };
    run(&mut yd)?;
    st.synchronize()?;
    let y = yd.to_vec()?;
    let fh = |v: u16| astral_quant::f16_to_f32(v);
    let mut maxerr = 0f32;
    for &tok in &[0usize, t / 2, t - 1] {
        for r in 0..n {
            let refv: f32 = (0..k).map(|i| fh(wh[r * k + i]) * fh(xh[tok * k + i])).sum();
            maxerr = maxerr.max((refv - y[tok * n + r]).abs());
        }
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        run(&mut yd)?;
    }
    st.synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("f16 tma gemm tile {tile} n{n} t{t} k{k}: {us:.1} us, {:.0} TFLOPs, max |err| {maxerr:.2e}", 2.0 * (n * t * k) as f64 / (us * 1e-6) / 1e12);
    Ok(maxerr < 1e-2)
}
