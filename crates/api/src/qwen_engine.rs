// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! The engine thread for Qwen3.5 checkpoints: one GPU session, one request at a time from the
//! bounded queue (greedy decoding, no draft).

use crate::engine::{Engine, EngineConfig, Event, Job};
use astral_core::{AstralError, Result};
use astral_model::qwen35::Session;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;

pub(crate) fn start(cfg: EngineConfig, tx: SyncSender<Job>, rx: Receiver<Job>) -> Result<Arc<Engine>> {
    let (ready_tx, ready_rx) = mpsc::channel::<Result<u32>>();
    std::thread::Builder::new()
        .name("astral-engine".into())
        .spawn(move || {
            let mut s = match load(&cfg) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(s.capacity()));
            while let Ok(job) = rx.recv() {
                run(&mut s, job);
            }
        })
        .map_err(|e| AstralError::Invalid(format!("engine thread: {e}")))?;
    let capacity = ready_rx.recv().map_err(|_| AstralError::Invalid("engine thread died while loading".into()))??;
    Ok(Arc::new(Engine { tx, capacity, draft: false, k: 0, max_seqs: 1 }))
}

fn load(cfg: &EngineConfig) -> Result<Session> {
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let t = std::time::Instant::now();
    let w = astral_model::qwen35::load(&ctx, &cfg.gguf)?;
    eprintln!("astral: loaded {:.2} GB of weights in {:.1} s", w.bytes as f64 / 1e9, t.elapsed().as_secs_f64());
    let mut s = Session::new(&ctx, w, cfg.ctx)?;
    // Warm-up: the first prompt pass and decode steps compile kernels and fill caches.
    let warm = [760u32, 6511, 314, 3841];
    let mut t1 = s.prefill(&warm)?;
    for _ in 0..4 {
        t1 = s.step(t1)?;
    }
    s.reset()?;
    Ok(s)
}

fn run(s: &mut Session, job: Job) {
    if job.prompt.is_empty() || job.max_tokens == 0 {
        let _ = job.tx.send(Event::Done { finish: "length" });
        return;
    }
    let go = |s: &mut Session| -> Result<&'static str> {
        s.reset()?;
        let mut tok = s.prefill(&job.prompt)?;
        let mut sent = 0usize;
        loop {
            if job.stop_ids.contains(&tok) {
                return Ok("stop");
            }
            if job.tx.send(Event::Token(tok)).is_err() {
                return Ok("cancelled"); // client gone
            }
            sent += 1;
            if sent >= job.max_tokens || s.pos >= s.capacity() {
                return Ok("length");
            }
            tok = s.step(tok)?;
        }
    };
    match go(s) {
        Ok("cancelled") => {}
        Ok(f) => {
            let _ = job.tx.send(Event::Done { finish: f });
        }
        Err(e) => {
            let _ = job.tx.send(Event::Error(e.to_string()));
        }
    }
}
