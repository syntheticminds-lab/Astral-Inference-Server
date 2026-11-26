// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! The engine thread: owns the GPU session (and the DSpark draft) and runs requests with
//! continuous batching. Every running request owns one sequence slot of the session (its KV
//! cache and short-conv state). Between decode steps the engine admits waiting requests
//! (prefilling each into a free slot), then decodes one token for every running request in a
//! single batched pass, and retires requests that hit a stop token, their length limit, or
//! whose client went away (a dropped receiver cancels its request at the next token).
//!
//! Speculative decoding runs when a request is alone: it starts with the DSpark draft if no
//! other request is running or waiting, and falls back to plain batched decoding for the rest
//! of its life as soon as another request arrives.

use astral_core::{AstralError, Result};
use astral_model::inkling::{DecodeRow, Session, Weights};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub gguf: PathBuf,
    /// Native checkpoint directory with `mtp.safetensors` (the DSpark draft); None: no draft.
    pub draft: Option<PathBuf>,
    /// Drafted tokens verified per round.
    pub k: usize,
    /// Context capacity (positions) of each sequence.
    pub ctx: u32,
    /// Requests waiting beyond the running ones.
    pub queue: usize,
    /// Requests decoded together (sequence slots); 0 picks as many as fit in GPU memory, up to 64.
    pub max_seqs: usize,
}

pub enum Event {
    Token(u32),
    /// The request finished: "stop" (a stop token) or "length".
    Done {
        finish: &'static str,
    },
    Error(String),
}

pub struct Job {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub stop_ids: Vec<u32>,
    pub use_draft: bool,
    pub tx: Sender<Event>,
}

pub struct Engine {
    tx: SyncSender<Job>,
    pub capacity: u32,
    pub draft: bool,
    pub k: usize,
    /// Requests decoded together.
    pub max_seqs: usize,
}

/// Most sequence slots picked automatically.
const AUTO_MAX_SEQS: usize = 64;
/// GPU memory left free for scratch buffers, graphs and the allocator when sizing slots.
const RESERVE_BYTES: usize = 6 << 30;

impl Engine {
    /// Loads the model on a new engine thread and returns once it is ready to serve.
    pub fn start(cfg: EngineConfig) -> Result<Arc<Engine>> {
        let (tx, rx) = mpsc::sync_channel::<Job>(cfg.queue.max(1));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(u32, bool, usize)>>();
        let k = cfg.k;
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
                let _ = ready_tx.send(Ok((s.capacity(), s.spec_enabled(), s.n_seqs())));
                Scheduler::new(&mut s, rx).run();
            })
            .map_err(|e| AstralError::Invalid(format!("engine thread: {e}")))?;
        let (capacity, draft, max_seqs) = ready_rx.recv().map_err(|_| AstralError::Invalid("engine thread died while loading".into()))??;
        Ok(Arc::new(Engine { tx, capacity, draft, k, max_seqs }))
    }

    /// Queues a request; `Err` when the queue is full (the caller answers 503).
    pub fn submit(&self, job: Job) -> std::result::Result<(), String> {
        match self.tx.try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("server busy: request queue is full".into()),
            Err(TrySendError::Disconnected(_)) => Err("engine stopped".into()),
        }
    }

    /// Positions a request needs beyond its prompt and output (a speculative round may write
    /// up to k + 1 positions past the last one it keeps).
    pub fn margin(&self) -> usize {
        if self.draft {
            self.k + 2
        } else {
            1
        }
    }
}

fn load(cfg: &EngineConfig) -> Result<Session> {
    let ctx = astral_gpu::Context::new(astral_gpu::Device::get(0)?)?;
    let g = astral_loader::open_model(&cfg.gguf)?;
    let t = std::time::Instant::now();
    let w = Weights::load(&ctx, &g)?;
    eprintln!("astral: loaded {:.2} GB of weights in {:.1} s", w.bytes as f64 / 1e9, t.elapsed().as_secs_f64());
    let per_seq = astral_model::inkling::seq_bytes(&w.cfg, cfg.ctx);
    let (free, _) = ctx.mem_info()?;
    // The draft's weights and buffers come out of the same memory.
    let reserve = RESERVE_BYTES + if cfg.draft.is_some() { 2 << 30 } else { 0 };
    let fit = free.saturating_sub(reserve) / per_seq.max(1);
    let seqs = if cfg.max_seqs == 0 { fit.clamp(1, AUTO_MAX_SEQS) } else { cfg.max_seqs };
    if seqs > fit.max(1) {
        eprintln!("astral: warning: {seqs} sequences of {} positions need {:.1} GB; {:.1} GB are free", cfg.ctx, (seqs * per_seq) as f64 / 1e9, free as f64 / 1e9);
    }
    let t = std::time::Instant::now();
    let mut s = Session::with_seqs(&ctx, w, cfg.ctx, seqs, seqs)?;
    if let Some(dir) = &cfg.draft {
        s.enable_spec(dir, cfg.k)?;
    }
    // Warm-up: capture the decode / speculative graphs before the first request.
    let warm = [200028u32, 760, 6511, 314];
    s.prefill(&warm)?;
    let mut t1 = 0;
    for _ in 0..4 {
        t1 = s.step(t1)?;
    }
    s.reset()?;
    if s.spec_enabled() {
        s.spec_begin(&warm)?;
        for _ in 0..4 {
            s.spec_next()?;
        }
        s.reset()?;
    }
    // Batched decode: one graph per batch size, exercised once on every slot.
    for n in 1..=s.max_batch() {
        s.capture_batch(n)?;
    }
    for q in 0..seqs {
        s.reset_seq(q)?;
        s.prefill_seq(q, 0, &warm)?;
    }
    let rows: Vec<DecodeRow> = (0..seqs).map(|q| DecodeRow { seq: q, pos: warm.len() as u32, token: warm[0] }).collect();
    s.decode_batch(&rows)?;
    s.decode_batch(&rows[..1])?;
    for q in 0..seqs {
        s.reset_seq(q)?;
    }
    eprintln!("astral: {seqs} sequence slots of {} positions ({:.1} GB), batch graphs ready in {:.1} s", cfg.ctx, (seqs * per_seq) as f64 / 1e9, t.elapsed().as_secs_f64());
    Ok(s)
}

/// A running request.
struct Active {
    job: Job,
    /// Its sequence slot.
    seq: usize,
    /// Positions written so far (the next input goes to this position).
    pos: u32,
    /// The last generated token: the next input.
    last: u32,
    /// Tokens sent to the client.
    sent: usize,
    /// Decoding with speculative rounds (sequence 0, alone).
    spec: bool,
}

/// Why a request ended, if it did.
enum Emit {
    More,
    Finished(&'static str),
    Cancelled,
}

impl Active {
    /// Sends one generated token; reports whether the request is over.
    fn emit(&mut self, t: u32) -> Emit {
        if self.job.stop_ids.contains(&t) {
            return Emit::Finished("stop");
        }
        if self.job.tx.send(Event::Token(t)).is_err() {
            return Emit::Cancelled;
        }
        self.sent += 1;
        self.last = t;
        if self.sent >= self.job.max_tokens {
            return Emit::Finished("length");
        }
        Emit::More
    }
}

/// A request whose prompt is still being processed, a chunk per step.
struct Pending {
    job: Job,
    seq: usize,
    /// Prompt tokens already in the sequence.
    done: usize,
}

struct Scheduler<'a> {
    s: &'a mut Session,
    rx: Receiver<Job>,
    waiting: VecDeque<Job>,
    /// Admitted requests still prefilling, oldest first.
    prefilling: VecDeque<Pending>,
    active: Vec<Active>,
    free: Vec<bool>,
    closed: bool,
    /// `ASTRAL_STEP_STATS=1`: time and tokens of mixed passes and decode-only steps.
    stats: Option<StepStats>,
}

#[derive(Default)]
struct StepStats {
    mixed: (u64, f64, usize, usize),
    decode: (u64, f64, usize),
    last: Option<std::time::Instant>,
}

impl StepStats {
    fn report(&mut self) {
        if self.last.is_some_and(|t| t.elapsed().as_secs_f64() < 5.0) {
            return;
        }
        self.last = Some(std::time::Instant::now());
        let (mn, mt, mp, md) = self.mixed;
        let (dn, dt, dr) = self.decode;
        eprintln!(
            "astral: steps: mixed {mn} ({mt:.1} s, {mp} prompt + {md} decode tokens, {:.1} ms avg), decode-only {dn} ({dt:.1} s, {dr} tokens, {:.1} ms avg)",
            mt * 1e3 / mn.max(1) as f64,
            dt * 1e3 / dn.max(1) as f64
        );
    }
}

impl<'a> Scheduler<'a> {
    fn new(s: &'a mut Session, rx: Receiver<Job>) -> Self {
        let n = s.n_seqs();
        Self { s, rx, waiting: VecDeque::new(), prefilling: VecDeque::new(), active: Vec::new(), free: vec![true; n], closed: false, stats: std::env::var_os("ASTRAL_STEP_STATS").map(|_| StepStats::default()) }
    }

    fn run(&mut self) {
        loop {
            // Block for work when idle; otherwise take whatever has arrived.
            if self.active.is_empty() && self.waiting.is_empty() && self.prefilling.is_empty() {
                match self.rx.recv() {
                    Ok(j) => self.waiting.push_back(j),
                    Err(_) => return,
                }
            }
            self.drain();
            if self.closed && self.active.is_empty() && self.waiting.is_empty() && self.prefilling.is_empty() {
                return;
            }
            self.admit();
            if let Err(e) = self.step() {
                // A failed step poisons every running request; the session is still usable.
                let msg = e.to_string();
                for a in self.active.drain(..) {
                    let _ = a.job.tx.send(Event::Error(msg.clone()));
                    self.free[a.seq] = true;
                }
                for p in self.prefilling.drain(..) {
                    let _ = p.job.tx.send(Event::Error(msg.clone()));
                    self.free[p.seq] = true;
                }
            }
        }
    }

    fn drain(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(j) => self.waiting.push_back(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
        }
    }

    /// Gives waiting requests free sequence slots (lowest slot first, so a request that starts
    /// alone gets slot 0, the one speculative decoding runs on). Their prompts are then processed
    /// a chunk per step, alongside the requests that are decoding.
    fn admit(&mut self) {
        while let Some(job) = self.waiting.pop_front() {
            if job.prompt.is_empty() || job.max_tokens == 0 {
                let _ = job.tx.send(Event::Done { finish: "length" });
                continue;
            }
            let Some(seq) = self.free.iter().position(|&f| f) else {
                self.waiting.push_front(job);
                break;
            };
            let alone = self.active.is_empty() && self.waiting.is_empty() && self.prefilling.is_empty();
            if alone && seq == 0 && job.use_draft && self.s.spec_enabled() {
                match self.s.reset().and_then(|_| self.s.spec_begin(&job.prompt)) {
                    Ok(first) => self.start(job, seq, first, true),
                    Err(e) => {
                        let _ = job.tx.send(Event::Error(e.to_string()));
                    }
                }
                continue;
            }
            if let Err(e) = self.s.reset_seq(seq) {
                let _ = job.tx.send(Event::Error(e.to_string()));
                continue;
            }
            self.free[seq] = false;
            self.prefilling.push_back(Pending { job, seq, done: 0 });
        }
    }

    /// A request whose prompt is in slot `seq` and whose first token is `first`.
    fn start(&mut self, job: Job, seq: usize, first: u32, spec: bool) {
        self.free[seq] = false;
        let mut a = Active { pos: job.prompt.len() as u32, job, seq, last: first, sent: 0, spec };
        match a.emit(first) {
            Emit::More => self.active.push(a),
            end => self.retire(a, end),
        }
    }

    fn retire(&mut self, a: Active, end: Emit) {
        if let Emit::Finished(f) = end {
            let _ = a.job.tx.send(Event::Done { finish: f });
        }
        self.free[a.seq] = true;
    }

    /// One decode step for every running request.
    fn step(&mut self) -> Result<()> {
        if self.active.is_empty() && self.prefilling.is_empty() {
            return Ok(());
        }
        // A lone speculative request runs one round; anything else decodes in a batch.
        if self.active.len() == 1 && self.active[0].spec && self.prefilling.is_empty() {
            let toks = self.s.spec_next()?;
            let mut a = self.active.pop().expect("one active request");
            a.pos = self.s.pos;
            if toks.is_empty() {
                self.retire(a, Emit::Finished("length"));
                return Ok(());
            }
            let mut end = Emit::More;
            for t in toks {
                end = a.emit(t);
                if !matches!(end, Emit::More) {
                    break;
                }
            }
            match end {
                Emit::More => self.active.push(a),
                e => self.retire(a, e),
            }
            return Ok(());
        }
        for a in &mut self.active {
            a.spec = false;
        }
        let cap = self.s.capacity();
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].pos >= cap {
                let a = self.active.swap_remove(i);
                self.retire(a, Emit::Finished("length"));
            } else {
                i += 1;
            }
        }
        let mb = self.s.max_batch();
        if !self.prefilling.is_empty() && self.active.len() + self.prefilling.len() <= mb {
            return self.mixed_step();
        }
        let mut start = 0;
        while start < self.active.len() {
            let end = (start + mb).min(self.active.len());
            let rows: Vec<DecodeRow> = self.active[start..end].iter().map(|a| DecodeRow { seq: a.seq, pos: a.pos, token: a.last }).collect();
            let t0 = std::time::Instant::now();
            let next = self.s.decode_batch(&rows)?;
            if let Some(st) = self.stats.as_mut() {
                st.decode.0 += 1;
                st.decode.1 += t0.elapsed().as_secs_f64();
                st.decode.2 += rows.len();
                st.report();
            }
            for (a, t) in self.active[start..end].iter_mut().zip(next) {
                a.pos += 1;
                a.last = t;
            }
            start = end;
        }
        // Emit after the pass (a retire reorders `active`).
        let mut i = 0;
        while i < self.active.len() {
            let t = self.active[i].last;
            match self.active[i].emit(t) {
                Emit::More => i += 1,
                end => {
                    let a = self.active.swap_remove(i);
                    self.retire(a, end);
                }
            }
        }
        Ok(())
    }

    /// One pass holding the next token of every decoding request and, in the rest of the token
    /// budget, the next prompt chunks of the prefilling requests (oldest first). Decoding never
    /// waits for a whole prompt, and the decode tokens share the pass's weight reads.
    fn mixed_step(&mut self) -> Result<()> {
        let budget = self.s.token_budget();
        let mut parts: Vec<(usize, u32, Vec<u32>)> = self.active.iter().map(|a| (a.seq, a.pos, vec![a.last])).collect();
        let mut left = budget.saturating_sub(parts.len());
        let mut chunks = Vec::new();
        for (i, p) in self.prefilling.iter().enumerate() {
            if left == 0 {
                break;
            }
            let n = (p.job.prompt.len() - p.done).min(left);
            parts.push((p.seq, p.done as u32, p.job.prompt[p.done..p.done + n].to_vec()));
            chunks.push((i, n));
            left -= n;
        }
        let views: Vec<(usize, u32, &[u32])> = parts.iter().map(|(q, pos, t)| (*q, *pos, t.as_slice())).collect();
        let t0 = std::time::Instant::now();
        let next = self.s.step_mixed(&views)?;
        if let Some(st) = self.stats.as_mut() {
            let p: usize = chunks.iter().map(|c| c.1).sum();
            st.mixed.0 += 1;
            st.mixed.1 += t0.elapsed().as_secs_f64();
            st.mixed.2 += p;
            st.mixed.3 += self.active.len();
            st.report();
        }
        let n_dec = self.active.len();
        for (a, &t) in self.active.iter_mut().zip(&next[..n_dec]) {
            a.pos += 1;
            a.last = t;
        }
        // Prompts that finished this pass start decoding with the token it produced.
        let mut finished = Vec::new();
        for (&(i, n), &t) in chunks.iter().zip(&next[n_dec..]) {
            let p = &mut self.prefilling[i];
            p.done += n;
            if p.done == p.job.prompt.len() {
                finished.push((i, t));
            }
        }
        let mut i = 0;
        while i < n_dec.min(self.active.len()) {
            let t = self.active[i].last;
            match self.active[i].emit(t) {
                Emit::More => i += 1,
                end => {
                    let a = self.active.swap_remove(i);
                    self.retire(a, end);
                }
            }
        }
        for &(i, t) in finished.iter().rev() {
            let p = self.prefilling.remove(i).expect("finished prompt is pending");
            self.start(p.job, p.seq, t, false);
        }
        Ok(())
    }
}
