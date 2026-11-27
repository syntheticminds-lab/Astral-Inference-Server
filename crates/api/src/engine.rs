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
    // (Batched drafting adds per-sequence draft caches and verification row states: ~5 GB.)
    let batch_spec = std::env::var("ASTRAL_BATCH_SPEC").map_or(true, |v| v != "0");
    let reserve = RESERVE_BYTES + if cfg.draft.is_some() { if batch_spec { 7 << 30 } else { 2 << 30 } } else { 0 };
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
        // Batched drafting for requests that start while few are running (ASTRAL_BATCH_SPEC=0:
        // lone requests only). At 8 concurrent requests serving measured ~10% faster.
        if std::env::var("ASTRAL_BATCH_SPEC").map_or(true, |v| v != "0") {
            s.enable_batch_spec(seqs)?;
            s.reset_seq(0)?;
            let first = s.prefill_seq_spec(0, 0, &warm)?;
            let d = s.spec_first_draft(0, warm.len() as u32, first)?;
            s.spec_round(&[(0, warm.len() as u32, first, d)])?;
            s.reset_seq(0)?;
        }
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
    /// Batched speculative decoding: the drafts for the next round (the draft context is
    /// current only while set, so a request never leaves and rejoins).
    drafts: Option<Vec<u32>>,
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
    /// When the request was admitted.
    since: std::time::Instant,
    /// Batched drafting: the prompt's passes store the draft context and the request starts
    /// with drafts.
    spec: bool,
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
    /// Set while the requests without drafts step beside a draft round: no prompt pass then.
    hold_prefill: bool,
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
        Self { s, rx, waiting: VecDeque::new(), prefilling: VecDeque::new(), active: Vec::new(), free: vec![true; n], closed: false, stats: std::env::var_os("ASTRAL_STEP_STATS").map(|_| StepStats::default()), hold_prefill: false }
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
            // Batched drafting while few requests run: the prompt now (with the draft context),
            // then speculative rounds.
            let spec_max: usize = std::env::var("ASTRAL_SPEC_MAX_ACTIVE").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            let load = self.active.len() + self.prefilling.len() + self.waiting.len() + 1;
            let alone = load == 1;
            let spec = job.use_draft && self.s.batch_spec_k() > 0 && !alone && load <= spec_max;
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
            self.prefilling.push_back(Pending { job, seq, done: 0, since: std::time::Instant::now(), spec });
        }
    }

    /// A request whose prompt is in slot `seq` and whose first token is `first`.
    fn start(&mut self, job: Job, seq: usize, first: u32, spec: bool) {
        self.free[seq] = false;
        let mut a = Active { pos: job.prompt.len() as u32, job, seq, last: first, sent: 0, spec, drafts: None };
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

    /// One decode step for every running request: a batched speculative round for those
    /// drafting, a decode (or mixed) pass for the rest.
    fn step(&mut self) -> Result<()> {
        // Too many requests for drafting to pay: everyone decodes in one batch from now on.
        let spec_max: usize = std::env::var("ASTRAL_SPEC_MAX_ACTIVE").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        if self.active.len() + self.prefilling.len() + self.waiting.len() > spec_max {
            for a in &mut self.active {
                a.drafts = None;
            }
        }
        // While prompts are pending, drafting requests ride in the packed pass as one-token
        // rows (their context is stored and they draft again after it).
        let mixing = !self.prefilling.is_empty() && self.active.len() + self.prefilling.len() <= self.s.max_batch() && self.prefill_now();
        if self.active.iter().any(|a| a.drafts.is_some()) && !mixing {
            let (spec, rest): (Vec<Active>, Vec<Active>) = std::mem::take(&mut self.active).into_iter().partition(|a| a.drafts.is_some());
            self.active = rest;
            let kept = self.spec_step(spec);
            // The prompts keep waiting (the decision was made with the drafting requests counted).
            let r = if self.active.is_empty() { Ok(()) } else { self.hold_prefill = true; let r = self.plain_step(); self.hold_prefill = false; r };
            match kept {
                Ok(k) => self.active.extend(k),
                Err(e) => return Err(e),
            }
            return r;
        }
        self.plain_step()
    }

    /// One batched speculative round for `spec` (requests with drafts); returns those still running.
    fn spec_step(&mut self, spec: Vec<Active>) -> Result<Vec<Active>> {
        let cap = self.s.capacity() as usize;
        let k = self.s.batch_spec_k();
        let (mut live, mut done) = (Vec::new(), Vec::new());
        for a in spec {
            if a.pos as usize + k + 2 > cap {
                done.push(a);
            } else {
                live.push(a);
            }
        }
        for a in done {
            self.retire(a, Emit::Finished("length"));
        }
        let mb = self.s.max_batch();
        let mut kept = Vec::with_capacity(live.len());
        for chunk in live.chunks_mut(mb) {
            let entries: Vec<(usize, u32, u32, Vec<u32>)> = chunk.iter().map(|a| (a.seq, a.pos, a.last, a.drafts.clone().unwrap_or_default())).collect();
            let t0 = std::time::Instant::now();
            let res = self.s.spec_round(&entries)?;
            if std::env::var_os("ASTRAL_SPEC_STATS").is_some() {
                static C: std::sync::Mutex<(u64, u64, u64, f64, Option<std::time::Instant>)> = std::sync::Mutex::new((0, 0, 0, 0.0, None));
                let mut c = C.lock().unwrap_or_else(|e| e.into_inner());
                c.0 += 1;
                c.1 += entries.len() as u64;
                c.2 += res.iter().map(|r| r.0.len() as u64).sum::<u64>();
                c.3 += t0.elapsed().as_secs_f64();
                let since = *c.4.get_or_insert(t0);
                if c.0 % 100 == 0 {
                    eprintln!(
                        "astral: spec rounds {}: {:.2} requests per round, {:.2} tokens per request, {:.2} ms per round, {:.1} s in rounds of {:.1} s",
                        c.0,
                        c.1 as f64 / c.0 as f64,
                        c.2 as f64 / c.1 as f64,
                        c.3 * 1e3 / c.0 as f64,
                        c.3,
                        since.elapsed().as_secs_f64()
                    );
                }
            }
            if let Some(st) = self.stats.as_mut() {
                st.decode.0 += 1;
                st.decode.1 += t0.elapsed().as_secs_f64();
                st.decode.2 += res.iter().map(|r| r.0.len()).sum::<usize>();
                st.report();
            }
            for (a, (acc, d)) in chunk.iter_mut().zip(res) {
                a.pos += acc.len() as u32;
                a.drafts = Some(d);
                let mut end = Emit::More;
                for &t in &acc {
                    end = a.emit(t);
                    if !matches!(end, Emit::More) {
                        break;
                    }
                }
                if let Some(&t) = acc.last() {
                    a.last = t;
                }
                if !matches!(end, Emit::More) {
                    a.drafts = None;
                    a.sent = usize::MAX;
                    if let Emit::Finished(f) = end {
                        let _ = a.job.tx.send(Event::Done { finish: f });
                    }
                }
            }
        }
        for a in live {
            if a.sent == usize::MAX {
                self.free[a.seq] = true;
            } else {
                kept.push(a);
            }
        }
        Ok(kept)
    }

    /// One decode (or mixed) step for the requests without drafts.
    fn plain_step(&mut self) -> Result<()> {
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
        if !self.prefilling.is_empty() && !self.hold_prefill && self.active.len() + self.prefilling.len() <= mb && self.prefill_now() {
            return self.mixed_step();
        }
        // A plain decode step does not store the draft context: drafting stops for these.
        for a in &mut self.active {
            a.drafts = None;
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

    /// Whether to run prompt chunks this step. A pass streams every expert's weights however few
    /// prompt tokens it holds, so while requests are decoding, prompts that arrive one by one
    /// wait (at most `ASTRAL_PREFILL_WAIT_MS`, default 0 = never) until `ASTRAL_PREFILL_BATCH`
    /// prompt tokens are pending, and are then prefilled together.
    fn prefill_now(&self) -> bool {
        // While requests are drafting, prompts wait (300 ms, or two prompts' worth of tokens)
        // to share a pass: a pass streams the experts however few tokens it holds.
        // (Only with several requests running: with one or two, the wait just idles the GPU.)
        let drafting = self.active.len() >= 4 && self.active.iter().any(|a| a.drafts.is_some());
        let wait_ms: u64 = std::env::var("ASTRAL_PREFILL_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(if drafting { 300 } else { 0 });
        if wait_ms == 0 || self.active.is_empty() {
            return true;
        }
        let want: usize = std::env::var("ASTRAL_PREFILL_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(if drafting { 1024 } else { self.s.token_budget() / 2 });
        let pending: usize = self.prefilling.iter().map(|p| p.job.prompt.len() - p.done).sum();
        let oldest = self.prefilling.front().map_or(0, |p| p.since.elapsed().as_millis() as u64);
        pending >= want || oldest >= wait_ms || self.active.len() + self.prefilling.len() >= self.s.max_batch()
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
        // Drafting requests' prompt chunks also store the draft context; the ones finishing
        // here get their first drafts.
        let n_dec0 = self.active.len();
        let mut spec_f = vec![false; views.len()];
        let mut first_f = vec![false; views.len()];
        for (i, a) in self.active.iter().enumerate() {
            spec_f[i] = a.drafts.is_some();
            first_f[i] = a.drafts.is_some();
        }
        for (c, &(i, n)) in chunks.iter().enumerate() {
            let p = &self.prefilling[i];
            spec_f[n_dec0 + c] = p.spec;
            first_f[n_dec0 + c] = p.spec && p.done + n == p.job.prompt.len();
        }
        let t0 = std::time::Instant::now();
        let (next, mut first_drafts) = self.s.step_mixed_spec(&views, &spec_f, &first_f)?;
        if let Some(st) = self.stats.as_mut() {
            let p: usize = chunks.iter().map(|c| c.1).sum();
            st.mixed.0 += 1;
            st.mixed.1 += t0.elapsed().as_secs_f64();
            st.mixed.2 += p;
            st.mixed.3 += self.active.len();
            st.report();
        }
        let n_dec = self.active.len();
        for (i, (a, &t)) in self.active.iter_mut().zip(&next[..n_dec]).enumerate() {
            a.pos += 1;
            a.last = t;
            if a.drafts.is_some() {
                a.drafts = first_drafts[i].take();
            }
        }
        // Prompts that finished this pass start decoding with the token it produced.
        let mut finished = Vec::new();
        for (c, (&(i, n), &t)) in chunks.iter().zip(&next[n_dec..]).enumerate() {
            let p = &mut self.prefilling[i];
            p.done += n;
            if p.done == p.job.prompt.len() {
                finished.push((i, t, c));
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
        for &(i, t, c) in finished.iter().rev() {
            let p = self.prefilling.remove(i).expect("finished prompt is pending");
            let seq = p.seq;
            self.start(p.job, seq, t, false);
            if let Some(d) = first_drafts[n_dec0 + c].take()
                && let Some(a) = self.active.iter_mut().find(|a| a.seq == seq)
            {
                a.drafts = Some(d);
            }
        }
        Ok(())
    }
}
