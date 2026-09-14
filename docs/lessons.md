# Antalia 1 (TR): what actually moved the needle

Audience: the Alania (EN) team. Every item below is something we did in the Antalia campaign that
produced an **immediate, measured** improvement — with the before/after and the mechanism. Ordered
roughly by impact-per-effort. Numbers are Whisper-large-v3-scored CER/WER and speaker-verifier
cosine similarity on our 120-prompt Turkish suite unless noted.

---

## 1. Clause chunking + a per-character rate floor (long-form fix)
**Boost: long-form CER 23.9% → 4.1%, WER 42.9% → ~11%, overnight.**

Diagnosis first, fix second: long prompts sounded like rushed mumbling. We measured speaking rate
and found the frozen duration predictor budgeting **0.0635 s/char on long text vs 0.083 s/char in
the speaker's natural speech** (~35% too fast) — it was trained on short Common Voice clips and
under-budgeted frames for anything longer, so the flow crammed syllables and smeared them.

Fix (inference-only, two parts):
- **Chunk text at sentence/comma boundaries into ≤120-char pieces**, synthesize each independently
  (in-distribution length for the model), join with a fixed 160 ms pause.
- **Rate floor:** a chunk's frame budget can't imply speech faster than 0.085 s/char, whatever the
  duration head predicts.

Chunking alone: long-form CER 23.1% → 4.1%. Lesson: when duration is a learned scalar, it fails
out-of-distribution *quietly*. Alania: keep the chunker; give the duration head length-diverse
training data from day one.

## 2. Low-LR consistency fine-tune with replay (the identity fix)
**Boost: worst-tenth speaker similarity p10 0.860 → 0.891 pre-selection; flip-rate visibly down; CER unchanged.**

The champion's failure mode was identity flips: the target voice entered a 1,500-speaker foundation
as a small additive bias, and trajectories sometimes settled in a neighboring speaker's basin. The
single most effective fix was a **consistency fine-tune**:

- Upweight the target speaker from ~1.5% to **~40% of batches**; the rest is **foundation replay**
  (keeps general Turkish from eroding).
- **Very low peak LR (~5e-7)**, init from the foundation's **EMA weights**, short run (12k updates).
- Checkpoint every few hundred updates; evaluate several, don't trust the last one.

Two failed variants first taught us the guardrails: an aggressive direct adaptation regressed
intelligibility by update 300–400, and only the 50%-replay + tiny-LR recipe improved identity
*without* moving CER. Alania: same recipe per arena voice on top of the multi-speaker foundation.

## 3. Automatic abort on validation degradation (training that defends itself)
**Boost: prevented shipping two regressed branches; zero human vigilance needed.**

Every fine-tune runs with `validation_degradation_factor`: if val loss exceeds
best × factor for `patience` consecutive evals, the run **aborts itself** and the parent checkpoint
remains immutable on disk. One adaptation run tripped it at update 450 — training stopped, the
foundation stayed untouched, and the eval later confirmed the abort was right. Combined with the
**zero-init/gated-conditioning rule** (below), branching experiments became cheap and safe.

## 4. Zero-initialized, gated new conditioning paths (bit-exactness discipline)
**Boost: every architecture experiment started from a provably unbroken model.**

Any new conditioning channel (context encoder, speaker adapter, prosody features) is added
**zero-initialized and gated**, so the checkpoint's output is **bit-exact** with the parent until
the new path actually trains. We verified bit-exactness explicitly (`torch.allclose` at 1e-5 on
matched sampling) before every branch. This is why we could try context-conditioning, infill,
prefix-geometry, and speaker-consistency branches in rapid succession without ever wondering
whether a regression came from the plumbing or the training. Alania: make this a hard PR rule.

## 5. Batched best-of-N sampling + verifier/gate selection (quality for free at inference)
**Boost: CER 8.18% → 5.42%, WER 17.0% → 12.7%, sim p10 0.8603 → 0.909 (best-of-4); best-of-8 final gated package: CER 3.65%, sim p10 0.9110.**

- **One batched flow pass renders N seeds** (padded mixed lengths, per-row noise, shared
  conditioning): best-of-8 costs **~1.5× a single generation, not 8×** (measured: 8 candidates in
  145 s vs 95 s for one across a 120-prompt suite).
- Selection = speaker-verifier similarity + Whisper CER, plus a small **F0-envelope penalty**
  (z-score of log-F0 mean/spread/voiced-ratio against the *real* recordings' distribution, weight
  0.05) — this catches "right speaker, wrong persona" takes the verifier is blind to; it raised
  pre-selection p10 0.860 → 0.891 → 0.912 after gating.
- Deterministic seeds → published metrics stay reproducible.

Lesson: a saturated verifier ranks; it doesn't discriminate persona. Add an acoustic-statistics
gate derived from real target audio. Alania: this is also the candidate generator for preference
training — same machinery, second job.

## 6. CFG polish: guidance rescale + mel clamp + sway
**Boost: (sweep, single-seed) CER 0.0934 → 0.0867 and SNR +4 dB from rescale 0.5 + clamp 5.0; sway −0.8 default from the same sweeps.**

Three cheap scalar knobs, each validated by sweep against the eval suite, all shipped in the
recipe: **text guidance 4.0** (pushes mel off the trained manifold at the peaks →) **mel clamp
±5σ** (drops clip-ratio ~3×) and **guidance rescale 0.5** (std-matching the guided output back to
the conditioned branch's distribution; softened output slightly, so it's on by default only in the
serving recipe where the sweep supported it). None of these need retraining. Alania: re-sweep per
model, but wire the flags in from the start.

## 7. Fix the ruler before the model (evaluator normalization)
**Boost: numbers-category "41.5% CER" collapsed to ~2.7% CER / 7.4% WER — the model was fine.**

Whisper transcribes "üç bin yedi yüz" as "3700"; our targets had spelled-out words. The category
looked catastrophic until we added `normalize_for_scoring` (spell digits, decimals,
thousands-groups the way the language speaks them) **in the evaluator only** — training text
untouched. Re-measured: the planned "numbers fine-tune" was cancelled; no training run was needed
at all. Alania: build ASR-comparison normalization for English (numerals, currency, ordinals,
abbreviations) into the eval harness *before* reading any category table.

## 8. Pin the mel/vocoder contract and prove it with reconstruction probes
**Boost: zero debugging hours lost to mel mismatch — because it couldn't happen.**

The BigVGAN-v2 contract (24 kHz, 100 bands, hop 256, 0–12 kHz, revision-pinned weights + sha256)
was frozen in config before the first training step, and every training-progress eval included a
**ground-truth vocoder reconstruction probe** (vocode the real mel of a training clip) as the
quality ceiling. When early FastPitch audio was garbage, the clean reconstruction probe (~24 dB
SNR, intelligible) immediately localized the problem to the acoustic model, not the vocoder or
featurization. Alania: same vocoder, same contract, same probes.

## 9. Prosody presets from real segments + auto-style (style without training)
**Boost: preset styles at CER cost ≈ 0 (4.88% vs 4.89% baseline, identical similarity); question/empathy/emotion contrasts confirmed audible in demos.**

Per-category **prosody centroids** (log-F0 mean/spread, energy, rate, voiced-ratio, 6-dim)
computed from the target speaker's *real* segments grouped by campaign category → served as
`--preset NAME --preset-strength`, with a keyword **auto-style** router at the chunk level
(question clauses get question prosody, apology keywords get restrained_emotion). Zero training;
guardrail sweep showed full-strength presets cost ≤1pp CER except one (capped at 0.7). Alania: the
arena's 4 use-case categories map directly onto this — derive presets per voice from the actors'
own expressive takes.

## 9b. Timbre profile gate — a human-calibrated identity ruler (added after CMOS v1)
**Boost: mean timbre distance −16% at +0.1pp CER, 44/120 picks improved, ear-confirmed same day; and the first identity metric that agrees with native listeners instead of the saturated verifier.**

CMOS v1 failed promotion with same-person 0/12 while the speaker verifier read 0.91+ — the listener
said timbre alone gave it away. A trivial metric captured it: **level-normalized 11-band long-term
spectrum shape, z-scored against real recordings** (`scripts/build-timbre-profile.py`). It separates
real clips (0.47–0.98) from synth (1.42–3.26) with zero overlap, i.e., it reproduces the human
identity verdict from audio alone. Added as a third best-of-N selection gate (w=0.05 beside the F0
envelope), and adopted as the **train-time target for fine-tunes**: held-out synthesis must enter
the real range before scheduling human evaluation. Two lessons for Alania: (a) verifier embeddings
are prosody-heavy and timbre-blind — never let them be the only identity signal; (b) when a human
names a failure dimension, build the cheap physical metric for exactly that dimension the same day.

## 10. Duration-bucketed batching + the boring infra that held
Not a single-number boost, but the reasons runs were cheap and never lost:
- **Duration-bucketed batch sampler** (batch by audio length): ~stable step times, no OOM roulette.
- **bf16 + activation checkpointing**: 300M model trained comfortably inside one 80 GB A100
  (~45–50 GB peak) — multi-GPU never became a dependency.
- **Rank-0 checkpointing, all-reduced validation, resume-if-exists**: every interruption (and we
  had several, including a broker death mid-session) resumed without loss.
- **EMA weights maintained throughout; serve and fine-tune from EMA**, not raw.
- Checkpoints + recipes recorded with **sha256 in every synthesis/quality report** — every audio
  artifact in the repo is traceable to exact weights, vocoder revision, and flag set.

## Anti-lessons (things that did NOT pay off, so Alania doesn't repeat them)
- **Bolt-on context/infill conditioning on a trained single-speaker checkpoint**: two attempts
  (context-conditioning v1, prefix-geometry v2) never beat the consistency-tune + selection stack
  on identity, and one aborted on validation degradation. Verdict then: infill must be trained
  **from step 0**, not retrofitted — which is exactly the Alania P1 design.
- **Chasing "very low WER" early**: with `learn_alignment`, FastPitch produced hallucination-grade
  output for the first ~40 epochs by design (alignment warmup); panicking before the binarization
  loss engages wastes effort. Know each architecture's "garbage-is-expected" window.
- **Judging categories before fixing scoring normalization** (see #7) — nearly bought a pointless
  fine-tune.
- **Single-seed judgment of identity**: per-seed similarity variance is large (flip takes); any
  identity metric read off one seed per prompt is noise. Evaluate with the selection stack you'll
  actually serve.
