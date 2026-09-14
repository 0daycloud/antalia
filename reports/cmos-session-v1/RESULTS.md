# CMOS session v1 — results (2026-08-09)

Listener: project owner (native TR), solo session — below the ≥3-listener protocol bar, treated as
the authoritative first read. Catch trials: 3/3 clean (0 non-zero) — ratings are attentive and valid.

## Scores (toward the candidate package)

| Pool | CMOS | 95% CI | n |
|---|---:|---:|---:|
| Pooled (all non-catch) | **−0.667** | ±0.653 | 27 |
| A/B: gated vs plain | +0.267 | ±0.728 | 15 |
| Anchored: synth vs real | **−1.833** | ±0.757 | 12 |

**Same-person rate on anchored pairs: 0/12 (0%).**

Per category: emotion +2.0, general +2.0, questions +0.2, ack 0.0, long −0.5,
names-places −1.0, voice-agent −1.0, foreign −1.14, numeric −2.0, **normalization −2.5**.

## Verdict

**DO NOT PROMOTE** — all three criteria failed (pooled < 0; same-person 0% vs ≥95% required;
five categories at/below −1.0).

## Interpretation

1. **The automated verifier is saturated, exactly as the protocol suspected.** Machine similarity
   says 0.91–0.94; the person who knows this voice best said "not the same person" on **every**
   synth-vs-real pair. Identity binding via a 256-d additive bias + selection is not sufficient
   for a "this is her voice" claim, however good the verifier numbers look.
2. **The gated serving recipe is still directionally right** (+0.27 over plain, CI overlaps 0) —
   the inference stack isn't the problem; the model's identity/pronunciation ceiling is.
3. **The failing categories are precisely what campaign 2 is recording right now**: numbers
   (numeric −2.0), normalization-style content (−2.5), foreign/tech terms (−1.14), plus
   names-places. Batch design predates this result and targets exactly these gaps.
4. Known caveats, stated not to soften the verdict: single expert listener who knows the study
   design; anchored pairs use different contents (the hard, product-realistic identity test);
   real clips carry room/mic signature that flags them to an informed ear. None of this plausibly
   flips 0/12.

## Decision (updates the launch plan Phase A)

- Champion `crossflow-candidate-consistency-v1/model_6000.pt` stays the engineering baseline;
  **not promoted to launch voice**.
- Path: complete campaign 2 (batches 2–5 in progress) → **Antalia 1.1 consistency retune** on the
  ~2× larger targeted corpus (proven recipe: 40% target / 50% replay / LR 5e-7 / EMA init) →
  fresh CMOS pack (v2, new trials, same protocol, ≥1 additional listener if possible).
- If v2 anchored same-person remains ≪95%: escalate to the reference-audio infilling rebuild
  (structural identity binding) before any launch claim — serving/web work continues in parallel
  either way, since it is model-agnostic.

## Follow-up (same day): timbre diagnosed, gated, ear-validated

The listener localized the anchored failure to voice tone alone ("everything else sounds like the
same person"). Band-spectrum analysis of the pack confirmed it: synth carries +4.0 dB excess at
4.7–6.8 kHz (z = +3.5 vs real-voice variability) plus a formant-region tilt; the vocoder is
transparent (±0.4 dB on real-mel reconstruction), so the acoustic model owns the error.

Response shipped:
- Timbre metric (level-normalized 11-band spectrum shape, z-scored against 13 real clips) —
  separates real (0.47–0.98) from synth (1.42–3.26) with zero overlap on this pack.
- `select-best-of-n.py --timbre-stats/--timbre-weight`; sweep on the frozen 8-seed set picked
  w=0.05: penalty 2.585 → 2.182, 44/120 picks switched, +0.1pp CER. Owner A/B-listened to the
  five biggest switches and confirmed the new picks sound closer to the real voice.
- Promoted as serving recipe v5 (`configs/crossflow/candidate-b-inference-recipe-v5.json`).

Standing target for Antalia 1.1: held-out synthesis timbre penalty inside the real range
[0.47, 0.98] before CMOS v2 is scheduled.
