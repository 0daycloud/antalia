# Native-listening CMOS protocol for Voice B promotion

Status: active protocol, v1 (2026-08-08)
Scope: replaces ad-hoc "listen and decide" sessions as the promotion gate required by PLAN.md §16.
Motivation: our automated verifier and Whisper WER are saturated (all shipped packages score sim ≥ 0.86 even on clips native ears reject). Following the evaluation findings in Sesame's CSM work, naturalness must be judged *in context*, pairwise, by native listeners.

## What is being judged

Each trial presents one **pair** of clips for the same prompt. Depending on the study, the pair is:

- **A/B seed pairs** — two takes of the same prompt from different serving candidates (e.g., plain vs envelope-gated selection).
- **Reference-anchored pairs** — one synthesized clip vs one real Voice B recording of comparable content (drawn from the held-out test split, never from training).

## Context rule

Never present clips in isolation. Before each pair, the listener reads the *preceding agent turn* (one or two sentences of dialogue context shown as text, e.g., the customer's question the prompt answers). The rating question is:

> "Hangisi bu konuşmanın devamı olarak daha doğal ve aynı kişinin sesi gibi geliyor?"
> (Which one sounds like a more natural continuation of this conversation, in the same person's voice?)

## Scale

7-point comparative scale per trial: −3 (A clearly better) … 0 (no preference) … +3 (B clearly better).
Clip order within each pair is randomized per trial; the listener never knows which system is which.

## Separate identity check

After each pair, one binary question: "İki kayıt da aynı kişiye mi ait?" (Do both recordings belong to the same person?). This isolates the persona-flip failure mode from general naturalness, giving us a measurable flip rate per system.

## Session design

- 30 pairs per session, ≤ 25 minutes, headphones required.
- Pairs sampled evenly across suite categories; at least 6 long-form and 6 numeric pairs (historically weakest).
- Minimum 3 native listeners per study; report per-listener and pooled scores.
- Include 3 catch trials (identical clip twice) per session; discard listeners who rate catch trials non-zero more than once.

## Decision rule

A serving candidate is promoted only if, against the current production configuration:

1. Pooled CMOS ≥ 0.0 (not worse), and
2. Same-person rate ≥ 95% on reference-anchored pairs, and
3. No category has pooled CMOS ≤ −1.0.

Automated metrics (CER/WER/similarity/envelope) remain regression tripwires, not promotion evidence.

## Assets

- Prompts: `configs/evaluation/turkish-v2.jsonl` (general) and `configs/evaluation/turkish-pronunciation-v1.jsonl` (pronunciation stress set).
- Real reference clips: Voice B held-out test split (`candidate-b-scripted-complete-v1.test.jsonl`).
- Candidate packages: `artifacts/crossflow-candidate-b-final-plain/`, `artifacts/crossflow-candidate-b-final-gated/`.
