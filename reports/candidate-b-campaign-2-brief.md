# Voice B recording campaign 2 — design brief

Status: proposal (2026-08-09)
Contributor: Voice B (existing consent covers commercial voice synthesis; reconfirm scope for expressive/paralinguistic material before session 1).
Current inventory: 621 scripted segments, 2.96 h, six categories. Target: **+4–6 h** across four batches, priority order below.

## Why these batches

Every measured or heard weakness of the current system maps to a coverage gap in the first campaign:
persona flips (too little identity data), flat agent prosody (no in-context dialogue), chunking crutch
(no long-form with natural pauses), stiff readbacks (no digit-by-digit confirmations).

## Batch 1 — In-context dialogue turns (~1.5 h, highest priority)

The prompter displays a customer line; Voice B *answers it*, never reads it. One session = one
coherent scenario (billing dispute, appointment change, delivery delay, onboarding) of 8–15 turns.

- Covers: greeting, clarification, confirmation, empathy, transition, closing — as reactions, not read sentences.
- Why: this is the data Sesame-style contextual prosody needs, and it directly feeds the CMOS
  "natural continuation" criterion. Our prosody presets are currently derived from read speech;
  reactive speech will sharpen them.
- Capture the shown customer line in the metadata for every turn (future conversation-context training).

## Batch 2 — Expressive range with graded intensity (~1 h)

Same short scripts rendered at labeled intensities (sakin / sıcak / coşkulu; üzgün-özür 1–3), plus a
controlled paralinguistic inventory: onaylama sesleri ("hı-hı", "tabii", "elbette"), düşünme dolgusu
("şöyle söyleyeyim…", kısa "ee"), nefes ve gülümseyerek konuşma. No free laughter takes unless the
contributor is comfortable; label every take.

- Why: intensity labels make the prosody dial trainable instead of inferred; fillers are what make a
  voice agent sound present rather than synthetic.

## Batch 3 — Long-form monologues with natural pausing (~1 h)

45–90 s continuous explanations (process descriptions, storytelling), instructed to breathe and pause
naturally at clause boundaries, never rushing.

- Why: attacks the long-form weakness at the source. Enough of this data lets us unfreeze the duration
  predictor and eventually retire the 120-character chunking crutch instead of papering over it.

## Batch 4 — Readback and confirmation inventory (~0.75 h)

Telephone-style readbacks: phone numbers digit-grouped, order codes letter-by-letter (with Turkish
letter names), IBAN-style sequences, dates/times/amounts confirmed back ("... doğru mu?"), spelling
out names ("S-A-R-A-Ç, saraç").

- Why: CER on numbers is fine, but confirmation *rhythm* (grouping, checklist intonation) is a distinct
  register the agent uses constantly and the first campaign never captured.

## Recording specs (unchanged from campaign 1, restated)

- Same microphone, same room, same mouth distance as campaign 1; 48 kHz/24-bit WAV masters.
- Quiet-room noise floor check before each session; re-record any take with audible events.
- Session length ≤ 45 min; water breaks; no sessions when the contributor has any vocal strain.
- Slate each take with prompt ID; no personal data or improvised PII in any script.

## Script hygiene

- All prompts machine-generated and PII-free, following campaign 1 conventions.
- Dedupe every prompt against `configs/evaluation/turkish-v2.jsonl` and
  `configs/evaluation/turkish-pronunciation-v1.jsonl` (evaluation contamination check already exists
  in the collection tooling).
- Category + intensity labels ride in campaign metadata exactly like `campaign_category` today, so
  prosody-preset derivation and replay weighting work unchanged.

## Expected payoff

- Batches 1–2 → persona-flip reduction and preset quality (identity + style density).
- Batch 3 → unfreezing duration/long-form modeling.
- Batch 4 → readback register the agent needs daily.
- Training path is already proven: merge → upweight → consistency fine-tune → best-of-N; no new
  infrastructure required.
