# Failure analysis — fastpitch-voicedata-adaptation-v1 (final, step 6000)

Date: 2026-08-01. Suite: configs/evaluation/turkish-v1.jsonl (39 prompts). Model: `artifacts/voicedata-adaptation-final/`.

## Per-category metrics (CER / WER / speaker sim)

| Category | CER | WER | Sim | Verdict |
|---|---:|---:|---:|---|
| acknowledgement | 0.272 | 0.583 | 0.414 | usable direction |
| emotional_style | 0.302 | 0.653 | 0.866 | good |
| voice_agent | 0.316 | 0.628 | 0.840 | good |
| questions_confirmations | 0.316 | 0.719 | 0.717 | good |
| long_form | 0.389 | 0.755 | 0.937 | decent |
| general | 0.393 | 0.682 | 0.776 | decent |
| names_places | 0.578 | 1.424 | 0.846 | broken |
| numeric | 0.607 | 1.032 | 0.897 | broken |
| foreign_abbreviations | 0.714 | 1.236 | 0.929 | broken |
| adversarial_normalization | 1.066 | 1.211 | 0.898 | fully broken |

## Root cause: acoustic vocabulary coverage, NOT text normalization

The deterministic normalizer expands every adversarial prompt correctly:

- `Sn. Yılmaz'ın 3. randevusu 07.08.2026'da saat 09:07'de` → `sayın yılmaz'ın üçüncü randevusu yedi ağustos iki bin yirmi altı'da saat dokuz sıfır yedi'de` (correct)
- `TCMB ve TÜİK` → `te ce me be ve te ü i ke` (correct letter spelling)
- Currency/percent/decimal/phone expansions all correct

The acoustic model then produces fluent word salad on these inputs (CER up to 1.6).
Spelled initialisms, long digit sequences, foreign names (Shakespeare, NASA, QR),
and rare inflected forms are absent from the 3.48 h VoiceData adaptation corpus.

## What works

Conversational core — acknowledgements, questions, confirmations, voice-agent turns,
emotional style — CER 0.27–0.32. Identity holds (sim 0.809 mean).

## Fix priority

1. **Data diversity** (highest leverage): bigger foundation from Common Voice with lifted
   per-speaker caps (67 h → up to 120 h filtered available); scripted CV sentences contain
   numbers, names, rare vocabulary. Then re-run adaptation + Stage C on top.
2. **Phoneme G2P**: grapheme model cannot generalize to unseen spelled forms and foreign
   words; phoneme input attacks this directly.
3. **Flow-matching acoustic model**: better OOD generalization than FastPitch duration/aligner
   stack; also removes the learned-aligner failure mode that killed foundation v1 and v2.
4. Do NOT touch the normalizer; do NOT add eval prompts from these categories to training.

## Contamination note

`ack-002` ("Anladım, teşekkür ederim.") collides with 2 Common Voice clips. Neither is in
any current training manifest (verified 2026-08-01). Guard now checks produced manifests.
