---
title: "Antalia 1: an open Turkish text-to-speech model, released as we stop working on it"
authors:
- user: cloud0day3
---

# Antalia 1: an open Turkish text-to-speech model, released as we stop working on it

Antalia 1 is a single-voice Turkish text-to-speech model built at Patientdesk.ai. Development has been discontinued. We are releasing the weights, the training and inference code, the evaluation suites, and a technical report so that the parts that worked are not lost, and so that the parts that did not work are documented rather than quietly forgotten. The model is usable, and its remaining problems are real; both are described below with numbers.

- Weights: [cloud0day3/antalia-1](https://huggingface.co/cloud0day3/antalia-1) (fine-tuned voice) and [cloud0day3/antalia-1-foundation](https://huggingface.co/cloud0day3/antalia-1-foundation) (speaker-agnostic base)
- Code: [github.com/0daycloud/antalia](https://github.com/0daycloud/antalia)
- Audio samples: [0daycloud.github.io/antalia](https://0daycloud.github.io/antalia/)
- Technical report: [arXiv](https://github.com/0daycloud/antalia/blob/main/paper/main.pdf)

## What it is

Antalia 1 speaks Turkish in one voice: that of a professional Turkish voice actor who recorded a scripted corpus for the project and signed a redistribution addendum that covers open weights. She is credited anonymously at her request. Her raw recordings are not released.

The acoustic model is CrossFlow, an independent implementation of character-conditioned rectified flow matching over 100-band log-mel frames at 24 kHz (hop 256). There is no phoneme front-end and no aligner: text goes through a deterministic Turkish normalizer (numbers, dates, currency and abbreviations spelled out, Turkish lowercasing) and is fed as graphemes to a 4-block ConvNeXt-style character encoder. A single scalar head predicts the total frame count. The decoder is 16 Transformer blocks (dim 768, 12 heads, SwiGLU) with self-attention over mel frames, cross-attention over text, and adaLN modulation from the timestep. The released voice model has 304,552,293 parameters; the final training stage trained only a 3.16M-parameter residual adapter and style vector on top of the frozen network. Waveforms come from NVIDIA's BigVGAN v2 (24 kHz, 100-band, 256x), which we do not redistribute; the loader pulls it from the Hub at a pinned commit.

The foundation model was trained from scratch for 100,000 updates on one A100-80GB using 61,469 clips (67.55 hours) of Common Voice 26.0 Turkish and FLEURS Turkish that passed our filters. The voice was then added through speaker conditioning, classifier-free-guidance and consistency fine-tunes, and finally the adapter stage, which used 1,073 corrected voice segments (5.008 hours).

## What works

On our 120-prompt evaluation suite (10 categories, 12 prompts each, scored with Whisper-large-v3 and WavLM x-vector similarity):

| Configuration | CER mean | CER p90 | WER mean | Speaker sim. mean |
|---|---:|---:|---:|---:|
| Single seed, 32 steps | 0.0528 | 0.1348 | 0.1297 | 0.9331 |
| Best-of-8 with timbre gate | 0.0298 | 0.1007 | 0.0934 | 0.9445 |

For comparison, the unconditioned foundation model sits at CER 0.1835, and our earlier FastPitch baseline scored CER 0.27 to 0.32 on conversational categories and 0.58 to 1.07 on names, numbers, foreign terms and normalization-heavy text. The move to flow matching on a larger Common Voice foundation fixed the acoustic-vocabulary problem that broke that baseline.

Conversational speech is the strong case. In our single-listener CMOS session, the emotion and general categories scored +2.0 in favor of the model, questions +0.2 and short acknowledgements 0.0. The vocoder is transparent (within 0.4 dB on real-mel reconstruction) and estimated SNR of the output is about 50 dB. On an A100 with the model resident, a 5-second utterance takes roughly 2.1 to 2.3 seconds at 32 steps; a 16-step variant cuts generation time by 45.1% with mean similarity within 0.0006 of the 32-step result, at the cost of lower p10 similarity.

Six prosody presets ship with the weights, along with the timbre profile and F0-envelope statistics used by the best-of-N selector.

## What does not

**Voice identity is the biggest gap.** The automated speaker similarity of 0.93 to 0.94 does not survive a human ear. In the CMOS session (one native listener who knows the voice, 27 non-catch trials, 3/3 catch trials clean), synthesized speech versus a real recording scored −1.833 ± 0.757 (n=12), and the listener rated the synthesized clip as the same person in 0 of 12 pairs. The listener localized the difference to timbre: the synthesized voice carries about +4.0 dB of excess energy in the 4.7 to 6.8 kHz band and a tilt in the formant region. The vocoder was ruled out. We responded with a timbre penalty in the selector and the adapter stage, which lowered the mean timbre penalty from 2.585 to 1.997, still outside the 0.47 to 0.98 range measured on real recordings.

**No multi-listener evaluation exists.** A three-or-more-listener CMOS protocol was designed and never run. Every human number above comes from a single listener who is also the project owner. There is no MOS.

**Long inputs need chunking.** The total-duration head under-budgets long text. One 370-character input produced 9.1 seconds of rushed speech unchunked versus 34.8 seconds when split at clause boundaries. The shipped recipe chunks at 120 characters with a 160 ms pause and applies a floor of 0.085 seconds per character; this cut long-form WER from 36.6% to 10.7% in a single-seed measurement. Without it the model rushes and mumbles.

**Numbers, normalization-heavy text and foreign names are the weakest categories.** In the CMOS session these scored −2.0 (numeric), −2.5 (normalization) and −1.14 (foreign). Best-of-8 recovers some prompts (a normalization prompt from 0.32 raw CER to 0.13; a foreign-term prompt from 0.24 to 0.04) and not others (a normalization prompt at 0.23 and a numeric prompt at 0.13 on all eight seeds).

**The headline numbers depend on best-of-8.** Per-prompt CER across seeds spreads from 0.0 to 0.17. The best-of-8 figure requires running Whisper-large-v3 and WavLM on eight candidates, eight times the compute of a single pass.

Also: one fixed voice, no zero-shot cloning, Turkish only, 24 kHz, grapheme input. No memorization audit was run. There is no watermark.

## Why we are publishing anyway, and why we are stopping

The hosted product this model was built for was cancelled, and with it the recording campaigns and retune that were meant to close the identity gap. Nothing runs as a service; there is no roadmap. Community pull requests are welcome, but we will not be developing the model further.

We are publishing because the rights-clean pipeline is worth more than the checkpoint. Every training clip is traceable: 120,407 Common Voice 26 clips decoded and hashed, 112,909 scanned with forced-Turkish Whisper, rejections logged by reason; voice recordings gated by dual-ASR consensus, CTC alignment, a spoken-PII scan and a speaker-consistency check; evaluation prompts deduplicated against training manifests. The voice actor's consent explicitly covers open weights. The release format carries the mel statistics, normalizer mode, vocabularies, vocoder pointer and provenance in one config file. That combination is rare in open Turkish TTS, and it is reusable even if the voice model itself is not what anyone deploys.

Two lessons we would pass on. First, sign redistribution consent before recording, not after. Second, automated speaker similarity saturates. A verifier at 0.91 to 0.94 told us the identity problem was solved while a person who knows the voice said "not the same person" twelve times out of twelve. If speaker identity matters, budget for native listening early, with more than one listener.

## How to run it

Clone the repository, install the package with `pip install -e .`, and put the pinned BigVGAN source tree on your path (the repository ships the one-line `huggingface_hub` compatibility patch it needs). Then:

```bash
python scripts/synthesize-crossflow.py \
  --checkpoint cloud0day3/antalia-1 \
  --vocoder nvidia/bigvgan_v2_24khz_100band_256x \
  --speaker voicedata-candidate-b \
  --prosody-presets prosody-presets.json --preset warm_voice_agent \
  --text-guidance 4.0 --sway -0.8 --steps 32 --mel-clamp 5.0 \
  --min-seconds-per-char 0.085 --chunk-chars 120 \
  --text "Merhaba, ben Antalia." --output out.wav
```

`prosody-presets.json` is in the weights repository next to `inference-recipe.json`, which records the full champion recipe including the eight seeds and the selection score used for best-of-8. `--seed-candidates 8` plus `scripts/select-best-of-n.py` reproduces the selected numbers above; a single seed reproduces the single-seed row.

## License and use

Code is Apache-2.0. Weights are released under OpenRAIL-M with use restrictions: no impersonation, fraud, deception or political robocalls, and generated audio must be disclosed as AI-generated. The voice belongs to a real person who agreed to this release under those terms; please respect them. The "Antalia" name is unregistered; it is unrelated to the Italian business "ANTALIA AI".

## Credits

The voice actor, who remains anonymous by choice and without whom there is no model. The contributors to Mozilla Common Voice 26 Turkish and to FLEURS, whose recordings make up the foundation. NVIDIA, for BigVGAN v2. Authors: Sezgin Saygili, Emre Kaplaner, Oncel Ozgul and Fikri San Koktas, Patientdesk.ai. Questions to sezgin@patientdesk.ai or GitHub issues.
