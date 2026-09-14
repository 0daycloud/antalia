# Antalia 1 — open Turkish text-to-speech

**Development of this model is discontinued.** This repository contains the training, inference,
and evaluation code for Antalia 1, a ~300M-parameter Turkish TTS model built on one consenting
speaker's recordings, published alongside the foundation model, the evaluation suites, and a
technical report. It is archived here so the parts that worked are not lost.

- Weights: https://huggingface.co/cloud0day3/antalia-1
- Foundation: https://huggingface.co/cloud0day3/antalia-1-foundation
- Evaluation data: https://huggingface.co/datasets/cloud0day3/antalia-eval
- Audio samples: https://0daycloud.github.io/antalia/
- Paper: https://arxiv.org/abs/TODO
- Contact: sezgin@patientdesk.ai or GitHub issues

## What it is

Antalia 1 is a CrossFlow rectified-flow acoustic model over 24 kHz, 100-band log-mel frames,
distributed with a self-describing release format (`config.json` + `model.safetensors`).
It speaks Turkish in one voice: that of a professional Turkish voice actor who consented to open
redistribution of the weights. There is no zero-shot cloning; speaker id 0 is the unconditioned
foundation path.

The voice is labelled "Voice B" in prose and `voicedata-candidate-b` in speaker identifiers,
run names, and config filenames. Both are internal labels; the person behind the voice is
credited anonymously at her request, and her raw recordings are not released.

On our 120-prompt Turkish evaluation suite (Whisper-large-v3, WavLM-x-vector similarity):

| Configuration | CER mean | Speaker sim. mean |
|---|---:|---:|
| Single seed, 32 steps | 0.0528 | 0.9331 |
| Best-of-8, timbre-gated | 0.0298 | 0.9445 |

Long inputs are chunked at ≤120 characters; numbers, foreign names, and normalization-heavy text
have the highest error rates; and automated speaker similarity of 0.93-0.94 does not certify
identity — a native listener rated 0/12 synthesized-vs-real pairs as the same person. See the paper.

## Install

```bash
uv sync            # or: pip install -e .
```

Python 3.12; CUDA GPU recommended (CPU works slowly). The vocoder is not in the package:
clone NVIDIA BigVGAN and apply the one-line hub compatibility patch:

```bash
git clone https://github.com/NVIDIA/BigVGAN /opt/bigvgan
git -C /opt/bigvgan checkout 7d2b454564a6c7d014227f635b7423881f14bdac
patch -d /opt/bigvgan -p4 < scripts/patches/bigvgan-huggingface-hub-1.patch
export PYTHONPATH=/opt/bigvgan
```

## Synthesize

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

`--checkpoint` accepts a Hub repo id, a local release directory, or a training `.pt` file.
`prosody-presets.json` ships in the weights repo next to `inference-recipe.json`, which records
the full champion recipe.

## Evaluate

```bash
python scripts/synthesize-crossflow.py \
  --checkpoint cloud0day3/antalia-1 \
  --evaluation-suite configs/evaluation/turkish-v2.jsonl \
  --seed-candidates 8 \
  --output-dir runs/analysis
python scripts/select-best-of-n.py \
  runs/analysis/crossflow.candidates.json \
  --envelope-stats envelope-stats.json \
  --timbre-stats timbre-profile.json
antalia baseline evaluate \
  --synthesis-report runs/analysis/crossflow.synthesis.json \
  --reference-manifest /path/to/reference.jsonl \
  --output runs/analysis/quality.json
```

## Train / lineage

The six-stage training lineage is preserved in `configs/crossflow/`:

```
foundation-v3-normalized.json          (100,000 updates, all params)
foundation-speaker-v2.json             (3,000 updates, speaker embeddings)
candidate-b-speaker-v1.json            (100 updates, one new speaker embedding)
cfg-foundation-v1.json                 (8,000 updates, CFG calibration)
candidate-consistency-v1.json          (6,000 updates, replay-heavy fine-tune)
candidate-b-timbre-adapter-v2.json     (1,400 updates, released)
```

## Repository layout

- `src/turkish_tts/` — model, training, inference, evaluation, normalizer, release format
- `scripts/` — synthesis, best-of-N selection, export, dataset/manifest builders, sweep shells
- `configs/` — training, inference, evaluation, campaign manifest configurations
- `reports/` — CMOS results, failure analysis, found-audio pilot, campaign brief
- `docs/lessons.md` — what measurably improved the model, with before/after numbers

## Limitations

1. Voice similarity gap (above).
2. Long inputs need chunking.
3. Numbers, normalization, foreign names — weakest categories.
4. Single-seed variance is large; headline numbers depend on best-of-8.
5. One fixed voice, Turkish only, 24 kHz, grapheme input.
6. No memorization audit was run; no audio watermark is embedded.

## License

Code: Apache-2.0. Model weights: Antalia Open RAIL-M — no impersonation, fraud, deception,
political robocalls, or use without AI-disclosure disclosure. See HF model cards.

## Contributing

Community pull requests are accepted. There is no roadmap and no support SLA.

## Citation

```bibtex
@misc{antalia1_2026,
  title  = {Antalia 1: An Open Turkish Text-to-Speech Model from a Rights-Clean Pipeline},
  author = {Saygili, Sezgin and Kaplaner, Emre and Ozgul, Oncel and Koktas, Fikri San},
  year   = {2026},
  note   = {arXiv:TODO},
  url    = {https://github.com/0daycloud/antalia}
}
```
