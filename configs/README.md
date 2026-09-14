# Configurations

## A note on paths

The training and inference configs in `crossflow/` are the **exact files used for the runs
reported in the paper**, unedited. Their `train_arrow`, `output_dir`, `initial_checkpoint`,
`vocoder`, and statistics paths therefore point at the original training VM
(`/mnt/disks/tts-data/...`, `/opt/f5-tts/data/...`). Those mounts do not exist anywhere else.

They are kept verbatim rather than sanitized so the lineage in the paper is auditable: each
config records which checkpoint it started from and which dataset it consumed. To re-run any
stage, point the paths at your own data.

The one config written for public use is `release/antalia-1-inference-recipe.json`, which
references Hugging Face repo ids instead of local paths and ships with the weights.

## Layout

| Directory | Contents |
|---|---|
| `crossflow/` | Training configs for every stage of the lineage, plus inference recipes v2-v8 |
| `evaluation/` | Prompt suites: `turkish-v2.jsonl` (120 prompts, all reported numbers), `turkish-v1.jsonl`, `turkish-pronunciation-v1.jsonl` |
| `release/` | The public inference recipe for the released weights |
| `inference/`, `fastpitch/` | Production inference settings and the superseded FastPitch baseline config |

## Lineage

The six stages that produced the released model, in order:

1. `crossflow/foundation-v3-normalized.json` — 100,000 updates, from scratch
2. `crossflow/foundation-speaker-v2.json` — 3,000 updates, speaker embeddings only
3. `crossflow/candidate-b-speaker-v1.json` — 100 updates, one new speaker embedding
4. `crossflow/cfg-foundation-v1.json` — 8,000 updates, classifier-free-guidance calibration
5. `crossflow/candidate-consistency-v1.json` — 6,000 updates, replay-heavy fine-tune
6. `crossflow/candidate-b-timbre-adapter-v2.json` — 1,400 updates, **released as Antalia 1**

`candidate-b` in file, run, and speaker identifiers is the project's internal label for the
released voice. In prose the same voice is called "Voice B"; the person behind it is credited
anonymously at her request.
