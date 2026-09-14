#!/usr/bin/env bash
# Sweep guided-sampling configurations for one CrossFlow checkpoint and print a summary line per run.
# Paths below refer to the original training VM; adjust them for your environment.
set -euo pipefail

CHECKPOINT="$1"
SPEAKER="$2"
OUTPUT_ROOT="$3"
VOCODER=/mnt/disks/tts-data/models/bigvgan-v2-24khz-100band-256x
SUITE=configs/evaluation/turkish-v2.jsonl
REFERENCE=/mnt/disks/tts-data/manifests/candidate-b-scripted-complete-v1/prepared/candidate-b-scripted-complete-v1.test.jsonl

run_config() {
  local name="$1"
  shift
  local out="$OUTPUT_ROOT/$name"
  if [ ! -f "$out/quality.json" ]; then
    .venv/bin/python scripts/synthesize-crossflow.py \
      --checkpoint "$CHECKPOINT" \
      ${SPEAKER:+--speaker "$SPEAKER" --prosody 0 0 0 0 0 0} \
      --evaluation-suite "$SUITE" \
      --output-dir "$out/synthesis" \
      --vocoder "$VOCODER" \
      "$@" >/dev/null
    .venv/bin/turkish-tts baseline evaluate \
      --synthesis-report "$out/synthesis/crossflow.synthesis.json" \
      --reference-manifest "$REFERENCE" \
      --output "$out/quality.json" \
      --device cuda >/dev/null 2>&1
  fi
  python3 - "$name" "$out/quality.json" <<'EOF'
import json, sys
name, path = sys.argv[1], sys.argv[2]
summary = json.load(open(path))["summary"]
print(f"RESULT {name} cer={summary['cer_mean']} wer={summary['wer_mean']} sim={summary['speaker_similarity_mean']} sim_p10={summary['speaker_similarity_p10']}")
EOF
}

run_config sway08-steps32 --sway -0.8 --steps 32
run_config sway08-steps64 --sway -0.8 --steps 64
run_config sway10-steps32 --sway -1.0 --steps 32
run_config sway08-midpoint --sway -0.8 --steps 32 --solver midpoint
if [ -n "$SPEAKER" ]; then
  run_config sway08-spk15 --sway -0.8 --steps 32 --speaker-guidance 1.5
  run_config sway08-spk20 --sway -0.8 --steps 32 --speaker-guidance 2.0
  run_config sway08-spk30 --sway -0.8 --steps 32 --speaker-guidance 3.0
fi
run_config sway08-text15 --sway -0.8 --steps 32 --text-guidance 1.5
echo SWEEP-DONE
