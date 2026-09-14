#!/usr/bin/env bash
# Sweep high text-guidance scales for one CrossFlow checkpoint and print a summary line per run.
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

run_config text35 --sway -0.8 --steps 32 --text-guidance 3.5
run_config text40 --sway -0.8 --steps 32 --text-guidance 4.0
run_config text50 --sway -0.8 --steps 32 --text-guidance 5.0
run_config text35-steps64 --sway -0.8 --steps 64 --text-guidance 3.5
echo SWEEP-DONE
