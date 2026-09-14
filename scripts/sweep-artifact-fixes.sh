#!/usr/bin/env bash
# Sweep artifact-reduction sampling variants for the champion Voice B recipe.
# Paths below refer to the original training VM; adjust them for your environment.
set -euo pipefail

CHECKPOINT="$1"
OUTPUT_ROOT="$2"
VOCODER=/mnt/disks/tts-data/models/bigvgan-v2-24khz-100band-256x
SUITE=configs/evaluation/turkish-v2.jsonl
REFERENCE=/mnt/disks/tts-data/manifests/candidate-b-scripted-complete-v1/prepared/candidate-b-scripted-complete-v1.test.jsonl
PROFILE=(-1.2398956 1.1943912 -2.1267404 -0.9549347 0.9637866 0.5145879)

run_config() {
  local name="$1"
  shift
  local out="$OUTPUT_ROOT/$name"
  if [ ! -f "$out/quality.json" ]; then
    .venv/bin/python scripts/synthesize-crossflow.py \
      --checkpoint "$CHECKPOINT" \
      --speaker voicedata-candidate-b \
      --prosody "${PROFILE[@]}" \
      --sway -0.8 \
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
print(
    f"RESULT {name} cer={summary['cer_mean']} wer={summary['wer_mean']} "
    f"sim={summary['speaker_similarity_mean']} sim_p10={summary['speaker_similarity_p10']} "
    f"snr={summary['estimated_snr_db_mean']} clip={summary['clipping_ratio_max']}"
)
EOF
}

run_config rescale07 --steps 32 --text-guidance 4.0 --guidance-rescale 0.7
run_config clamp5 --steps 32 --text-guidance 4.0 --mel-clamp 5.0
run_config rescale07-clamp5 --steps 32 --text-guidance 4.0 --guidance-rescale 0.7 --mel-clamp 5.0
run_config text30-rescale07-clamp5 --steps 32 --text-guidance 3.0 --guidance-rescale 0.7 --mel-clamp 5.0
run_config rescale05-clamp5 --steps 32 --text-guidance 4.0 --guidance-rescale 0.5 --mel-clamp 5.0
echo SWEEP-DONE
