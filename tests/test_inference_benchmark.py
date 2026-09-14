from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from turkish_tts.crossflow_train import CharacterTokenizer, validate_crossflow_synthesis_text
from turkish_tts.inference_benchmark import InferenceBenchmarkPlan, run_inference_benchmark


def test_inference_benchmark_measures_targets_rejections_and_queueing(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "version": "fixture-v1",
                "warmup": {"runs": 1, "text": "TARGET=1"},
                "duration_tolerance_ratio": 0.1,
                "concurrency": {
                    "levels": [1, 2],
                    "requests_per_level": 2,
                    "scenario_id": "five",
                },
                "scenarios": [
                    {
                        "id": "five",
                        "category": "latency",
                        "text": "TARGET=5",
                        "target_seconds": 5,
                    },
                    {
                        "id": "thirty",
                        "category": "long_form",
                        "text": "TARGET=30",
                        "target_seconds": 30,
                    },
                    {
                        "id": "ninety",
                        "category": "long_form",
                        "text": "TARGET=90",
                        "target_seconds": 90,
                    },
                    {
                        "id": "empty",
                        "category": "malformed",
                        "text": "",
                        "expect_success": False,
                    },
                ],
            }
        )
    )
    plan = InferenceBenchmarkPlan.load(plan_path)
    sample_rate = 100

    def synthesize(text: str, _seed: int) -> tuple[np.ndarray, str, float]:
        if not text:
            raise ValueError("empty input")
        seconds = int(text.split("=", maxsplit=1)[1])
        timeline = np.arange(seconds * sample_rate, dtype=np.float32) / sample_rate
        waveform = np.asarray(0.1 * np.sin(2 * np.pi * 5 * timeline), dtype=np.float32)
        return waveform, text.lower(), float(seconds * sample_rate)

    report_path = run_inference_benchmark(
        plan=plan,
        output_dir=tmp_path / "benchmark",
        sample_rate=sample_rate,
        synthesize=synthesize,
        provenance={"checkpoint": {"sha256": "fixture"}},
        seed=17,
    )
    report = json.loads(report_path.read_text())

    assert report["requirements_met"] is True
    assert report["sequential_summary"]["passed"] == 3
    assert report["sequential_summary"]["expected_rejections"] == 1
    assert [result["level"] for result in report["concurrency"]] == [1, 2]
    assert all(result["failures"] == 0 for result in report["concurrency"])
    assert report["runtime"]["streaming_supported"] is False
    assert all(
        sample["duration_within_tolerance"]
        for sample in report["samples"]
        if sample["status"] == "passed"
    )


def test_crossflow_synthesis_input_validation_rejects_malformed_and_oversized_text() -> None:
    tokenizer = CharacterTokenizer.from_texts(["merhaba dünya"])

    normalized, encoded = validate_crossflow_synthesis_text(
        "Merhaba Dünya", tokenizer, max_text_tokens=20
    )
    assert normalized == "merhaba dünya"
    assert encoded == tokenizer.encode(normalized)

    with pytest.raises(ValueError, match="empty"):
        validate_crossflow_synthesis_text("   ", tokenizer, max_text_tokens=20)
    with pytest.raises(ValueError, match="control"):
        validate_crossflow_synthesis_text("geçersiz\x00girdi", tokenizer, max_text_tokens=20)
    with pytest.raises(ValueError, match="maximum"):
        validate_crossflow_synthesis_text("a" * 21, tokenizer, max_text_tokens=20)


def test_crossflow_synthesis_dispatches_to_english_normalization() -> None:
    pytest.importorskip("alania.normalize")
    tokenizer = CharacterTokenizer.from_texts(["it costs three dollars and fifty cents"])

    normalized, encoded = validate_crossflow_synthesis_text(
        "It costs $3.50.",
        tokenizer,
        max_text_tokens=50,
        text_normalization="english",
    )

    assert normalized == "it costs three dollars and fifty cents."
    assert encoded == tokenizer.encode(normalized)
