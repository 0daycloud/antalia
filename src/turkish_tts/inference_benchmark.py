from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import soundfile as sf

from turkish_tts.audio import analyze_signal


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    id: str
    category: str
    text: str
    target_seconds: float | None
    expect_success: bool


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkPlan:
    version: str
    warmup_text: str
    warmup_runs: int
    scenarios: tuple[BenchmarkScenario, ...]
    concurrency_levels: tuple[int, ...]
    requests_per_level: int
    concurrency_scenario_id: str
    duration_tolerance_ratio: float

    @classmethod
    def load(cls, path: Path) -> InferenceBenchmarkPlan:
        payload = orjson.loads(path.read_bytes())
        raw_scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
        if not isinstance(raw_scenarios, list) or not raw_scenarios:
            raise ValueError("benchmark plan must contain scenarios")
        scenarios = tuple(
            BenchmarkScenario(
                id=_required_string(row.get("id"), "scenario id"),
                category=_required_string(row.get("category"), "scenario category"),
                text=str(row.get("text", "")),
                target_seconds=(
                    _positive_float(row["target_seconds"], "target_seconds")
                    if row.get("target_seconds") is not None
                    else None
                ),
                expect_success=bool(row.get("expect_success", True)),
            )
            for row in raw_scenarios
            if isinstance(row, dict)
        )
        if len(scenarios) != len(raw_scenarios):
            raise ValueError("every benchmark scenario must be an object")
        ids = [scenario.id for scenario in scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("benchmark scenario IDs must be unique")
        required_durations = {5.0, 30.0, 90.0}
        configured_durations = {
            scenario.target_seconds for scenario in scenarios if scenario.target_seconds is not None
        }
        if not required_durations.issubset(configured_durations):
            raise ValueError("benchmark plan must include 5, 30, and 90 second scenarios")
        if not any(not scenario.expect_success for scenario in scenarios):
            raise ValueError("benchmark plan must include at least one expected input rejection")

        concurrency = payload.get("concurrency")
        if not isinstance(concurrency, dict):
            raise ValueError("benchmark plan must contain concurrency settings")
        levels = tuple(int(level) for level in concurrency.get("levels", []))
        if not levels or any(level < 1 for level in levels):
            raise ValueError("concurrency levels must be positive integers")
        scenario_id = _required_string(concurrency.get("scenario_id"), "concurrency scenario_id")
        scenario_by_id = {scenario.id: scenario for scenario in scenarios}
        if scenario_id not in scenario_by_id or not scenario_by_id[scenario_id].expect_success:
            raise ValueError("concurrency scenario must name a successful scenario")

        warmup = payload.get("warmup")
        if not isinstance(warmup, dict):
            raise ValueError("benchmark plan must contain warmup settings")
        return cls(
            version=_required_string(payload.get("version"), "benchmark version"),
            warmup_text=_required_string(warmup.get("text"), "warmup text"),
            warmup_runs=_positive_int(warmup.get("runs"), "warmup runs"),
            scenarios=scenarios,
            concurrency_levels=levels,
            requests_per_level=_positive_int(concurrency.get("requests_per_level"), "requests_per_level"),
            concurrency_scenario_id=scenario_id,
            duration_tolerance_ratio=_ratio(payload.get("duration_tolerance_ratio", 0.35), "duration_tolerance_ratio"),
        )


SynthesisFunction = Callable[[str, int], tuple[np.ndarray[Any, np.dtype[np.float32]], str, float]]


def run_inference_benchmark(
    *,
    plan: InferenceBenchmarkPlan,
    output_dir: Path,
    sample_rate: int,
    synthesize: SynthesisFunction,
    provenance: dict[str, Any],
    seed: int,
    gpu_memory: Callable[[], dict[str, int]] | None = None,
) -> Path:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    warmup_started = time.monotonic()
    for index in range(plan.warmup_runs):
        synthesize(plan.warmup_text, seed + index)
    warmup_seconds = time.monotonic() - warmup_started

    samples: list[dict[str, Any]] = []
    for index, scenario in enumerate(plan.scenarios, start=1):
        samples.append(
            _measure_request(
                scenario=scenario,
                request_id=f"sequential-{index:03d}",
                output_path=audio_dir / f"{index:03d}-{scenario.id}.wav",
                sample_rate=sample_rate,
                synthesize=synthesize,
                seed=seed + 1000 + index,
                target_tolerance=plan.duration_tolerance_ratio,
            )
        )

    concurrency_scenario = next(scenario for scenario in plan.scenarios if scenario.id == plan.concurrency_scenario_id)
    concurrency_results: list[dict[str, Any]] = []
    synthesis_lock = threading.Lock()
    concurrent_seed = seed + 10_000
    for level in plan.concurrency_levels:
        level_started = time.monotonic()

        def execute(request_index: int, concurrency_level: int = level) -> dict[str, Any]:
            submitted = time.monotonic()
            with synthesis_lock:
                service_started = time.monotonic()
                result = _measure_request(
                    scenario=concurrency_scenario,
                    request_id=f"concurrency-{concurrency_level}-{request_index:03d}",
                    output_path=None,
                    sample_rate=sample_rate,
                    synthesize=synthesize,
                    seed=concurrent_seed + concurrency_level * 1000 + request_index,
                    target_tolerance=plan.duration_tolerance_ratio,
                )
            result["queue_seconds"] = service_started - submitted
            result["request_latency_seconds"] = time.monotonic() - submitted
            return result

        with ThreadPoolExecutor(max_workers=level) as executor:
            requests = list(executor.map(execute, range(plan.requests_per_level)))
        wall_seconds = time.monotonic() - level_started
        request_latencies = [float(row["request_latency_seconds"]) for row in requests]
        queue_latencies = [float(row["queue_seconds"]) for row in requests]
        concurrency_results.append(
            {
                "level": level,
                "requests": len(requests),
                "scheduling": "serialized-single-gpu-worker",
                "wall_seconds": wall_seconds,
                "requests_per_second": len(requests) / max(wall_seconds, 1e-9),
                "request_latency_seconds": _distribution(request_latencies),
                "queue_seconds": _distribution(queue_latencies),
                "failures": sum(row["status"] != "passed" for row in requests),
            }
        )

    successful = [row for row in samples if row["status"] == "passed"]
    requirements = {
        "expected_outcomes": all(row["expectation_met"] for row in samples),
        "target_durations": all(
            row.get("duration_within_tolerance", True) for row in samples if row["expected_success"]
        ),
        "signal_stability": all(row.get("signal_stable", False) for row in samples if row["status"] == "passed"),
        "concurrency": all(result["failures"] == 0 for result in concurrency_results),
    }
    report = {
        "benchmark_version": plan.version,
        "created_unix_seconds": time.time(),
        "provenance": provenance,
        "runtime": {
            "sample_rate_hz": sample_rate,
            "streaming_supported": False,
            "first_audio_semantics": (
                "full-response latency; current runtime emits audio only after synthesis completes"
            ),
            "concurrency_scheduling": "one serialized GPU synthesis worker with concurrent request queueing",
            "warmup_runs": plan.warmup_runs,
            "warmup_seconds": warmup_seconds,
            "gpu_memory_bytes": gpu_memory() if gpu_memory is not None else None,
        },
        "samples": samples,
        "sequential_summary": {
            "requests": len(samples),
            "passed": len(successful),
            "expected_rejections": sum(row["status"] == "rejected" and row["expectation_met"] for row in samples),
            "unexpected_failures": sum(not row["expectation_met"] for row in samples),
            "latency_seconds": _distribution([float(row["latency_seconds"]) for row in successful]),
            "real_time_factor": _distribution([float(row["real_time_factor"]) for row in successful]),
        },
        "concurrency": concurrency_results,
        "requirements": requirements,
        "requirements_met": all(requirements.values()),
    }
    report_path = output_dir / "inference-benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report_path


def benchmark_crossflow_checkpoint(
    *,
    checkpoint_path: Path,
    benchmark_plan: Path,
    output_dir: Path,
    vocoder_dir: Path,
    device: str = "cuda",
    steps: int = 32,
    seed: int = 20260803,
    duration_scale: float = 1.0,
) -> Path:
    import torch

    from turkish_tts.crossflow_train import (
        CrossFlowTrainConfig,
        MelNormalizer,
        _load_crossflow_vocoder,
        _synthesize_loaded_crossflow,
        load_crossflow_checkpoint,
    )

    plan = InferenceBenchmarkPlan.load(benchmark_plan)
    model, tokenizer, payload = load_crossflow_checkpoint(checkpoint_path, device, use_ema=True)
    train_config = CrossFlowTrainConfig(**payload["train_config"])
    mel_normalizer = MelNormalizer(train_config, device)
    vocoder = _load_crossflow_vocoder(vocoder_dir, device)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    def synthesize(text: str, request_seed: int) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str, float]:
        waveform, normalized, predicted_frames = _synthesize_loaded_crossflow(
            model=model,
            tokenizer=tokenizer,
            vocoder=vocoder,
            mel_normalizer=mel_normalizer,
            text=text,
            sample_rate=train_config.sample_rate,
            hop_length=train_config.hop_length,
            device=device,
            steps=steps,
            seed=request_seed,
            duration_scale=duration_scale,
            speaker_id=0,
            prosody=None,
            text_guidance_scale=1.0,
            speaker_guidance_scale=1.0,
            sway_coefficient=0.0,
            solver="euler",
            guidance_rescale=0.0,
            mel_clamp=None,
            reference=None,
            context_guidance_scale=1.0,
            min_seconds_per_char=0.0,
            chunk_character_limit=0,
            chunk_pause_seconds=0.0,
            text_normalization=train_config.text_normalization,
        )
        return np.asarray(waveform, dtype=np.float32), normalized, predicted_frames

    def gpu_memory() -> dict[str, int]:
        if torch_device.type != "cuda":
            return {"allocated": 0, "reserved": 0}
        return {
            "allocated": int(torch.cuda.max_memory_allocated(torch_device)),
            "reserved": int(torch.cuda.max_memory_reserved(torch_device)),
        }

    provenance = {
        "checkpoint": checkpoint_provenance(checkpoint_path, payload),
        "vocoder": vocoder_provenance(vocoder_dir),
        "inference": {
            "steps": steps,
            "seed": seed,
            "duration_scale": duration_scale,
            "device": device,
        },
    }
    return run_inference_benchmark(
        plan=plan,
        output_dir=output_dir,
        sample_rate=train_config.sample_rate,
        synthesize=synthesize,
        provenance=provenance,
        seed=seed,
        gpu_memory=gpu_memory,
    )


def checkpoint_provenance(checkpoint_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_version": payload.get("run_version"),
        "checkpoint_update": payload.get("update"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_weights": "ema",
        "training_provenance": payload.get("provenance"),
    }


def vocoder_provenance(vocoder_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in vocoder_dir.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(path.relative_to(vocoder_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"vocoder directory contains no files: {vocoder_dir}")
    manifest_digest = hashlib.sha256(orjson.dumps(files, option=orjson.OPT_SORT_KEYS)).hexdigest()
    return {
        "path": str(vocoder_dir),
        "files": files,
        "manifest_sha256": manifest_digest,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure_request(
    *,
    scenario: BenchmarkScenario,
    request_id: str,
    output_path: Path | None,
    sample_rate: int,
    synthesize: SynthesisFunction,
    seed: int,
    target_tolerance: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        waveform, normalized, predicted_frames = synthesize(scenario.text, seed)
        latency = time.monotonic() - started
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("synthesis returned invalid audio")
        metrics = analyze_signal(audio, sample_rate)
        if output_path is not None:
            sf.write(output_path, audio, sample_rate)
        audio_seconds = metrics.duration_seconds
        duration_error = (
            abs(audio_seconds - scenario.target_seconds) / scenario.target_seconds
            if scenario.target_seconds is not None
            else None
        )
        status = "passed"
        expectation_met = scenario.expect_success
        signal_stable = (
            metrics.clipping_ratio <= 0.001 and metrics.rms_dbfs >= -50.0 and metrics.active_frame_ratio >= 0.1
        )
        return {
            "request_id": request_id,
            "id": scenario.id,
            "category": scenario.category,
            "text": scenario.text,
            "normalized_text": normalized,
            "expected_success": scenario.expect_success,
            "status": status,
            "expectation_met": expectation_met,
            "latency_seconds": latency,
            "first_audio_seconds": latency,
            "audio_seconds": audio_seconds,
            "real_time_factor": latency / max(audio_seconds, 1e-9),
            "predicted_frames": predicted_frames,
            "target_seconds": scenario.target_seconds,
            "duration_error_ratio": duration_error,
            "duration_within_tolerance": (duration_error <= target_tolerance if duration_error is not None else True),
            "signal": metrics.as_metadata(),
            "signal_stable": signal_stable,
            "output_path": str(output_path) if output_path is not None else None,
            "seed": seed,
        }
    except (ValueError, RuntimeError) as error:
        latency = time.monotonic() - started
        return {
            "request_id": request_id,
            "id": scenario.id,
            "category": scenario.category,
            "text": scenario.text,
            "expected_success": scenario.expect_success,
            "status": "failed" if scenario.expect_success else "rejected",
            "expectation_met": not scenario.expect_success,
            "latency_seconds": latency,
            "first_audio_seconds": None,
            "error": {"type": type(error).__name__, "message": str(error)},
            "target_seconds": scenario.target_seconds,
            "seed": seed,
        }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _ratio(value: object, name: str) -> float:
    parsed = _positive_float(value, name)
    if parsed >= 1.0:
        raise ValueError(f"{name} must be less than one")
    return parsed
