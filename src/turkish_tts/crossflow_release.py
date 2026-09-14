"""Public release format for CrossFlow checkpoints.

A release directory holds ``model.safetensors`` (EMA-applied weights) and ``config.json`` with
everything inference needs: architecture, mel front-end statistics, text normalization mode,
character vocabulary, speaker vocabulary, vocoder pointer, and provenance. Training-only state
(optimizer, raw model weights, internal paths) is not exported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RELEASE_FORMAT = "antalia-crossflow-release-v1"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
DEFAULT_VOCODER_REPO = "nvidia/bigvgan_v2_24khz_100band_256x"

_AUDIO_FIELDS = (
    "sample_rate",
    "n_fft",
    "hop_length",
    "win_length",
    "n_mels",
    "f_min",
    "f_max",
    "mel_mean",
    "mel_std",
    "mel_normalized_clip",
)


def is_release_directory(path: Path) -> bool:
    return path.is_dir() and (path / CONFIG_FILENAME).is_file() and (path / WEIGHTS_FILENAME).is_file()


def resolve_release_path(reference: str) -> Path:
    """Return a local release directory for a path or a Hugging Face repo id.

    A string that does not exist locally and looks like ``owner/name`` is downloaded from the Hub.
    """
    local = Path(reference)
    if local.exists():
        return local
    if reference.count("/") == 1 and not reference.startswith((".", "/")):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=reference, allow_patterns=[CONFIG_FILENAME, WEIGHTS_FILENAME]))
    raise FileNotFoundError(f"checkpoint does not exist locally and is not a Hub repo id: {reference}")


def export_crossflow_release(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    use_ema: bool = True,
    vocoder_repo: str = DEFAULT_VOCODER_REPO,
    vocoder_revision: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``model.safetensors`` and ``config.json`` for a training checkpoint."""
    import torch
    from safetensors.torch import save_file

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {name: tensor.detach().clone().contiguous() for name, tensor in payload["model"].items()}
    weights = "raw"
    if use_ema and "values" in payload["ema"]:
        for name, value in payload["ema"]["values"].items():
            if name not in state:
                raise KeyError(f"EMA tensor is not part of the model state: {name}")
            state[name] = value.detach().clone().contiguous()
        weights = "ema"
    train_config = payload["train_config"]
    provenance = payload.get("provenance") or {}
    config: dict[str, Any] = {
        "format": RELEASE_FORMAT,
        "architecture": "CrossFlow",
        "model_config": payload["model_config"],
        "audio": {field: train_config[field] for field in _AUDIO_FIELDS},
        "text_normalization": train_config.get("text_normalization", "turkish"),
        "vocabulary": list(payload["vocabulary"]),
        "speaker_vocabulary": list(payload["speaker_vocabulary"]) if payload.get("speaker_vocabulary") else None,
        "vocoder": {"repo_id": vocoder_repo, "revision": vocoder_revision},
        "source": {
            "run_version": payload["run_version"],
            "update": int(payload["update"]),
            "epoch": int(payload["epoch"]),
            "weights": weights,
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "provenance": {"statement": provenance.get("statement")},
        "parameters": int(sum(tensor.numel() for tensor in state.values())),
    }
    if extra:
        config.update(extra)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_dir / WEIGHTS_FILENAME), metadata={"format": "pt", "release": RELEASE_FORMAT})
    (output_dir / CONFIG_FILENAME).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return config


def load_release_payload(release_dir: Path, device: str) -> tuple[Any, dict[str, Any]]:
    """Build a CrossFlow model from a release directory and a checkpoint-compatible payload."""
    from safetensors.torch import load_file

    from turkish_tts.crossflow import CrossFlow, CrossFlowModelConfig

    config = json.loads((release_dir / CONFIG_FILENAME).read_text())
    if config.get("format") != RELEASE_FORMAT:
        raise ValueError(f"unsupported release format: {config.get('format')!r}")
    model = CrossFlow(CrossFlowModelConfig.from_dict(config["model_config"]))
    model.load_state_dict(load_file(str(release_dir / WEIGHTS_FILENAME)), strict=True)
    model.to(device).eval()
    source = config["source"]
    payload: dict[str, Any] = {
        "format_version": config["format"],
        "run_version": source["run_version"],
        "update": source["update"],
        "epoch": source["epoch"],
        "model_config": config["model_config"],
        "train_config": {
            "run_version": source["run_version"],
            "train_arrow": "",
            "validation_arrow": "",
            "output_dir": "",
            "text_normalization": config["text_normalization"],
            **config["audio"],
        },
        "vocabulary": config["vocabulary"],
        "speaker_vocabulary": config.get("speaker_vocabulary"),
        "ema": {},
        "provenance": config.get("provenance", {}),
        "vocoder": config.get("vocoder", {}),
    }
    return model, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
