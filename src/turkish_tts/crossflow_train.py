from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import orjson
import soundfile as sf

from turkish_tts.normalize import normalize_for_model, turkish_lower


@dataclass(frozen=True)
class CrossFlowTrainConfig:
    run_version: str
    train_arrow: str
    validation_arrow: str
    output_dir: str
    text_normalization: str = "turkish"
    initial_checkpoint: str | None = None
    initial_checkpoint_sha256: str | None = None
    initialize_from_ema: bool = False
    sample_rate: int = 24_000
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    n_mels: int = 100
    f_min: float = 0.0
    f_max: float = 12_000.0
    mel_mean: float | list[float] | None = None
    mel_std: float | list[float] | None = None
    mel_normalized_clip: float = 5.0
    model_dim: int = 512
    depth: int = 12
    heads: int = 8
    ff_dim: int = 2_048
    text_depth: int = 4
    text_kernel_size: int = 7
    dropout: float = 0.0
    checkpoint_activations: bool = True
    adapter_dim: int = 0
    train_adapter_only: bool = False
    speaker_conditioning: bool = False
    speaker_embedding_dim: int = 256
    prosody_dim: int = 0
    initialize_conditioning_from_base: bool = False
    new_speaker_id: str | None = None
    freeze_linguistic_components: bool = False
    train_speaker_embedding_only: bool = False
    train_conditioning_only: bool = False
    train_context_only: bool = False
    text_dropout_probability: float = 0.0
    speaker_dropout_probability: float = 0.0
    context_conditioning: bool = False
    global_reference_conditioning: bool = False
    initialize_context_from_base: bool = False
    infill_probability: float = 0.0
    infill_context_min_fraction: float = 0.1
    infill_context_max_fraction: float = 0.5
    infill_prefix_probability: float = 0.0
    paired_reference_conditioning: bool = False
    max_reference_seconds: float = 12.0
    max_audio_seconds: float = 24.0
    max_text_tokens: int = 512
    frames_per_gpu: int = 7_200
    max_samples_per_gpu: int = 32
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_updates: int = 2_000
    max_updates: int = 100_000
    duration_weight: float = 0.1
    foundation_preservation_weight: float = 0.0
    speaker_consistency_weight: float = 0.0
    speaker_consistency_every_updates: int = 8
    speaker_consistency_max_seconds: float = 4.0
    speaker_consistency_reference_manifest: str | None = None
    vocoder_path: str | None = None
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    save_every_updates: int = 1_000
    validate_every_updates: int = 1_000
    validation_batches: int = 16
    validation_degradation_factor: float = 3.0
    validation_degradation_patience: int = 2
    log_every_updates: int = 10
    num_workers: int = 8
    seed: int = 20260803
    compile_model: bool = False
    provenance_statement: str = (
        "Independent implementation trained only on the project's cleared Turkish manifests; "
        "no FreyaTTS code, weights, audio, latents, or training data used."
    )

    @classmethod
    def load(cls, path: Path) -> CrossFlowTrainConfig:
        payload = orjson.loads(path.read_bytes())
        if not isinstance(payload, dict):
            raise ValueError(f"configuration must be an object: {path}")
        config = cls(**payload)
        if config.text_normalization not in {"turkish", "english"}:
            raise ValueError("text_normalization must be turkish or english")
        if bool(config.initial_checkpoint) != bool(config.initial_checkpoint_sha256):
            raise ValueError("initial_checkpoint and initial_checkpoint_sha256 must be set together")
        if (config.mel_mean is None) != (config.mel_std is None):
            raise ValueError("mel_mean and mel_std must be set together")
        if isinstance(config.mel_mean, list) and len(config.mel_mean) != config.n_mels:
            raise ValueError("mel_mean must be a scalar or contain exactly n_mels values")
        if isinstance(config.mel_std, list) and len(config.mel_std) != config.n_mels:
            raise ValueError("mel_std must be a scalar or contain exactly n_mels values")
        if config.mel_std is not None:
            standard_deviations = config.mel_std if isinstance(config.mel_std, list) else [config.mel_std]
            if any(value <= 0.0 for value in standard_deviations):
                raise ValueError("mel standard deviations must be positive")
        if config.mel_normalized_clip <= 0.0:
            raise ValueError("mel_normalized_clip must be positive")
        if config.train_adapter_only and config.freeze_linguistic_components:
            raise ValueError("adapter-only training already freezes all base components")
        if config.adapter_dim < 0:
            raise ValueError("adapter_dim cannot be negative")
        if config.train_adapter_only and config.adapter_dim == 0:
            raise ValueError("adapter-only training requires a positive adapter_dim")
        if config.train_adapter_only and config.initial_checkpoint is None:
            raise ValueError("adapter-only training requires an initial checkpoint")
        if config.train_speaker_embedding_only and config.new_speaker_id is None:
            raise ValueError("speaker-embedding-only training requires new_speaker_id")
        if config.train_speaker_embedding_only and config.train_adapter_only:
            raise ValueError("speaker embedding and adapter-only training are mutually exclusive")
        exclusive_training_modes = sum(
            (
                config.train_adapter_only,
                config.train_speaker_embedding_only,
                config.train_conditioning_only,
                config.train_context_only,
            )
        )
        if exclusive_training_modes > 1:
            raise ValueError("adapter, speaker embedding, conditioning, and context-only modes are exclusive")
        if config.train_conditioning_only and config.initial_checkpoint is None:
            raise ValueError("conditioning-only training requires an initial checkpoint")
        if config.train_conditioning_only and not (config.speaker_conditioning or config.prosody_dim > 0):
            raise ValueError("conditioning-only training requires acoustic conditions")
        if config.train_context_only and config.initial_checkpoint is None:
            raise ValueError("context-only training requires an initial checkpoint")
        if config.train_context_only and not config.context_conditioning:
            raise ValueError("context-only training requires context conditioning")
        if not 0.0 <= config.text_dropout_probability < 1.0:
            raise ValueError("text_dropout_probability must be in [0, 1)")
        if not 0.0 <= config.speaker_dropout_probability < 1.0:
            raise ValueError("speaker_dropout_probability must be in [0, 1)")
        if config.text_dropout_probability > 0.0 and config.duration_weight != 0.0:
            raise ValueError("text dropout corrupts duration targets; set duration_weight to zero")
        if config.speaker_dropout_probability > 0.0 and not config.speaker_conditioning:
            raise ValueError("speaker dropout requires speaker conditioning")
        if not 0.0 <= config.infill_probability < 1.0:
            raise ValueError("infill_probability must be in [0, 1)")
        if config.infill_probability > 0.0 and not config.context_conditioning:
            raise ValueError("infill training requires context conditioning")
        if not 0.0 < config.infill_context_min_fraction <= config.infill_context_max_fraction < 1.0:
            raise ValueError("infill context fractions must satisfy 0 < min <= max < 1")
        if not 0.0 <= config.infill_prefix_probability <= 1.0:
            raise ValueError("infill_prefix_probability must be in [0, 1]")
        if config.initialize_context_from_base and config.initial_checkpoint is None:
            raise ValueError("context initialization requires an initial checkpoint")
        if config.initialize_context_from_base and not config.context_conditioning:
            raise ValueError("context initialization requires context conditioning")
        if config.global_reference_conditioning and not config.context_conditioning:
            raise ValueError("global reference conditioning requires context conditioning")
        if config.paired_reference_conditioning and not config.context_conditioning:
            raise ValueError("paired-reference training requires context conditioning")
        if config.max_reference_seconds <= 0.0:
            raise ValueError("max_reference_seconds must be positive")
        if config.speaker_embedding_dim < 1:
            raise ValueError("speaker_embedding_dim must be positive")
        if config.prosody_dim < 0:
            raise ValueError("prosody_dim cannot be negative")
        if config.initialize_conditioning_from_base and config.initial_checkpoint is None:
            raise ValueError("conditioning initialization requires an initial checkpoint")
        if config.initialize_conditioning_from_base and not (config.speaker_conditioning or config.prosody_dim > 0):
            raise ValueError("conditioning initialization requires acoustic conditions")
        if config.new_speaker_id is not None and not config.speaker_conditioning:
            raise ValueError("new_speaker_id requires speaker conditioning")
        if config.new_speaker_id is not None and config.initial_checkpoint is None:
            raise ValueError("new_speaker_id requires an initial checkpoint")
        if config.foundation_preservation_weight < 0.0:
            raise ValueError("foundation_preservation_weight cannot be negative")
        if config.speaker_consistency_weight < 0.0:
            raise ValueError("speaker_consistency_weight cannot be negative")
        if config.speaker_consistency_weight > 0.0 and not config.speaker_consistency_reference_manifest:
            raise ValueError("speaker consistency training requires a reference manifest for the target centroid")
        if config.speaker_consistency_weight > 0.0 and not config.vocoder_path:
            raise ValueError("speaker consistency training requires vocoder_path")
        if config.speaker_consistency_every_updates < 1:
            raise ValueError("speaker_consistency_every_updates must be at least one")
        if config.speaker_consistency_max_seconds <= 0.0:
            raise ValueError("speaker_consistency_max_seconds must be positive")
        if config.validation_degradation_factor <= 1.0:
            raise ValueError("validation_degradation_factor must be greater than one")
        if config.validation_degradation_patience < 1:
            raise ValueError("validation_degradation_patience must be at least one")
        return config

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CharacterTokenizer:
    def __init__(self, symbols: Sequence[str]) -> None:
        if len(symbols) != len(set(symbols)):
            raise ValueError("vocabulary contains duplicate symbols")
        self.symbols = tuple(symbols)
        self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}
        if self.symbols[:2] != ("<pad>", "<unk>"):
            raise ValueError("vocabulary must begin with <pad> and <unk>")

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> CharacterTokenizer:
        characters = sorted({character for text in texts for character in text})
        return cls(("<pad>", "<unk>", *characters))

    @classmethod
    def load(cls, path: Path) -> CharacterTokenizer:
        payload = orjson.loads(path.read_bytes())
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError(f"invalid vocabulary: {path}")
        return cls(payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.symbols, option=orjson.OPT_INDENT_2) + b"\n")

    def encode(self, text: str) -> list[int]:
        unknown = self.symbol_to_id["<unk>"]
        return [self.symbol_to_id.get(character, unknown) for character in text]


class SpeakerVocabulary:
    UNCONDITIONED = "<unconditioned>"

    def __init__(self, speakers: Sequence[str]) -> None:
        if not speakers or speakers[0] != self.UNCONDITIONED:
            raise ValueError("speaker vocabulary must begin with <unconditioned>")
        if len(speakers) != len(set(speakers)):
            raise ValueError("speaker vocabulary contains duplicate IDs")
        self.speakers = tuple(speakers)
        self.speaker_to_id = {speaker: index for index, speaker in enumerate(self.speakers)}

    @classmethod
    def from_speakers(cls, speakers: Sequence[str]) -> SpeakerVocabulary:
        unique = sorted({speaker for speaker in speakers if speaker})
        return cls((cls.UNCONDITIONED, *unique))

    def with_speaker(self, speaker: str) -> SpeakerVocabulary:
        if not speaker or speaker == self.UNCONDITIONED:
            raise ValueError("new speaker ID must be non-empty and conditioned")
        if speaker in self.speaker_to_id:
            raise ValueError(f"speaker ID already exists: {speaker}")
        return SpeakerVocabulary((*self.speakers, speaker))

    def encode(self, speaker: str | None) -> int:
        if speaker is None:
            return 0
        return self.speaker_to_id.get(speaker, 0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.speakers, option=orjson.OPT_INDENT_2) + b"\n")


@dataclass(frozen=True)
class AcousticRecord:
    audio_path: str
    text: str
    duration: float
    speaker: str | None
    prosody: tuple[float, ...]
    reference_audio_path: str | None
    reference_text: str | None
    reference_duration: float

    @property
    def total_duration(self) -> float:
        return self.duration + self.reference_duration


class ArrowAcousticDataset:
    def __init__(
        self,
        arrow_path: Path,
        tokenizer: CharacterTokenizer | None = None,
        speaker_vocabulary: SpeakerVocabulary | None = None,
        *,
        max_audio_seconds: float,
        max_text_tokens: int,
        max_reference_seconds: float = 12.0,
        require_speaker: bool = False,
        prosody_dim: int = 0,
    ) -> None:
        from datasets import Dataset  # type: ignore[import-untyped]

        raw = Dataset.from_file(str(arrow_path))
        records: list[AcousticRecord] = []
        for row in raw:
            raw_text = row["text"]
            text = "".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
            duration = float(row.get("duration", 0.0))
            audio_path = str(row["audio_path"])
            if not text or duration <= 0.0 or duration > max_audio_seconds:
                continue

            raw_reference_audio_path = row.get("reference_audio_path")
            raw_reference_text = row.get("reference_text")
            raw_reference_duration = row.get("reference_duration")
            has_reference_identity = raw_reference_audio_path is not None or raw_reference_text is not None
            reference_audio_path: str | None = None
            reference_text: str | None = None
            reference_duration = 0.0
            if not has_reference_identity:
                if raw_reference_duration not in (None, 0, 0.0):
                    raise ValueError(f"unreferenced rows must have zero reference duration in {arrow_path}")
            else:
                if (
                    raw_reference_audio_path is None
                    or raw_reference_text is None
                    or raw_reference_duration is None
                ):
                    raise ValueError(f"reference fields must be provided together in {arrow_path}")
                reference_text = (
                    "".join(raw_reference_text) if isinstance(raw_reference_text, list) else str(raw_reference_text)
                )
                reference_audio_path = str(raw_reference_audio_path)
                reference_duration = float(raw_reference_duration)
                if not reference_text or reference_duration <= 0.0:
                    raise ValueError(f"reference text and duration must be positive in {arrow_path}")
                if reference_duration > max_reference_seconds:
                    continue
                if duration + reference_duration > max_audio_seconds:
                    continue

            combined_text_length = len(text) + (len(reference_text) + 1 if reference_text is not None else 0)
            if combined_text_length > max_text_tokens:
                continue
            raw_speaker = row.get("speaker")
            speaker = str(raw_speaker) if raw_speaker is not None else None
            raw_prosody = row.get("prosody", [])
            if not isinstance(raw_prosody, (list, tuple)):
                raise ValueError(f"prosody must be a sequence in {arrow_path}")
            prosody = tuple(float(value) for value in raw_prosody)
            if require_speaker and not speaker:
                raise ValueError(f"speaker is required in {arrow_path}")
            if len(prosody) != prosody_dim:
                raise ValueError(f"prosody must contain {prosody_dim} values in {arrow_path}; got {len(prosody)}")
            records.append(
                AcousticRecord(
                    audio_path=audio_path,
                    text=text,
                    duration=duration,
                    speaker=speaker,
                    prosody=prosody,
                    reference_audio_path=reference_audio_path,
                    reference_text=reference_text,
                    reference_duration=reference_duration,
                )
            )
        if not records:
            raise ValueError(f"no usable records in {arrow_path}")
        self.records = records
        self.tokenizer = tokenizer
        self.speaker_vocabulary = speaker_vocabulary

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("dataset tokenizer has not been assigned")
        record = self.records[index]
        if self.speaker_vocabulary is None and record.speaker is not None:
            raise RuntimeError("dataset speaker vocabulary has not been assigned")
        audio, sample_rate = sf.read(record.audio_path, dtype="float32", always_2d=True)
        waveform = np.mean(audio, axis=1, dtype=np.float32)
        reference_waveform: np.ndarray[Any, np.dtype[np.float32]] = np.empty(0, dtype=np.float32)
        reference_sample_rate = sample_rate
        reference_token_ids: list[int] = []
        if record.reference_audio_path is not None and record.reference_text is not None:
            reference_audio, reference_sample_rate = sf.read(
                record.reference_audio_path,
                dtype="float32",
                always_2d=True,
            )
            reference_waveform = cast(
                np.ndarray[Any, np.dtype[np.float32]],
                np.mean(reference_audio, axis=1, dtype=np.float32),
            )
            reference_token_ids = self.tokenizer.encode(f"{record.reference_text} ")
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "token_ids": self.tokenizer.encode(record.text),
            "text": record.text,
            "duration": record.duration,
            "audio_path": record.audio_path,
            "speaker_id": (
                self.speaker_vocabulary.encode(record.speaker) if self.speaker_vocabulary is not None else 0
            ),
            "prosody": record.prosody,
            "has_reference": record.reference_audio_path is not None,
            "reference_waveform": reference_waveform,
            "reference_sample_rate": reference_sample_rate,
            "reference_token_ids": reference_token_ids,
            "reference_text": record.reference_text,
            "reference_duration": record.reference_duration,
            "reference_audio_path": record.reference_audio_path,
        }


class DurationBatchSampler:
    def __init__(
        self,
        durations: Sequence[float],
        *,
        sample_rate: int,
        hop_length: int,
        frames_per_gpu: int,
        max_samples: int,
        seed: int,
        rank: int,
        world_size: int,
        shuffle: bool,
        drop_remainder: bool = True,
    ) -> None:
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.epoch = 0
        ordered = sorted(range(len(durations)), key=lambda index: durations[index])
        batches: list[list[int]] = []
        batch: list[int] = []
        frames = 0
        for index in ordered:
            item_frames = max(1, math.ceil(durations[index] * sample_rate / hop_length))
            if batch and (frames + item_frames > frames_per_gpu or len(batch) >= max_samples):
                batches.append(batch)
                batch = []
                frames = 0
            batch.append(index)
            frames += item_frames
        if batch:
            batches.append(batch)
        usable = len(batches) - (len(batches) % world_size)
        self.batches = batches[:usable] if drop_remainder else batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.batches)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        for position in indices[self.rank :: self.world_size]:
            yield self.batches[position]

    def __len__(self) -> int:
        return len(range(self.rank, len(self.batches), self.world_size))


class LogMelFrontend:
    def __init__(self, config: CrossFlowTrainConfig, device: str) -> None:
        import torch
        import torchaudio

        self.torch = torch
        self.device = torch.device(device)
        self.config = config
        self.window = torch.hann_window(config.win_length, device=self.device)
        self.mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=config.n_fft // 2 + 1,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).to(self.device)

    def __call__(self, waveforms: Any, sample_lengths: Any) -> tuple[Any, Any]:
        torch = self.torch
        padding = (self.config.n_fft - self.config.hop_length) // 2
        waveforms = torch.nn.functional.pad(waveforms.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spectrum = torch.stft(
            waveforms,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()
        mel = torch.einsum("bft,fm->btm", spectrum, self.mel_basis)
        mel = torch.log(mel.clamp_min(1e-5))
        frame_lengths = torch.div(sample_lengths, self.config.hop_length, rounding_mode="floor").clamp_min(1)
        frame_lengths = frame_lengths.clamp_max(mel.shape[1])
        return mel, frame_lengths


class MelNormalizer:
    def __init__(self, config: CrossFlowTrainConfig, device: str) -> None:
        import torch

        self.clip = config.mel_normalized_clip
        if config.mel_mean is None or config.mel_std is None:
            self.mean = torch.zeros(config.n_mels, device=device)
            self.std = torch.ones(config.n_mels, device=device)
            self.enabled = False
        else:
            self.mean = torch.tensor(config.mel_mean, device=device, dtype=torch.float32)
            self.std = torch.tensor(config.mel_std, device=device, dtype=torch.float32)
            self.enabled = True

    def normalize(self, mel: Any) -> Any:
        normalized = (mel - self.mean) / self.std
        return normalized.clamp(min=-self.clip, max=self.clip)

    def denormalize(self, mel: Any) -> Any:
        return mel * self.std + self.mean


class SpeakerConsistencyScorer:
    """Differentiable speaker-embedding distance against a fixed target centroid.

    Decodes the model's predicted clean mel through the frozen production vocoder,
    resamples to 16 kHz, and scores with the same frozen WavLM x-vector model used
    by offline evaluation. Gradients flow to the acoustic model; the vocoder and
    speaker encoder remain frozen.
    """

    SPEAKER_MODEL_ID = "microsoft/wavlm-base-plus-sv"
    SPEAKER_MODEL_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"

    def __init__(
        self,
        *,
        reference_manifest: Path,
        vocoder_dir: Path,
        device: str,
        sample_rate: int,
        hop_length: int,
    ) -> None:
        import torch
        import torchaudio
        from transformers import WavLMForXVector

        self._torch = torch
        self._device = device
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        self._vocoder = _load_crossflow_vocoder(vocoder_dir, device)
        for parameter in self._vocoder.parameters():
            parameter.requires_grad_(False)
        self._vocoder.eval()
        speaker_model: Any = WavLMForXVector.from_pretrained(
            self.SPEAKER_MODEL_ID,
            revision=self.SPEAKER_MODEL_REVISION,
        )
        self._speaker_model = speaker_model.to(device)
        for parameter in self._speaker_model.parameters():
            parameter.requires_grad_(False)
        self._speaker_model.eval()
        self._resampler = torchaudio.transforms.Resample(sample_rate, 16_000).to(device)
        self._centroid = self._build_centroid(reference_manifest)

    def _embed_16k(self, audio: Any) -> Any:
        torch = self._torch
        normalized = (audio - audio.mean(dim=-1, keepdim=True)) / audio.var(dim=-1, keepdim=True, unbiased=False).add(
            1e-5
        ).sqrt()
        embeddings = self._speaker_model(input_values=normalized).embeddings
        return torch.nn.functional.normalize(embeddings.float(), dim=-1)

    def _build_centroid(self, reference_manifest: Path) -> Any:
        torch = self._torch
        vectors: list[Any] = []
        with reference_manifest.open(encoding="utf-8") as handle:
            rows = [orjson.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"speaker consistency reference manifest is empty: {reference_manifest}")
        with torch.inference_mode():
            for row in rows:
                audio_path = str(row.get("audio_filepath", ""))
                if not audio_path or not Path(audio_path).is_file():
                    raise FileNotFoundError(f"speaker consistency reference audio missing: {audio_path}")
                audio, source_rate = sf.read(audio_path, dtype="float32", always_2d=True)
                mono = torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32)).to(self._device)
                if source_rate != self._sample_rate:
                    raise ValueError(f"reference audio must be {self._sample_rate} Hz: {audio_path}")
                resampled = self._resampler(mono)
                vectors.append(self._embed_16k(resampled.unsqueeze(0)).squeeze(0))
        centroid = torch.stack(vectors).mean(dim=0)
        return torch.nn.functional.normalize(centroid, dim=-1).detach()

    def loss(
        self,
        predicted_mel: Any,
        scored_frame_mask: Any,
        mel_normalizer: MelNormalizer,
        *,
        max_seconds: float,
    ) -> Any | None:
        """Cosine distance of vocoded predicted mel to the target centroid.

        Returns None when the batch has no utterance long enough to score.
        """
        utterance_frames = scored_frame_mask.sum(dim=1)
        common_frames = int(utterance_frames.min().item())
        max_frames = int(max_seconds * self._sample_rate / self._hop_length)
        crop_frames = min(common_frames, max_frames)
        if crop_frames * self._hop_length < self._sample_rate:
            return None
        cropped = predicted_mel[:, :crop_frames, :]
        mel = mel_normalizer.denormalize(cropped).transpose(1, 2).float()
        waveform = self._vocoder(mel).squeeze(1)
        audio_16k = self._resampler(waveform)
        embeddings = self._embed_16k(audio_16k)
        similarity = (embeddings * self._centroid.unsqueeze(0)).sum(dim=-1)
        return (1.0 - similarity).mean()


class ExponentialMovingAverage:
    def __init__(self, model: Any, decay: float) -> None:
        self.decay = decay
        self.updates = 0
        self.values = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad
        }

    def update(self, model: Any) -> None:
        self.updates += 1
        warmup_decay = (1.0 + self.updates) / (10.0 + self.updates)
        decay = min(self.decay, warmup_decay)
        for name, parameter in model.named_parameters():
            if name in self.values:
                self.values[name].lerp_(parameter.detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {"updates": self.updates, "values": self.values}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.updates = int(state["updates"])
        for name, value in state["values"].items():
            if name in self.values:
                self.values[name].copy_(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collate(batch: Sequence[dict[str, Any]], expected_sample_rate: int) -> dict[str, Any]:
    import torch
    import torchaudio.functional as audio_functional

    waveforms: list[Any] = []
    token_sequences: list[Any] = []
    reference_waveforms: list[Any] = []
    reference_token_sequences: list[Any] = []
    has_references: list[bool] = []
    speaker_ids: list[int] = []
    prosody_features: list[tuple[float, ...]] = []
    for sample in batch:
        waveform = torch.from_numpy(sample["waveform"])
        sample_rate = int(sample["sample_rate"])
        if sample_rate != expected_sample_rate:
            waveform = audio_functional.resample(waveform, sample_rate, expected_sample_rate)
        reference_waveform = torch.from_numpy(sample["reference_waveform"])
        reference_sample_rate = int(sample["reference_sample_rate"])
        if reference_waveform.numel() > 0 and reference_sample_rate != expected_sample_rate:
            reference_waveform = audio_functional.resample(
                reference_waveform,
                reference_sample_rate,
                expected_sample_rate,
            )
        waveforms.append(waveform)
        token_sequences.append(torch.tensor(sample["token_ids"], dtype=torch.long))
        reference_waveforms.append(reference_waveform)
        reference_token_sequences.append(torch.tensor(sample["reference_token_ids"], dtype=torch.long))
        has_references.append(bool(sample["has_reference"]))
        speaker_ids.append(int(sample["speaker_id"]))
        prosody_features.append(tuple(sample["prosody"]))
    sample_lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    token_lengths = torch.tensor([tokens.numel() for tokens in token_sequences], dtype=torch.long)
    reference_sample_lengths = torch.tensor(
        [waveform.numel() for waveform in reference_waveforms],
        dtype=torch.long,
    )
    reference_token_lengths = torch.tensor(
        [tokens.numel() for tokens in reference_token_sequences],
        dtype=torch.long,
    )
    return {
        "waveforms": torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True),
        "sample_lengths": sample_lengths,
        "token_ids": torch.nn.utils.rnn.pad_sequence(token_sequences, batch_first=True),
        "token_lengths": token_lengths,
        "reference_waveforms": torch.nn.utils.rnn.pad_sequence(reference_waveforms, batch_first=True),
        "reference_sample_lengths": reference_sample_lengths,
        "reference_token_ids": torch.nn.utils.rnn.pad_sequence(reference_token_sequences, batch_first=True),
        "reference_token_lengths": reference_token_lengths,
        "has_references": torch.tensor(has_references, dtype=torch.bool),
        "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
        "prosody_features": torch.tensor(prosody_features, dtype=torch.float32),
    }


def _prepend_paired_references(
    mel: Any,
    frame_lengths: Any,
    token_ids: Any,
    token_lengths: Any,
    reference_mel: Any,
    reference_frame_lengths: Any,
    reference_token_ids: Any,
    reference_token_lengths: Any,
    has_references: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    if has_references.shape != (mel.shape[0],):
        raise ValueError("has_references must contain one flag per batch item")
    effective_reference_frames = torch.where(
        has_references,
        reference_frame_lengths,
        torch.zeros_like(reference_frame_lengths),
    )
    effective_reference_tokens = torch.where(
        has_references,
        reference_token_lengths,
        torch.zeros_like(reference_token_lengths),
    )
    combined_frame_lengths = frame_lengths + effective_reference_frames
    combined_token_lengths = token_lengths + effective_reference_tokens
    combined_mel = mel.new_zeros((mel.shape[0], int(combined_frame_lengths.max().item()), mel.shape[2]))
    combined_token_ids = token_ids.new_zeros((token_ids.shape[0], int(combined_token_lengths.max().item())))
    context_frame_mask = torch.zeros(
        combined_mel.shape[:2],
        device=mel.device,
        dtype=torch.bool,
    )
    for row in range(mel.shape[0]):
        reference_frames = int(effective_reference_frames[row].item())
        target_frames = int(frame_lengths[row].item())
        reference_tokens = int(effective_reference_tokens[row].item())
        target_tokens = int(token_lengths[row].item())
        if reference_frames:
            combined_mel[row, :reference_frames] = reference_mel[row, :reference_frames]
            context_frame_mask[row, :reference_frames] = True
        combined_mel[row, reference_frames : reference_frames + target_frames] = mel[row, :target_frames]
        if reference_tokens:
            combined_token_ids[row, :reference_tokens] = reference_token_ids[row, :reference_tokens]
        combined_token_ids[row, reference_tokens : reference_tokens + target_tokens] = token_ids[row, :target_tokens]
    return (
        combined_mel,
        combined_frame_lengths,
        combined_token_ids,
        combined_token_lengths,
        context_frame_mask,
    )


def _make_masks(frame_lengths: Any, token_lengths: Any, frame_count: int, token_count: int) -> tuple[Any, Any]:
    import torch

    frame_positions = torch.arange(frame_count, device=frame_lengths.device).unsqueeze(0)
    token_positions = torch.arange(token_count, device=token_lengths.device).unsqueeze(0)
    return frame_positions < frame_lengths.unsqueeze(1), token_positions < token_lengths.unsqueeze(1)


def _learning_rate(config: CrossFlowTrainConfig, update: int) -> float:
    if update < config.warmup_updates:
        return config.learning_rate * (update + 1) / max(config.warmup_updates, 1)
    progress = min(
        1.0,
        (update - config.warmup_updates) / max(config.max_updates - config.warmup_updates, 1),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def _checkpoint_payload(
    *,
    model: Any,
    ema: ExponentialMovingAverage,
    optimizer: Any,
    config: CrossFlowTrainConfig,
    model_config: Any,
    tokenizer: CharacterTokenizer,
    speaker_vocabulary: SpeakerVocabulary | None,
    update: int,
    epoch: int,
    train_arrow_sha256: str,
) -> dict[str, Any]:
    provenance = {
        "statement": config.provenance_statement,
        "train_arrow": config.train_arrow,
        "train_arrow_sha256": train_arrow_sha256,
    }
    if config.initial_checkpoint is not None:
        provenance["initial_checkpoint"] = config.initial_checkpoint
        provenance["initial_checkpoint_sha256"] = config.initial_checkpoint_sha256 or ""
    return {
        "format_version": 2,
        "run_version": config.run_version,
        "update": update,
        "epoch": epoch,
        "model_config": model_config.as_dict(),
        "train_config": config.as_dict(),
        "vocabulary": list(tokenizer.symbols),
        "speaker_vocabulary": (list(speaker_vocabulary.speakers) if speaker_vocabulary is not None else None),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "provenance": provenance,
    }


def _should_save_checkpoint(
    *,
    update: int,
    max_updates: int,
    save_every_updates: int,
    best_validation_update: int,
) -> bool:
    return update % save_every_updates == 0 or update in (max_updates, best_validation_update)


def _load_initial_crossflow_state(
    model: Any,
    model_config: Any,
    payload: dict[str, Any],
    *,
    initialize_from_ema: bool,
    adapter_only: bool,
    initialize_conditioning_from_base: bool = False,
    expand_speaker_embedding: bool = False,
    initialize_context_from_base: bool = False,
) -> None:
    import torch

    from turkish_tts.crossflow import CrossFlowModelConfig

    initial_config = CrossFlowModelConfig.from_dict(payload["model_config"])
    expected_payload = model_config.as_dict()
    if adapter_only:
        expected_payload["adapter_dim"] = 0
    if initialize_conditioning_from_base:
        expected_payload["speaker_count"] = initial_config.speaker_count
        expected_payload["speaker_embedding_dim"] = initial_config.speaker_embedding_dim
        expected_payload["prosody_dim"] = initial_config.prosody_dim
    if expand_speaker_embedding:
        expected_payload["speaker_count"] = initial_config.speaker_count
    if initialize_context_from_base:
        expected_payload["context_conditioning"] = initial_config.context_conditioning
        expected_payload["global_reference_conditioning"] = initial_config.global_reference_conditioning
    expected_config = CrossFlowModelConfig.from_dict(expected_payload)
    if initial_config != expected_config:
        raise ValueError("initial checkpoint architecture does not match training configuration")
    initial_state = dict(payload["model"])
    if initialize_from_ema and "values" in payload["ema"]:
        initial_state.update(payload["ema"]["values"])
    expected_missing: set[str] = set()
    if adapter_only:
        expected_missing.update(model.adapter_parameter_names())
    if initialize_conditioning_from_base:
        expected_missing.update(model.conditioning_parameter_names())
    initial_speaker_embedding = initial_state.get("speaker_embedding.weight")
    if expand_speaker_embedding:
        if initial_speaker_embedding is None or model.speaker_embedding is None:
            raise ValueError("speaker expansion requires conditioned checkpoint weights")
        initial_state.pop("speaker_embedding.weight")
        expected_missing.add("speaker_embedding.weight")
    if initialize_context_from_base and not initial_config.context_conditioning:
        expected_missing.update(model.context_parameter_names())
    if not expected_missing:
        model.load_state_dict(initial_state, strict=True)
        return
    incompatible = model.load_state_dict(initial_state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise ValueError(f"initialization mismatch: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}")
    if expand_speaker_embedding:
        assert initial_speaker_embedding is not None
        assert model.speaker_embedding is not None
        if initial_speaker_embedding.shape[0] + 1 != model.speaker_embedding.weight.shape[0]:
            raise ValueError("speaker expansion must add exactly one embedding")
        with torch.no_grad():
            model.speaker_embedding.weight[:-1].copy_(initial_speaker_embedding)


def train_crossflow(config_path: Path, *, resume_path: Path | None = None) -> Path:
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader

    from turkish_tts.crossflow import (
        CrossFlow,
        CrossFlowModelConfig,
        flow_matching_loss,
        parameter_count,
    )

    config = CrossFlowTrainConfig.load(config_path)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_run = world_size > 1
    if distributed_run:
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ArrowAcousticDataset(
        Path(config.train_arrow),
        max_audio_seconds=config.max_audio_seconds,
        max_text_tokens=config.max_text_tokens,
        max_reference_seconds=config.max_reference_seconds,
        require_speaker=config.speaker_conditioning,
        prosody_dim=config.prosody_dim,
    )
    if config.paired_reference_conditioning and not any(
        record.reference_audio_path is not None for record in train_dataset.records
    ):
        raise ValueError("paired-reference training requires referenced records")
    vocabulary_path = output_dir / "vocab.json"
    speaker_vocabulary_path = output_dir / "speakers.json"
    if resume_path is not None and config.initial_checkpoint is not None:
        raise ValueError("resume_path and initial_checkpoint are mutually exclusive")
    resume_payload: dict[str, Any] | None = None
    initialization_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        tokenizer = CharacterTokenizer(resume_payload["vocabulary"])
    elif config.initial_checkpoint is not None:
        initial_path = Path(config.initial_checkpoint)
        digest_holder: list[str | None] = [_sha256(initial_path) if rank == 0 else None]
        if distributed_run:
            distributed.broadcast_object_list(digest_holder, src=0)
        if digest_holder[0] != config.initial_checkpoint_sha256:
            raise ValueError(
                f"initial checkpoint checksum mismatch: expected {config.initial_checkpoint_sha256}, "
                f"got {digest_holder[0]}"
            )
        initialization_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        tokenizer = CharacterTokenizer(initialization_payload["vocabulary"])
    elif vocabulary_path.is_file():
        tokenizer = CharacterTokenizer.load(vocabulary_path)
    else:
        tokenizer = CharacterTokenizer.from_texts(
            [text for record in train_dataset.records for text in (record.text, record.reference_text or "")]
        )

    speaker_vocabulary: SpeakerVocabulary | None = None
    if config.speaker_conditioning:
        checkpoint_speakers: object = None
        if resume_payload is not None:
            checkpoint_speakers = resume_payload.get("speaker_vocabulary")
        elif initialization_payload is not None:
            checkpoint_speakers = initialization_payload.get("speaker_vocabulary")
        if checkpoint_speakers is not None:
            if not isinstance(checkpoint_speakers, (list, tuple)) or not all(
                isinstance(item, str) for item in checkpoint_speakers
            ):
                raise ValueError("checkpoint speaker vocabulary is invalid")
            speaker_vocabulary = SpeakerVocabulary(checkpoint_speakers)
        else:
            speaker_vocabulary = SpeakerVocabulary.from_speakers(
                [record.speaker or "" for record in train_dataset.records]
            )
        if config.new_speaker_id is not None:
            if checkpoint_speakers is None:
                raise ValueError("new speaker expansion requires a conditioned checkpoint")
            speaker_vocabulary = speaker_vocabulary.with_speaker(config.new_speaker_id)
        unknown_train_speakers = sorted(
            {
                record.speaker
                for record in train_dataset.records
                if record.speaker is not None and record.speaker not in speaker_vocabulary.speaker_to_id
            }
        )
        if unknown_train_speakers:
            raise ValueError(f"training data contains unknown speakers: {unknown_train_speakers!r}")

    if rank == 0:
        tokenizer.save(vocabulary_path)
        if speaker_vocabulary is not None:
            speaker_vocabulary.save(speaker_vocabulary_path)
    if distributed_run:
        distributed.barrier()
    train_dataset.tokenizer = tokenizer
    train_dataset.speaker_vocabulary = speaker_vocabulary
    validation_dataset = ArrowAcousticDataset(
        Path(config.validation_arrow),
        tokenizer,
        speaker_vocabulary,
        max_audio_seconds=config.max_audio_seconds,
        max_text_tokens=config.max_text_tokens,
        max_reference_seconds=config.max_reference_seconds,
        require_speaker=config.speaker_conditioning,
        prosody_dim=config.prosody_dim,
    )
    if config.paired_reference_conditioning and not any(
        record.reference_audio_path is not None for record in validation_dataset.records
    ):
        raise ValueError("paired-reference validation requires referenced records")
    validation_dataset.tokenizer = tokenizer
    unknown_validation_speakers = sorted(
        {
            record.speaker
            for record in validation_dataset.records
            if speaker_vocabulary is not None
            and record.speaker is not None
            and record.speaker not in speaker_vocabulary.speaker_to_id
        }
    )
    if unknown_validation_speakers:
        raise ValueError(f"validation data contains unknown speakers: {unknown_validation_speakers!r}")

    train_sampler = DurationBatchSampler(
        [record.total_duration for record in train_dataset.records],
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        frames_per_gpu=config.frames_per_gpu,
        max_samples=config.max_samples_per_gpu,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    validation_sampler = DurationBatchSampler(
        [record.total_duration for record in validation_dataset.records],
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        frames_per_gpu=config.frames_per_gpu,
        max_samples=config.max_samples_per_gpu,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_remainder=False,
    )
    collate = partial(_collate, expected_sample_rate=config.sample_rate)
    train_loader: DataLoader[Any] = DataLoader(
        cast(Any, train_dataset),
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    validation_loader: DataLoader[Any] = DataLoader(
        cast(Any, validation_dataset),
        batch_sampler=validation_sampler,
        collate_fn=collate,
        num_workers=min(config.num_workers, 4),
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    model_config = CrossFlowModelConfig(
        vocab_size=len(tokenizer.symbols),
        mel_channels=config.n_mels,
        model_dim=config.model_dim,
        depth=config.depth,
        heads=config.heads,
        ff_dim=config.ff_dim,
        text_depth=config.text_depth,
        text_kernel_size=config.text_kernel_size,
        dropout=config.dropout,
        checkpoint_activations=config.checkpoint_activations,
        max_frames=math.ceil(config.max_audio_seconds * config.sample_rate / config.hop_length),
        max_text_tokens=config.max_text_tokens,
        adapter_dim=config.adapter_dim,
        speaker_count=len(speaker_vocabulary.speakers) if speaker_vocabulary is not None else 0,
        speaker_embedding_dim=config.speaker_embedding_dim,
        prosody_dim=config.prosody_dim,
        context_conditioning=config.context_conditioning,
        global_reference_conditioning=config.global_reference_conditioning,
    )
    raw_model = CrossFlow(model_config).to(device)
    if config.train_adapter_only:
        raw_model.freeze_base_for_adapter_training()
    if config.freeze_linguistic_components:
        raw_model.freeze_linguistic_components()
    if config.train_speaker_embedding_only:
        raw_model.freeze_for_speaker_embedding_training()
    if config.train_conditioning_only:
        raw_model.freeze_for_conditioning_training()
    if config.train_context_only:
        raw_model.freeze_for_context_training()
    trainable_parameters = [parameter for parameter in raw_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    ema = ExponentialMovingAverage(raw_model, config.ema_decay)
    update = 0
    epoch = 0
    if resume_payload is not None:
        raw_model.load_state_dict(resume_payload["model"], strict=True)
        if "values" in resume_payload["ema"]:
            ema.load_state_dict(resume_payload["ema"])
        else:
            ema = ExponentialMovingAverage(raw_model, config.ema_decay)
        optimizer.load_state_dict(resume_payload["optimizer"])
        update = int(resume_payload["update"])
        epoch = int(resume_payload["epoch"])
    elif initialization_payload is not None:
        _load_initial_crossflow_state(
            raw_model,
            model_config,
            initialization_payload,
            initialize_from_ema=config.initialize_from_ema,
            adapter_only=config.train_adapter_only,
            initialize_conditioning_from_base=config.initialize_conditioning_from_base,
            expand_speaker_embedding=config.new_speaker_id is not None,
            initialize_context_from_base=config.initialize_context_from_base,
        )
        ema = ExponentialMovingAverage(raw_model, config.ema_decay)
    train_arrow_sha256 = _sha256(Path(config.train_arrow))
    model: Any = raw_model
    if config.compile_model:
        model = torch.compile(model)
    if distributed_run:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    frontend = LogMelFrontend(config, str(device))
    mel_normalizer = MelNormalizer(config, str(device))
    speaker_scorer: SpeakerConsistencyScorer | None = None
    if config.speaker_consistency_weight > 0.0:
        assert config.speaker_consistency_reference_manifest is not None
        assert config.vocoder_path is not None
        speaker_scorer = SpeakerConsistencyScorer(
            reference_manifest=Path(config.speaker_consistency_reference_manifest),
            vocoder_dir=Path(config.vocoder_path),
            device=str(device),
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
        )
    log_path = output_dir / "metrics.jsonl"
    if rank == 0:
        metadata = {
            "run_version": config.run_version,
            "parameters": parameter_count(raw_model),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
            "world_size": world_size,
            "train_records": len(train_dataset),
            "validation_records": len(validation_dataset),
            "train_reference_records": sum(record.reference_audio_path is not None for record in train_dataset.records),
            "validation_reference_records": sum(
                record.reference_audio_path is not None for record in validation_dataset.records
            ),
            "train_batches_per_rank": len(train_sampler),
            "model_config": model_config.as_dict(),
            "train_config": config.as_dict(),
            "vocab_sha256": _sha256(vocabulary_path),
            "speaker_vocab_sha256": (_sha256(speaker_vocabulary_path) if speaker_vocabulary is not None else None),
            "train_arrow_sha256": train_arrow_sha256,
        }
        (output_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(metadata, ensure_ascii=False), flush=True)

    def all_ranks_finite(value: Any) -> bool:
        finite = torch.isfinite(value).all().to(device=device, dtype=torch.int32)
        if distributed_run:
            distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
        return bool(finite.item())

    def abort_run(status: str, metric: str, observed: float, failure_update: int) -> None:
        if rank == 0:
            failure = {
                "status": status,
                "metric": metric,
                "observed_on_rank_zero": observed,
                "update": failure_update,
                "epoch": epoch,
                "run_version": config.run_version,
            }
            temporary = output_dir / "failure.json.tmp"
            temporary.write_text(json.dumps(failure, indent=2) + "\n")
            temporary.replace(output_dir / "failure.json")
            print(json.dumps(failure), flush=True)
        if distributed_run:
            distributed.barrier()
            distributed.destroy_process_group()
        raise RuntimeError(f"{status}: {metric} at update {failure_update}")

    def model_parameters_are_finite() -> bool:
        finite = torch.ones((), device=device, dtype=torch.bool)
        for parameter in raw_model.parameters():
            finite = finite & torch.isfinite(parameter.detach()).all()
        return all_ranks_finite(finite)

    def total_gradient_norm() -> Any:
        gradients = [parameter.grad for parameter in raw_model.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("model produced no gradients")
        per_parameter = torch.stack([torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients])
        return torch.linalg.vector_norm(per_parameter)

    def clip_gradients(gradient_norm: Any) -> None:
        coefficient = (config.gradient_clip / (gradient_norm + 1e-6)).clamp(max=1.0)
        for parameter in raw_model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)

    def prepare_batch(batch: dict[str, Any], *, training: bool) -> tuple[Any, ...]:
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        sample_lengths = batch["sample_lengths"].to(device, non_blocking=True)
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        token_lengths = batch["token_lengths"].to(device, non_blocking=True)
        speaker_ids = batch["speaker_ids"].to(device, non_blocking=True)
        prosody_features = batch["prosody_features"].to(device, non_blocking=True)
        mel, frame_lengths = frontend(waveforms, sample_lengths)
        mel = mel_normalizer.normalize(mel)
        duration_frame_lengths = frame_lengths
        context_frame_mask = None
        if config.paired_reference_conditioning:
            has_references = batch["has_references"].to(device, non_blocking=True)
            reference_token_ids = batch["reference_token_ids"].to(device, non_blocking=True)
            reference_token_lengths = batch["reference_token_lengths"].to(device, non_blocking=True)
            if bool(has_references.any()):
                reference_waveforms = batch["reference_waveforms"].to(device, non_blocking=True)
                reference_sample_lengths = batch["reference_sample_lengths"].to(device, non_blocking=True)
                reference_mel, reference_frame_lengths = frontend(
                    reference_waveforms,
                    reference_sample_lengths,
                )
                reference_mel = mel_normalizer.normalize(reference_mel)
            else:
                reference_mel = mel.new_zeros((mel.shape[0], 1, mel.shape[2]))
                reference_frame_lengths = torch.zeros_like(frame_lengths)
            (
                mel,
                frame_lengths,
                token_ids,
                token_lengths,
                context_frame_mask,
            ) = _prepend_paired_references(
                mel,
                frame_lengths,
                token_ids,
                token_lengths,
                reference_mel,
                reference_frame_lengths,
                reference_token_ids,
                reference_token_lengths,
                has_references,
            )
        if training and config.text_dropout_probability > 0.0:
            dropped_text = torch.rand(token_ids.shape[0], device=device) < config.text_dropout_probability
            if bool(dropped_text.any()):
                token_ids[dropped_text] = 0
                token_ids[dropped_text, 0] = 1
                token_lengths[dropped_text] = 1
        if training and config.speaker_dropout_probability > 0.0:
            dropped_speaker = torch.rand(speaker_ids.shape[0], device=device) < config.speaker_dropout_probability
            speaker_ids = speaker_ids * ~dropped_speaker
        frame_mask, token_mask = _make_masks(frame_lengths, token_lengths, mel.shape[1], token_ids.shape[1])
        if training and config.infill_probability > 0.0:
            if context_frame_mask is None:
                context_frame_mask = torch.zeros_like(frame_mask)
            for row in range(frame_mask.shape[0]):
                reference_frames = int(context_frame_mask[row].sum().item())
                target_frames = int(frame_lengths[row].item()) - reference_frames
                if target_frames < 16 or random.random() >= config.infill_probability:
                    continue
                fraction = random.uniform(
                    config.infill_context_min_fraction,
                    config.infill_context_max_fraction,
                )
                span = max(4, int(target_frames * fraction))
                if random.random() < config.infill_prefix_probability:
                    start = reference_frames
                else:
                    start = reference_frames + random.randint(0, target_frames - span)
                context_frame_mask[row, start : start + span] = True
            # Preserve an all-false tensor: every DDP rank must execute the context branch
            # even when its local batch happens not to select an infill span.
        return (
            mel,
            token_ids,
            frame_mask,
            token_mask,
            speaker_ids,
            prosody_features,
            duration_frame_lengths,
            context_frame_mask,
        )

    def validation_loss() -> float:
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader):
                if batch_index >= config.validation_batches:
                    break
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    (
                        mel,
                        token_ids,
                        frame_mask,
                        token_mask,
                        speaker_ids,
                        prosody_features,
                        duration_frame_lengths,
                        context_frame_mask,
                    ) = prepare_batch(batch, training=False)
                    loss, _ = flow_matching_loss(
                        model,
                        mel,
                        token_ids,
                        frame_mask,
                        token_mask,
                        speaker_ids=speaker_ids,
                        prosody_features=prosody_features,
                        context_frame_mask=context_frame_mask,
                        duration_frame_lengths=duration_frame_lengths,
                        duration_weight=config.duration_weight,
                        foundation_preservation_weight=config.foundation_preservation_weight,
                    )
                losses.append(float(loss.item()))
        model.train()
        loss_sum = sum(losses)
        loss_count = len(losses)
        if distributed_run:
            totals = torch.tensor([loss_sum, loss_count], device=device, dtype=torch.float64)
            distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
            loss_sum, loss_count = float(totals[0].item()), int(totals[1].item())
        if loss_count == 0:
            raise RuntimeError("validation loader produced no batches")
        return loss_sum / loss_count

    validation_state_path = output_dir / "validation_state.json"
    if resume_path is not None and validation_state_path.is_file():
        validation_state = json.loads(validation_state_path.read_text())
        best_validation_loss = float(validation_state["best_validation_loss"])
        best_validation_update = int(validation_state["best_validation_update"])
        degradation_count = int(validation_state["degradation_count"])
    else:
        best_validation_loss = math.inf
        best_validation_update = -1
        degradation_count = 0

    starting_update = update
    started = time.monotonic()
    model.train()
    while update < config.max_updates:
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if update >= config.max_updates:
                break
            learning_rate = _learning_rate(config, update)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            # Consistency scoring runs the frozen vocoder + WavLM through the
            # prediction graph, so it is scheduled sparsely to amortize memory.
            consistency_requested = (
                speaker_scorer is not None and update % config.speaker_consistency_every_updates == 0
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                (
                    mel,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids,
                    prosody_features,
                    duration_frame_lengths,
                    context_frame_mask,
                ) = prepare_batch(batch, training=True)
                loss, components = flow_matching_loss(
                    model,
                    mel,
                    token_ids,
                    frame_mask,
                    token_mask,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_features,
                    context_frame_mask=context_frame_mask,
                    duration_frame_lengths=duration_frame_lengths,
                    duration_weight=config.duration_weight,
                    foundation_preservation_weight=config.foundation_preservation_weight,
                    return_predictions=consistency_requested,
                )
            if consistency_requested and "predicted_mel" in components:
                assert speaker_scorer is not None
                consistency_loss = speaker_scorer.loss(
                    components["predicted_mel"],
                    components["scored_frame_mask"],
                    mel_normalizer,
                    max_seconds=config.speaker_consistency_max_seconds,
                )
                if consistency_loss is not None:
                    loss = loss + config.speaker_consistency_weight * consistency_loss
                    components["speaker_consistency"] = consistency_loss.detach()
            if not all_ranks_finite(loss.detach()):
                abort_run("aborted_nonfinite", "loss", float(loss.detach().item()), update)
            torch.autograd.backward(loss)
            gradient_norm = total_gradient_norm()
            if not all_ranks_finite(gradient_norm):
                abort_run("aborted_nonfinite", "gradient_norm", float(gradient_norm.item()), update)
            clip_gradients(gradient_norm)
            optimizer.step()
            update += 1
            if update % config.log_every_updates == 0 and not model_parameters_are_finite():
                abort_run("aborted_nonfinite", "model_parameters", float("nan"), update)
            ema.update(raw_model)

            if update % config.log_every_updates == 0 and rank == 0:
                elapsed = time.monotonic() - started
                metrics = {
                    "update": update,
                    "epoch": epoch,
                    "loss": float(loss.detach().item()),
                    "flow_loss": float(components["flow"].item()),
                    "duration_loss": float(components["duration"].item()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": learning_rate,
                    "updates_per_second": (update - starting_update) / max(elapsed, 1e-6),
                }
                if "foundation_preservation" in components:
                    metrics["foundation_preservation_loss"] = float(components["foundation_preservation"].item())
                if "speaker_consistency" in components:
                    metrics["speaker_consistency_loss"] = float(components["speaker_consistency"].item())
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)

            should_validate = update % config.validate_every_updates == 0
            if should_validate:
                value = validation_loss()
                if not math.isfinite(value):
                    abort_run("aborted_nonfinite", "validation_loss", value, update)
                if value < best_validation_loss:
                    best_validation_loss = value
                    best_validation_update = update
                    degradation_count = 0
                elif value > best_validation_loss * config.validation_degradation_factor:
                    degradation_count += 1
                else:
                    degradation_count = 0
                validation_state = {
                    "best_validation_loss": best_validation_loss,
                    "best_validation_update": best_validation_update,
                    "degradation_count": degradation_count,
                    "last_validation_loss": value,
                    "last_validation_update": update,
                }
                if rank == 0:
                    validation_state_path.write_text(json.dumps(validation_state, indent=2) + "\n")
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"update": update, "validation_loss": value}) + "\n")
                    print(json.dumps({"update": update, "validation_loss": value}), flush=True)
                if degradation_count >= config.validation_degradation_patience:
                    abort_run("aborted_validation_degradation", "validation_loss", value, update)

            should_save = _should_save_checkpoint(
                update=update,
                max_updates=config.max_updates,
                save_every_updates=config.save_every_updates,
                best_validation_update=best_validation_update,
            )
            if should_save:
                if distributed_run:
                    distributed.barrier()
                if rank == 0:
                    payload = _checkpoint_payload(
                        model=raw_model,
                        ema=ema,
                        optimizer=optimizer,
                        config=config,
                        model_config=model_config,
                        tokenizer=tokenizer,
                        speaker_vocabulary=speaker_vocabulary,
                        update=update,
                        epoch=epoch,
                        train_arrow_sha256=train_arrow_sha256,
                    )
                    checkpoint = output_dir / f"model_{update}.pt"
                    temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
                    torch.save(payload, temporary_checkpoint)
                    temporary_checkpoint.replace(checkpoint)
                    latest_checkpoint = output_dir / "model_last.pt"
                    latest_checkpoint.unlink(missing_ok=True)
                    os.link(checkpoint, latest_checkpoint)
                    print(f"saved checkpoint {checkpoint}", flush=True)
                    if update == best_validation_update:
                        best_checkpoint = output_dir / "model_best.pt"
                        best_checkpoint.unlink(missing_ok=True)
                        os.link(checkpoint, best_checkpoint)
                if distributed_run:
                    distributed.barrier()
        epoch += 1

    if distributed_run:
        distributed.destroy_process_group()
    return output_dir / "model_last.pt"


def load_crossflow_checkpoint(
    checkpoint_path: Path,
    device: str,
    *,
    use_ema: bool = True,
) -> tuple[Any, CharacterTokenizer, dict[str, Any]]:
    """Load a training checkpoint (``.pt``), a release directory, or a Hugging Face repo id."""
    import torch

    from turkish_tts.crossflow import CrossFlow, CrossFlowModelConfig
    from turkish_tts.crossflow_release import is_release_directory, load_release_payload, resolve_release_path

    if not checkpoint_path.is_file():
        release_dir = resolve_release_path(str(checkpoint_path))
        if not is_release_directory(release_dir):
            raise FileNotFoundError(f"not a checkpoint file or release directory: {checkpoint_path}")
        model, payload = load_release_payload(release_dir, device)
        return model, CharacterTokenizer(payload["vocabulary"]), payload
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CrossFlow(CrossFlowModelConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    if use_ema and "values" in payload["ema"]:
        parameters = dict(model.named_parameters())
        for name, value in payload["ema"]["values"].items():
            parameters[name].data.copy_(value)
    model.to(device).eval()
    return model, CharacterTokenizer(payload["vocabulary"]), payload


def _load_crossflow_vocoder(vocoder_dir: Path, device: str) -> Any:
    """Load BigVGAN from a local directory or, when the path does not exist, a Hub repo id."""
    try:
        from bigvgan import BigVGAN  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("CrossFlow synthesis requires the pinned BigVGAN source tree") from error
    vocoder = BigVGAN.from_pretrained(str(vocoder_dir), use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    return vocoder.eval().to(device)


def _normalize_crossflow_text(text: str, text_normalization: str) -> str:
    if text_normalization == "turkish":
        return turkish_lower(normalize_for_model(text))
    if text_normalization == "english":
        try:
            normalize_english = cast(
                Callable[[str], str],
                import_module("alania.normalize").normalize_for_model,
            )
        except ImportError as error:
            raise RuntimeError(
                "English synthesis requires the Alania package; install it or add alania/src to PYTHONPATH"
            ) from error
        return normalize_english(text).lower()
    raise ValueError(f"unsupported text normalization: {text_normalization}")


def validate_crossflow_synthesis_text(
    text: str,
    tokenizer: CharacterTokenizer,
    *,
    max_text_tokens: int,
    text_normalization: str = "turkish",
) -> tuple[str, list[int]]:
    if any(not character.isprintable() for character in text):
        raise ValueError("synthesis text contains control characters")
    normalized = _normalize_crossflow_text(text, text_normalization)
    if not normalized:
        raise ValueError("synthesis text is empty after normalization")
    # The vocabulary is built from training text, so <unk> never received a gradient: feeding it
    # produces an arbitrary sound. Dropping the character is the lesser harm.
    unknown = tokenizer.symbol_to_id["<unk>"]
    tokens = tokenizer.encode(normalized)
    normalized = "".join(
        character for character, token in zip(normalized, tokens, strict=True) if token != unknown
    ).strip()
    if not normalized:
        raise ValueError("synthesis text contains no characters the model knows")
    encoded = tokenizer.encode(normalized)
    if len(encoded) > max_text_tokens:
        raise ValueError(f"synthesis text has {len(encoded)} tokens; maximum is {max_text_tokens}")
    return normalized, encoded


def _resolve_speaker_id(payload: dict[str, Any], speaker: str | None) -> int:
    if speaker is None:
        return 0
    vocabulary = payload.get("speaker_vocabulary")
    if not isinstance(vocabulary, (list, tuple)) or speaker not in vocabulary:
        raise ValueError(f"speaker is not present in checkpoint vocabulary: {speaker}")
    return vocabulary.index(speaker)


def _prepare_reference(
    *,
    reference_audio: Path,
    reference_text: str,
    train_config: CrossFlowTrainConfig,
    tokenizer: CharacterTokenizer,
    mel_normalizer: MelNormalizer,
    max_text_tokens: int,
    device: str,
) -> tuple[Any, str]:
    import torch
    import torchaudio.functional as audio_functional

    audio, sample_rate = sf.read(reference_audio, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.mean(audio, axis=1, dtype=np.float32))
    if sample_rate != train_config.sample_rate:
        waveform = audio_functional.resample(waveform, sample_rate, train_config.sample_rate)
    frontend = LogMelFrontend(train_config, device)
    frame_budget = math.ceil(train_config.max_audio_seconds * train_config.sample_rate / train_config.hop_length)
    max_reference_frames = frame_budget // 2
    max_reference_samples = max_reference_frames * train_config.hop_length
    if waveform.numel() > max_reference_samples:
        # Keep the head of the enrollment clip; references only need to anchor
        # timbre, and the model frame budget must leave room for the target.
        waveform = waveform[:max_reference_samples]
    lengths = torch.tensor([waveform.numel()], dtype=torch.long, device=device)
    with torch.inference_mode():
        mel, _ = frontend(waveform.unsqueeze(0).to(device), lengths)
        mel = mel_normalizer.normalize(mel)
    normalized_reference, _ = validate_crossflow_synthesis_text(
        reference_text,
        tokenizer,
        max_text_tokens=max_text_tokens,
        text_normalization=train_config.text_normalization,
    )
    return mel, normalized_reference


PROSODY_FEATURE_ORDER = (
    "log_seconds_per_character",
    "log_energy_mean",
    "log_energy_std",
    "log_f0_mean",
    "log_f0_std",
    "voiced_ratio",
)
STYLE_QUESTION_PRESET = "questions_confirmations"
STYLE_EMPATHY_PRESET = "restrained_emotion"
STYLE_EMPATHY_KEYWORDS = ("üzgün", "özür", "maalesef", "anlıyor", "endişe", "sabrınız")  # noqa: RUF001


def load_prosody_presets(path: Path) -> dict[str, Any]:
    """Load and validate a Candidate prosody preset payload."""
    payload = json.loads(path.read_text())
    if tuple(payload.get("feature_order", ())) != PROSODY_FEATURE_ORDER:
        raise ValueError(f"preset feature order does not match the model contract: {path}")
    global_mean = payload.get("global_mean")
    if not isinstance(global_mean, list) or len(global_mean) != len(PROSODY_FEATURE_ORDER):
        raise ValueError(f"preset payload is missing a valid global_mean: {path}")
    presets = payload.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError(f"preset payload contains no presets: {path}")
    for name, preset in presets.items():
        vector = preset.get("prosody")
        if not isinstance(vector, list) or len(vector) != len(PROSODY_FEATURE_ORDER):
            raise ValueError(f"preset {name!r} has an invalid prosody vector: {path}")
    return cast(dict[str, Any], payload)


def blend_prosody_preset(payload: dict[str, Any], name: str, strength: float) -> list[float]:
    """Interpolate from the speaker's global mean toward a named preset centroid."""
    if name not in payload["presets"]:
        raise ValueError(f"unknown prosody preset: {name}; available: {sorted(payload['presets'])}")
    if not 0.0 <= strength <= 1.5:
        raise ValueError("preset strength must be between zero and one and a half")
    global_mean = payload["global_mean"]
    target = payload["presets"][name]["prosody"]
    return [
        max(-5.0, min(5.0, base + strength * (value - base))) for base, value in zip(global_mean, target, strict=True)
    ]


def select_chunk_style(chunk: str) -> str | None:
    """Pick a dialog-act preset for one clause based on surface cues."""
    lowered = turkish_lower(chunk)
    if "?" in chunk:
        return STYLE_QUESTION_PRESET
    if any(keyword in lowered for keyword in STYLE_EMPATHY_KEYWORDS):
        return STYLE_EMPATHY_PRESET
    return None


def _resolve_chunk_prosody(
    chunk: str,
    base_prosody: Sequence[float] | None,
    auto_style: dict[str, Any] | None,
) -> Sequence[float] | None:
    if auto_style is None:
        return base_prosody
    style = select_chunk_style(chunk)
    if style is None or style not in auto_style["payload"]["presets"]:
        return base_prosody
    return blend_prosody_preset(auto_style["payload"], style, float(auto_style["strength"]))


def _chunk_text(text: str, limit: int, text_normalization: str = "turkish") -> list[str]:
    """Split text into clause-sized chunks no longer than `limit` characters where possible.

    Normalizes first: splitting raw text would break at the dot in "Dr. Kaya" and insert a
    pause mid-phrase, whereas the expanded "doktor kaya" has no dot to split on.
    """
    stripped = _normalize_crossflow_text(text.strip(), text_normalization)
    if limit <= 0 or len(stripped) <= limit:
        return [stripped]
    sentences = re.split(r"(?<=[.!?;:])\s+", stripped)
    parts: list[str] = []
    for sentence in sentences:
        if len(sentence) > limit:
            parts.extend(re.split(r"(?<=,)\s+", sentence))
        else:
            parts.append(sentence)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if current and len(candidate) > limit:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _synthesize_loaded_crossflow(
    *,
    model: Any,
    tokenizer: CharacterTokenizer,
    vocoder: Any,
    mel_normalizer: MelNormalizer,
    text: str,
    sample_rate: int,
    hop_length: int,
    device: str,
    steps: int,
    seed: int,
    duration_scale: float,
    speaker_id: int,
    prosody: Sequence[float] | None,
    text_guidance_scale: float,
    speaker_guidance_scale: float,
    sway_coefficient: float,
    solver: str,
    guidance_rescale: float,
    mel_clamp: float | None,
    reference: tuple[Any, str] | None,
    context_guidance_scale: float,
    min_seconds_per_char: float,
    chunk_character_limit: int,
    chunk_pause_seconds: float,
    auto_style: dict[str, Any] | None = None,
    mel_correction: Sequence[float] | None = None,
    text_normalization: str = "turkish",
) -> tuple[Any, str, float]:
    import torch

    if reference is not None and chunk_character_limit > 0:
        raise ValueError("chunked synthesis does not support reference conditioning")

    def prosody_tensor(chunk: str) -> Any:
        chunk_prosody = _resolve_chunk_prosody(chunk, prosody, auto_style)
        if chunk_prosody is None:
            return None
        prosody_values = list(chunk_prosody)
        if len(prosody_values) != model.config.prosody_dim:
            raise ValueError(f"prosody must contain {model.config.prosody_dim} values; got {len(prosody_values)}")
        return torch.tensor([prosody_values], dtype=torch.float32, device=device)

    def render(chunk: str, chunk_seed: int) -> tuple[Any, str, float]:
        normalized, encoded = validate_crossflow_synthesis_text(
            chunk,
            tokenizer,
            max_text_tokens=model.config.max_text_tokens,
            text_normalization=text_normalization,
        )
        reference_frames = 0
        reference_mel = None
        if reference is not None:
            if model.config.context_conditioning is False:
                raise ValueError("reference synthesis requires a context-conditioned checkpoint")
            reference_mel, reference_text_normalized = reference
            reference_frames = reference_mel.shape[1]
            combined = tokenizer.encode(f"{reference_text_normalized} {normalized}")
            if len(combined) > model.config.max_text_tokens:
                raise ValueError("reference plus synthesis text exceeds the token budget")
            token_ids = torch.tensor([combined], dtype=torch.long, device=device)
        else:
            token_ids = torch.tensor([encoded], dtype=torch.long, device=device)
        token_mask = token_ids.ne(0)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ),
        ):
            frame_count = None
            target_ids = torch.tensor([encoded], dtype=torch.long, device=device)
            _, target_log_frames = model.encode_text(target_ids, target_ids.ne(0))
            predicted = round(math.exp(float(target_log_frames.item())) * duration_scale)
            if reference is not None:
                frame_count = reference_frames + max(8, predicted)
            elif min_seconds_per_char > 0.0:
                characters = sum(not character.isspace() for character in normalized)
                floor = math.ceil(characters * min_seconds_per_char * sample_rate / hop_length)
                frame_count = min(max(predicted, floor), model.config.max_frames)
            mel, predicted_log_frames = model.sample(
                token_ids,
                token_mask,
                steps=steps,
                seed=chunk_seed,
                duration_scale=duration_scale,
                frame_count=frame_count,
                context_mel=reference_mel,
                speaker_id=speaker_id,
                prosody_features=prosody_tensor(chunk),
                text_guidance_scale=text_guidance_scale,
                speaker_guidance_scale=speaker_guidance_scale,
                sway_coefficient=sway_coefficient,
                solver=solver,
                guidance_rescale=guidance_rescale,
                mel_clamp=mel_clamp,
                context_guidance_scale=context_guidance_scale if reference is not None else 1.0,
            )
        if reference is not None:
            mel = mel[:, reference_frames:]
        mel = mel_normalizer.denormalize(mel)
        if mel_correction is not None:
            correction = torch.tensor(list(mel_correction), device=mel.device, dtype=mel.dtype)
            if int(correction.numel()) != int(mel.shape[-1]):
                raise ValueError(f"mel correction has {correction.numel()} bands; model produces {mel.shape[-1]}")
            mel = mel + correction
        with torch.inference_mode():
            waveform = vocoder(mel.transpose(1, 2).float()).squeeze().detach().cpu().numpy()
        if waveform.size < sample_rate // 20:
            raise RuntimeError(f"synthesized waveform is implausibly short: {waveform.size} samples")
        return waveform, normalized, math.exp(float(predicted_log_frames.item()))

    chunks = _chunk_text(text, chunk_character_limit, text_normalization)
    if len(chunks) == 1:
        return render(chunks[0], seed)
    pause = np.zeros(int(chunk_pause_seconds * sample_rate), dtype=np.float32)
    waveforms: list[Any] = []
    normalized_chunks: list[str] = []
    predicted_total = 0.0
    for chunk_index, chunk in enumerate(chunks):
        waveform, normalized, predicted_frames = render(chunk, seed + chunk_index * 100_000)
        if chunk_index > 0:
            waveforms.append(pause)
        waveforms.append(waveform.astype(np.float32))
        normalized_chunks.append(normalized)
        predicted_total += predicted_frames
    return np.concatenate(waveforms), " ".join(normalized_chunks), predicted_total


def _synthesize_loaded_crossflow_batch(
    *,
    model: Any,
    tokenizer: CharacterTokenizer,
    vocoder: Any,
    mel_normalizer: MelNormalizer,
    texts: Sequence[str],
    sample_rate: int,
    hop_length: int,
    device: str,
    steps: int,
    seeds: Sequence[int],
    duration_scale: float,
    speaker_id: int,
    prosodies: Sequence[Sequence[float]],
    text_guidance_scale: float,
    speaker_guidance_scale: float,
    sway_coefficient: float,
    solver: str,
    guidance_rescale: float,
    mel_clamp: float | None,
    min_seconds_per_char: float,
    auto_styles: Sequence[dict[str, Any] | None],
    mel_correction: Sequence[float] | None = None,
    text_normalization: str = "turkish",
    reference: tuple[Any, str] | None = None,
    context_guidance_scale: float = 1.0,
) -> list[tuple[Any, str, float]]:
    """Render unrelated requests in one padded flow and vocoder batch.

    With ``reference``, every row is conditioned on the same fixed enrollment
    context (production single-speaker anchoring). Reference audio and text are
    pre-encoded by the caller via ``_prepare_reference``.
    """
    import torch

    if reference is not None and model.config.context_conditioning is False:
        raise ValueError("reference synthesis requires a context-conditioned checkpoint")

    batch_size = len(texts)
    if batch_size < 1:
        raise ValueError("request batch requires at least one text")
    if not (len(seeds) == len(prosodies) == len(auto_styles) == batch_size):
        raise ValueError("texts, seeds, prosodies, and auto styles must have equal lengths")

    normalized_texts: list[str] = []
    token_rows = []
    prosody_rows: list[list[float]] = []
    for text, prosody, auto_style in zip(texts, prosodies, auto_styles, strict=True):
        normalized, encoded = validate_crossflow_synthesis_text(
            text,
            tokenizer,
            max_text_tokens=model.config.max_text_tokens,
            text_normalization=text_normalization,
        )
        normalized_texts.append(normalized)
        token_rows.append(torch.tensor(encoded, dtype=torch.long, device=device))
        resolved_prosody = _resolve_chunk_prosody(normalized, prosody, auto_style)
        if resolved_prosody is None or len(resolved_prosody) != model.config.prosody_dim:
            actual = 0 if resolved_prosody is None else len(resolved_prosody)
            raise ValueError(f"prosody must contain {model.config.prosody_dim} values; got {actual}")
        prosody_rows.append(list(resolved_prosody))

    token_ids = torch.nn.utils.rnn.pad_sequence(token_rows, batch_first=True, padding_value=0)
    token_mask = token_ids.ne(0)
    speaker_ids = torch.full((batch_size,), speaker_id, dtype=torch.long, device=device)
    prosody_features = torch.tensor(prosody_rows, dtype=torch.float32, device=device)
    context_mel = None
    context_frame_mask = None
    if reference is not None:
        context_mel, _reference_text = reference
        context_frame_mask = torch.ones(
            context_mel.shape[0],
            context_mel.shape[1],
            device=device,
            dtype=torch.bool,
        )
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ),
    ):
        _, log_frames = model.encode_text(token_ids, token_mask)
        predicted_frames = [
            round(math.exp(float(value)) * duration_scale) for value in log_frames.detach().float().cpu().tolist()
        ]
        frame_counts = []
        for normalized, predicted in zip(normalized_texts, predicted_frames, strict=True):
            frame_count = max(8, min(model.config.max_frames, predicted))
            if min_seconds_per_char > 0.0:
                characters = sum(not character.isspace() for character in normalized)
                floor = math.ceil(characters * min_seconds_per_char * sample_rate / hop_length)
                frame_count = min(max(frame_count, floor), model.config.max_frames)
            frame_counts.append(frame_count)
        mel = model.sample_batched(
            token_ids,
            token_mask,
            frame_counts=frame_counts,
            seeds=list(seeds),
            steps=steps,
            speaker_ids=speaker_ids,
            prosody_features=prosody_features,
            text_guidance_scale=text_guidance_scale,
            speaker_guidance_scale=speaker_guidance_scale,
            sway_coefficient=sway_coefficient,
            solver=solver,
            guidance_rescale=guidance_rescale,
            mel_clamp=mel_clamp,
            context_mel=context_mel,
            context_frame_mask=context_frame_mask,
            context_guidance_scale=context_guidance_scale,
        )
    mel = mel_normalizer.denormalize(mel)
    if mel_correction is not None:
        correction = torch.tensor(list(mel_correction), device=mel.device, dtype=mel.dtype)
        if int(correction.numel()) != int(mel.shape[-1]):
            raise ValueError(f"mel correction has {correction.numel()} bands; model produces {mel.shape[-1]}")
        mel = mel + correction
    with torch.inference_mode():
        audio = vocoder(mel.transpose(1, 2).float()).squeeze(1).detach().cpu().numpy()

    results = []
    for row, (normalized, predicted, frame_count) in enumerate(
        zip(normalized_texts, predicted_frames, frame_counts, strict=True)
    ):
        waveform = audio[row, : frame_count * hop_length]
        if waveform.size < sample_rate // 20:
            raise RuntimeError(f"synthesized waveform is implausibly short: {waveform.size} samples")
        results.append((waveform, normalized, float(predicted)))
    return results


def _synthesize_loaded_crossflow_candidates(
    *,
    model: Any,
    tokenizer: CharacterTokenizer,
    vocoder: Any,
    mel_normalizer: MelNormalizer,
    text: str,
    sample_rate: int,
    hop_length: int,
    device: str,
    steps: int,
    seeds: Sequence[int],
    duration_scale: float,
    speaker_id: int,
    prosody: Sequence[float] | None,
    text_guidance_scale: float,
    speaker_guidance_scale: float,
    sway_coefficient: float,
    solver: str,
    guidance_rescale: float,
    mel_clamp: float | None,
    min_seconds_per_char: float,
    chunk_character_limit: int,
    chunk_pause_seconds: float,
    auto_style: dict[str, Any] | None = None,
    mel_correction: Sequence[float] | None = None,
    text_normalization: str = "turkish",
    prune_keep: int = 0,
    prune_after_step: int = 8,
    prune_scorer: Any | None = None,
) -> list[tuple[Any, str, float]]:
    """Render one take per seed in a single batched flow pass per chunk.

    With ``prune_keep`` in (0, len(seeds)), the first chunk runs mid-flow best-of-N pruning:
    after ``prune_after_step`` flow steps the weakest candidates are dropped and only the
    surviving seeds are rendered for this and every later chunk, so one utterance keeps one
    seed identity throughout. Results then contain ``prune_keep`` entries, in original seed
    order. ``prune_scorer`` defaults to the deterministic mel-statistics scorer.
    """
    import torch

    candidate_count = len(seeds)
    if candidate_count < 1:
        raise ValueError("candidate synthesis requires at least one seed")
    if prune_keep < 0 or prune_keep > candidate_count:
        raise ValueError("prune_keep must be between zero and the seed count")
    pruning_enabled = 0 < prune_keep < candidate_count
    if pruning_enabled and prune_scorer is None:
        from turkish_tts.quality import MelStatisticsScorer

        prune_scorer = MelStatisticsScorer()

    def prosody_tensor(chunk: str, count: int) -> Any:
        chunk_prosody = _resolve_chunk_prosody(chunk, prosody, auto_style)
        if chunk_prosody is None:
            return None
        prosody_values = list(chunk_prosody)
        if len(prosody_values) != model.config.prosody_dim:
            raise ValueError(f"prosody must contain {model.config.prosody_dim} values; got {len(prosody_values)}")
        return torch.tensor([prosody_values] * count, dtype=torch.float32, device=device)

    def render_batch(
        chunk: str,
        chunk_seeds: Sequence[int],
        *,
        prune_now: bool,
    ) -> tuple[list[Any], str, float, list[int]]:
        normalized, encoded = validate_crossflow_synthesis_text(
            chunk,
            tokenizer,
            max_text_tokens=model.config.max_text_tokens,
            text_normalization=text_normalization,
        )
        count = len(chunk_seeds)
        token_row = torch.tensor([encoded], dtype=torch.long, device=device)
        token_ids = token_row.expand(count, -1)
        token_mask = token_ids.ne(0)
        speaker_ids = torch.full((count,), speaker_id, dtype=torch.long, device=device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ),
        ):
            _, log_frames = model.encode_text(token_row, token_row.ne(0))
            predicted = round(math.exp(float(log_frames.item())) * duration_scale)
            frame_count = max(8, min(model.config.max_frames, predicted))
            if min_seconds_per_char > 0.0:
                characters = sum(not character.isspace() for character in normalized)
                floor = math.ceil(characters * min_seconds_per_char * sample_rate / hop_length)
                frame_count = min(max(frame_count, floor), model.config.max_frames)
            kept = list(range(count))
            if prune_now:
                mel, kept = model.sample_batched_pruned(
                    token_ids,
                    token_mask,
                    frame_counts=[frame_count] * count,
                    seeds=list(chunk_seeds),
                    prune_scorer=prune_scorer,
                    prune_after_step=prune_after_step,
                    prune_keep=prune_keep,
                    steps=steps,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_tensor(chunk, count),
                    text_guidance_scale=text_guidance_scale,
                    speaker_guidance_scale=speaker_guidance_scale,
                    sway_coefficient=sway_coefficient,
                    solver=solver,
                    guidance_rescale=guidance_rescale,
                    mel_clamp=mel_clamp,
                )
            else:
                mel = model.sample_batched(
                    token_ids,
                    token_mask,
                    frame_counts=[frame_count] * count,
                    seeds=list(chunk_seeds),
                    steps=steps,
                    speaker_ids=speaker_ids,
                    prosody_features=prosody_tensor(chunk, count),
                    text_guidance_scale=text_guidance_scale,
                    speaker_guidance_scale=speaker_guidance_scale,
                    sway_coefficient=sway_coefficient,
                    solver=solver,
                    guidance_rescale=guidance_rescale,
                    mel_clamp=mel_clamp,
                )
        mel = mel_normalizer.denormalize(mel)
        if mel_correction is not None:
            correction = torch.tensor(list(mel_correction), device=mel.device, dtype=mel.dtype)
            if int(correction.numel()) != int(mel.shape[-1]):
                raise ValueError(f"mel correction has {correction.numel()} bands; model produces {mel.shape[-1]}")
            mel = mel + correction
        with torch.inference_mode():
            audio = vocoder(mel.transpose(1, 2).float()).squeeze(1).detach().cpu().numpy()
        waveforms = []
        for row in range(mel.shape[0]):
            waveform = audio[row]
            if waveform.size < sample_rate // 20:
                raise RuntimeError(f"synthesized waveform is implausibly short: {waveform.size} samples")
            waveforms.append(waveform)
        return waveforms, normalized, float(predicted), kept

    chunks = _chunk_text(text, chunk_character_limit, text_normalization)
    pause = np.zeros(int(chunk_pause_seconds * sample_rate), dtype=np.float32)
    active_seeds = list(seeds)
    per_candidate_parts: list[list[Any]] = []
    normalized_chunks: list[str] = []
    predicted_total = 0.0
    for chunk_index, chunk in enumerate(chunks):
        chunk_seeds = [candidate_seed + chunk_index * 100_000 for candidate_seed in active_seeds]
        prune_now = pruning_enabled and chunk_index == 0
        waveforms, normalized, predicted_frames, kept = render_batch(chunk, chunk_seeds, prune_now=prune_now)
        if prune_now:
            active_seeds = [active_seeds[index] for index in kept]
        if not per_candidate_parts:
            per_candidate_parts = [[] for _ in waveforms]
        for candidate, waveform in enumerate(waveforms):
            if chunk_index > 0:
                per_candidate_parts[candidate].append(pause)
            per_candidate_parts[candidate].append(waveform.astype(np.float32))
        normalized_chunks.append(normalized)
        predicted_total += predicted_frames
    joined_text = " ".join(normalized_chunks)
    return [(np.concatenate(parts), joined_text, predicted_total) for parts in per_candidate_parts]


def synthesize_crossflow(
    *,
    checkpoint_path: Path,
    text: str,
    output_path: Path,
    vocoder_dir: Path,
    device: str = "cuda",
    steps: int = 32,
    seed: int = 20260803,
    duration_scale: float = 1.0,
    adapter_scale: float = 1.0,
    speaker: str | None = None,
    prosody: Sequence[float] | None = None,
    text_guidance_scale: float = 1.0,
    speaker_guidance_scale: float = 1.0,
    sway_coefficient: float = 0.0,
    solver: str = "euler",
    guidance_rescale: float = 0.0,
    mel_clamp: float | None = None,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
    context_guidance_scale: float = 1.0,
    min_seconds_per_char: float = 0.0,
    chunk_character_limit: int = 0,
    chunk_pause_seconds: float = 0.16,
    auto_style: dict[str, Any] | None = None,
    mel_correction: Sequence[float] | None = None,
    use_ema: bool = True,
) -> dict[str, Any]:
    model, tokenizer, payload = load_crossflow_checkpoint(checkpoint_path, device, use_ema=use_ema)
    speaker_id = _resolve_speaker_id(payload, speaker)
    model.set_adapter_scale(adapter_scale)
    train_config = CrossFlowTrainConfig(**payload["train_config"])
    mel_normalizer = MelNormalizer(train_config, device)
    vocoder = _load_crossflow_vocoder(vocoder_dir, device)
    if (reference_audio is None) != (reference_text is None):
        raise ValueError("reference_audio and reference_text must be provided together")
    reference = None
    if reference_audio is not None and reference_text is not None:
        reference = _prepare_reference(
            reference_audio=reference_audio,
            reference_text=reference_text,
            train_config=train_config,
            tokenizer=tokenizer,
            mel_normalizer=mel_normalizer,
            max_text_tokens=model.config.max_text_tokens,
            device=device,
        )
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
        seed=seed,
        duration_scale=duration_scale,
        speaker_id=speaker_id,
        prosody=prosody,
        text_guidance_scale=text_guidance_scale,
        speaker_guidance_scale=speaker_guidance_scale,
        sway_coefficient=sway_coefficient,
        solver=solver,
        guidance_rescale=guidance_rescale,
        mel_clamp=mel_clamp,
        reference=reference,
        context_guidance_scale=context_guidance_scale,
        min_seconds_per_char=min_seconds_per_char,
        chunk_character_limit=chunk_character_limit,
        chunk_pause_seconds=chunk_pause_seconds,
        auto_style=auto_style,
        mel_correction=mel_correction,
        text_normalization=train_config.text_normalization,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, train_config.sample_rate)
    return {
        "run_version": payload["run_version"],
        "checkpoint": str(checkpoint_path),
        "text": text,
        "normalized_text": normalized,
        "output_path": str(output_path),
        "sample_rate_hz": train_config.sample_rate,
        "audio_seconds": len(waveform) / train_config.sample_rate,
        "predicted_frames": predicted_frames,
        "steps": steps,
        "seed": seed,
        "model_weights": "ema" if use_ema else "raw",
        "adapter_scale": adapter_scale,
        "speaker": speaker,
        "speaker_id": speaker_id,
        "prosody": list(prosody) if prosody is not None else None,
        "text_guidance_scale": text_guidance_scale,
        "speaker_guidance_scale": speaker_guidance_scale,
        "sway_coefficient": sway_coefficient,
        "solver": solver,
        "guidance_rescale": guidance_rescale,
        "mel_clamp": mel_clamp,
        "reference_audio": str(reference_audio) if reference_audio is not None else None,
        "reference_text": reference_text,
        "context_guidance_scale": context_guidance_scale,
        "min_seconds_per_char": min_seconds_per_char,
        "chunk_character_limit": chunk_character_limit,
        "chunk_pause_seconds": chunk_pause_seconds,
        "auto_style": auto_style is not None,
        "mel_correction": mel_correction is not None,
    }


def synthesize_crossflow_suite(
    *,
    checkpoint_path: Path,
    prompts: Sequence[dict[str, str]],
    output_dir: Path,
    vocoder_dir: Path,
    device: str = "cuda",
    steps: int = 32,
    seed: int = 20260803,
    duration_scale: float = 1.0,
    adapter_scale: float = 1.0,
    speaker: str | None = None,
    prosody: Sequence[float] | None = None,
    text_guidance_scale: float = 1.0,
    speaker_guidance_scale: float = 1.0,
    sway_coefficient: float = 0.0,
    solver: str = "euler",
    guidance_rescale: float = 0.0,
    mel_clamp: float | None = None,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
    context_guidance_scale: float = 1.0,
    min_seconds_per_char: float = 0.0,
    chunk_character_limit: int = 0,
    chunk_pause_seconds: float = 0.16,
    auto_style: dict[str, Any] | None = None,
    mel_correction: Sequence[float] | None = None,
    seed_candidates: int = 1,
    use_ema: bool = True,
) -> Path:
    model, tokenizer, payload = load_crossflow_checkpoint(checkpoint_path, device, use_ema=use_ema)
    speaker_id = _resolve_speaker_id(payload, speaker)
    model.set_adapter_scale(adapter_scale)
    train_config = CrossFlowTrainConfig(**payload["train_config"])
    mel_normalizer = MelNormalizer(train_config, device)
    vocoder = _load_crossflow_vocoder(vocoder_dir, device)
    if (reference_audio is None) != (reference_text is None):
        raise ValueError("reference_audio and reference_text must be provided together")
    if seed_candidates < 1:
        raise ValueError("seed_candidates must be at least one")
    if seed_candidates > 1 and reference_audio is not None:
        raise ValueError("candidate batching does not support reference conditioning")
    reference = None
    if reference_audio is not None and reference_text is not None:
        reference = _prepare_reference(
            reference_audio=reference_audio,
            reference_text=reference_text,
            train_config=train_config,
            tokenizer=tokenizer,
            mel_normalizer=mel_normalizer,
            max_text_tokens=model.config.max_text_tokens,
            device=device,
        )
    if seed_candidates > 1:
        base_config = {
            "run_version": f"{payload['run_version']}-checkpoint-{payload['update']}",
            "checkpoint": str(checkpoint_path),
            "checkpoint_update": payload["update"],
            "sample_rate_hz": train_config.sample_rate,
            "steps": steps,
            "seed": seed,
            "model_weights": "ema" if use_ema else "raw",
            "adapter_scale": adapter_scale,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "prosody": list(prosody) if prosody is not None else None,
            "text_guidance_scale": text_guidance_scale,
            "speaker_guidance_scale": speaker_guidance_scale,
            "sway_coefficient": sway_coefficient,
            "solver": solver,
            "guidance_rescale": guidance_rescale,
            "mel_clamp": mel_clamp,
            "min_seconds_per_char": min_seconds_per_char,
            "chunk_character_limit": chunk_character_limit,
            "chunk_pause_seconds": chunk_pause_seconds,
            "seed_candidates": seed_candidates,
        }
        candidate_bases = [seed + candidate * 10_000_019 for candidate in range(seed_candidates)]
        candidate_dirs = [
            output_dir / f"candidate-{candidate:02d}" / "synthesis" for candidate in range(seed_candidates)
        ]
        for directory in candidate_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        candidate_samples: list[list[dict[str, Any]]] = [[] for _ in range(seed_candidates)]
        candidate_audio_seconds = [0.0 for _ in range(seed_candidates)]
        started = time.monotonic()
        for index, prompt in enumerate(prompts, start=1):
            prompt_id = prompt["id"]
            text = prompt["text"]
            prompt_seeds = [base + index - 1 for base in candidate_bases]
            results = _synthesize_loaded_crossflow_candidates(
                model=model,
                tokenizer=tokenizer,
                vocoder=vocoder,
                mel_normalizer=mel_normalizer,
                text=text,
                sample_rate=train_config.sample_rate,
                hop_length=train_config.hop_length,
                device=device,
                steps=steps,
                seeds=prompt_seeds,
                duration_scale=duration_scale,
                speaker_id=speaker_id,
                prosody=prosody,
                text_guidance_scale=text_guidance_scale,
                speaker_guidance_scale=speaker_guidance_scale,
                sway_coefficient=sway_coefficient,
                solver=solver,
                guidance_rescale=guidance_rescale,
                mel_clamp=mel_clamp,
                min_seconds_per_char=min_seconds_per_char,
                chunk_character_limit=chunk_character_limit,
                chunk_pause_seconds=chunk_pause_seconds,
                auto_style=auto_style,
                mel_correction=mel_correction,
                text_normalization=train_config.text_normalization,
            )
            for candidate, (waveform, normalized, predicted_frames) in enumerate(results):
                audio_path = candidate_dirs[candidate] / f"{index:03d}-{prompt_id}.wav"
                sf.write(audio_path, waveform, train_config.sample_rate)
                audio_seconds = len(waveform) / train_config.sample_rate
                candidate_audio_seconds[candidate] += audio_seconds
                candidate_samples[candidate].append(
                    {
                        "id": prompt_id,
                        "category": prompt["category"],
                        "text": text,
                        "normalized_text": normalized,
                        "audio_path": str(audio_path),
                        "audio_seconds": audio_seconds,
                        "predicted_frames": predicted_frames,
                        "seed": prompt_seeds[candidate],
                    }
                )
            print(f"{index:03d}/{len(prompts):03d} {prompt_id} x{seed_candidates}", flush=True)
        elapsed = time.monotonic() - started
        for candidate, directory in enumerate(candidate_dirs):
            candidate_report = {
                **base_config,
                "candidate_index": candidate,
                "seed": candidate_bases[candidate],
                "samples": candidate_samples[candidate],
                "totals": {
                    "prompts": len(candidate_samples[candidate]),
                    "audio_seconds": candidate_audio_seconds[candidate],
                    "generation_seconds": elapsed / seed_candidates,
                    "real_time_factor": (elapsed / seed_candidates) / max(candidate_audio_seconds[candidate], 1e-6),
                },
            }
            (directory / "crossflow.synthesis.json").write_text(
                json.dumps(candidate_report, ensure_ascii=False, indent=2) + "\n"
            )
        manifest = {
            **base_config,
            "candidates": [str(directory.parent) for directory in candidate_dirs],
            "generation_seconds": elapsed,
        }
        manifest_path = output_dir / "crossflow.candidates.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return manifest_path
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    total_audio_seconds = 0.0
    for index, prompt in enumerate(prompts, start=1):
        prompt_id = prompt["id"]
        text = prompt["text"]
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
            seed=seed + index - 1,
            duration_scale=duration_scale,
            speaker_id=speaker_id,
            prosody=prosody,
            text_guidance_scale=text_guidance_scale,
            speaker_guidance_scale=speaker_guidance_scale,
            sway_coefficient=sway_coefficient,
            solver=solver,
            guidance_rescale=guidance_rescale,
            mel_clamp=mel_clamp,
            reference=reference,
            context_guidance_scale=context_guidance_scale,
            min_seconds_per_char=min_seconds_per_char,
            chunk_character_limit=chunk_character_limit,
            chunk_pause_seconds=chunk_pause_seconds,
            auto_style=auto_style,
            mel_correction=mel_correction,
            text_normalization=train_config.text_normalization,
        )
        audio_path = output_dir / f"{index:03d}-{prompt_id}.wav"
        sf.write(audio_path, waveform, train_config.sample_rate)
        audio_seconds = len(waveform) / train_config.sample_rate
        total_audio_seconds += audio_seconds
        samples.append(
            {
                "id": prompt_id,
                "category": prompt["category"],
                "text": text,
                "normalized_text": normalized,
                "audio_path": str(audio_path),
                "audio_seconds": audio_seconds,
                "predicted_frames": predicted_frames,
                "seed": seed + index - 1,
            }
        )
        print(f"{index:03d}/{len(prompts):03d} {prompt_id} {audio_seconds:.2f}s", flush=True)
    elapsed = time.monotonic() - started
    report = {
        "run_version": f"{payload['run_version']}-checkpoint-{payload['update']}",
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": payload["update"],
        "sample_rate_hz": train_config.sample_rate,
        "steps": steps,
        "seed": seed,
        "model_weights": "ema" if use_ema else "raw",
        "adapter_scale": adapter_scale,
        "speaker": speaker,
        "speaker_id": speaker_id,
        "prosody": list(prosody) if prosody is not None else None,
        "text_guidance_scale": text_guidance_scale,
        "speaker_guidance_scale": speaker_guidance_scale,
        "sway_coefficient": sway_coefficient,
        "solver": solver,
        "guidance_rescale": guidance_rescale,
        "mel_clamp": mel_clamp,
        "reference_audio": str(reference_audio) if reference_audio is not None else None,
        "reference_text": reference_text,
        "context_guidance_scale": context_guidance_scale,
        "min_seconds_per_char": min_seconds_per_char,
        "chunk_character_limit": chunk_character_limit,
        "chunk_pause_seconds": chunk_pause_seconds,
        "samples": samples,
        "totals": {
            "prompts": len(samples),
            "audio_seconds": total_audio_seconds,
            "generation_seconds": elapsed,
            "real_time_factor": elapsed / max(total_audio_seconds, 1e-6),
        },
    }
    report_path = output_dir / "crossflow.synthesis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report_path
