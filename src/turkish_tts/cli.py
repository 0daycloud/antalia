from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from turkish_tts.asr_labeling import (
    TurkishCtcRecognizer,
    calibrate_labeling_thresholds,
    labeling_summary,
    read_labeling_thresholds,
    run_ctc_consensus_and_alignment,
    run_whisper_pseudolabels,
    write_labeling_config,
)
from turkish_tts.audio import enrich_record
from turkish_tts.baseline import (
    BASELINE_DATASET_VERSION,
    evaluate_fastpitch_synthesis,
    load_evaluation_suite,
    prepare_candidate_baseline_data,
    prepare_fastpitch_features,
    prepare_fastpitch_stage_data,
    prepare_target_speaker_data,
    synthesize_fastpitch_prompts,
    train_fastpitch_candidate,
)
from turkish_tts.common_voice import iter_common_voice_validated
from turkish_tts.common_voice_prepare import (
    TransformersWhisperBatchTranscriber,
    acoustic_audit_summary,
    audit_common_voice_records,
    finalize_common_voice_splits,
    transcribe_common_voice_with_checkpoints,
)
from turkish_tts.dataset_finalize import CombinedDatasetConfig, build_combined_candidate_dataset
from turkish_tts.fleurs import (
    FLEURS_LICENSE_ID,
    iter_fleurs_turkish,
    resolve_fleurs_revision,
)
from turkish_tts.manifests import RightsState, iter_jsonl, write_export_summary, write_jsonl
from turkish_tts.normalize import normalize_for_asr_comparison, normalize_orthography
from turkish_tts.quality_gates import (
    SpeakerEmbedder,
    calibrate_speaker_thresholds,
    quality_gate_summary,
    read_speaker_thresholds,
    run_acoustic_fingerprint_gate,
    run_privacy_gate,
    run_speaker_consistency_gate,
    write_speaker_config,
)
from turkish_tts.scripted_segments import (
    ElevenLabsScribeTimedTranscriber,
    FasterWhisperTimedTranscriber,
    TimedTranscriber,
    segment_scripted_records,
)
from turkish_tts.settings import Settings
from turkish_tts.transcribe import transcribe_records

app = typer.Typer(no_args_is_help=True, help="Turkish flagship TTS data pipeline.")
manifest_app = typer.Typer(no_args_is_help=True, help="Build and validate data manifests.")
label_app = typer.Typer(no_args_is_help=True, help="Run automatic VoiceData labeling and quality gates.")
baseline_app = typer.Typer(no_args_is_help=True, help="Prepare and train the Candidate A baseline.")
crossflow_app = typer.Typer(
    no_args_is_help=True,
    help="Train and synthesize the independent cross-attention flow model.",
)
app.add_typer(manifest_app, name="manifest")
app.add_typer(label_app, name="label")
app.add_typer(baseline_app, name="baseline")
app.add_typer(crossflow_app, name="crossflow")
console = Console()


@manifest_app.command("common-voice")
def common_voice_manifest(
    validated_tsv: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    clips_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    output: Annotated[Path, typer.Option()],
    version: Annotated[str, typer.Option()] = "26.0",
    allow_missing_audio: Annotated[bool, typer.Option()] = False,
) -> None:
    records = iter_common_voice_validated(
        tsv_path=validated_tsv,
        clips_dir=clips_dir,
        version=version,
        require_audio=not allow_missing_audio,
    )
    count = write_jsonl(output, records)
    console.print(f"Wrote {count:,} Common Voice records to {output}")


@manifest_app.command("fleurs")
def fleurs_manifest(
    output_manifest: Annotated[Path, typer.Option()],
    output_audio_dir: Annotated[Path, typer.Option()],
    revision: Annotated[str | None, typer.Option()] = None,
    report: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = Settings()
    resolved_revision = revision or resolve_fleurs_revision(token=settings.hf_token)
    records = list(
        iter_fleurs_turkish(
            output_dir=output_audio_dir,
            revision=resolved_revision,
            token=settings.hf_token,
        )
    )
    if len({record.clip_id for record in records}) != len(records):
        raise ValueError("FLEURS import produced duplicate clip IDs")
    if len({record.audio_path for record in records}) != len(records):
        raise ValueError("FLEURS import produced duplicate audio paths")
    count = write_jsonl(output_manifest, records)
    split_counts = Counter(record.source_split or "unknown" for record in records)
    summary: dict[str, object] = {
        "dataset": "google/fleurs",
        "config": "tr_tr",
        "revision": resolved_revision,
        "license_id": FLEURS_LICENSE_ID,
        "records": count,
        "hours": round(sum(record.duration_seconds or 0 for record in records) / 3600, 3),
        "splits": dict(sorted(split_counts.items())),
    }
    write_export_summary(report or output_manifest.with_suffix(".report.json"), summary)
    console.print(f"Wrote {count:,} FLEURS Turkish records at {resolved_revision} to {output_manifest}")


@manifest_app.command("common-voice-audit")
def common_voice_audit(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    report: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    audited = audit_common_voice_records(records, max_concurrency=settings.max_concurrency)
    count = write_jsonl(output_manifest, audited)
    write_export_summary(report or output_manifest.with_suffix(".report.json"), acoustic_audit_summary(audited))
    console.print(f"Wrote {count:,} acoustically audited Common Voice records to {output_manifest}")


@manifest_app.command("common-voice-asr")
def common_voice_asr(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path | None, typer.Option()] = None,
    model: Annotated[str, typer.Option()] = "openai/whisper-large-v3",
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
) -> None:
    records = list(iter_jsonl(input_manifest))
    transcriber = TransformersWhisperBatchTranscriber(model_name=model, batch_size=batch_size)
    annotated = transcribe_common_voice_with_checkpoints(
        records,
        transcriber=transcriber,
        model_name=model,
        checkpoint_path=checkpoint or output_manifest.with_suffix(".checkpoint.jsonl"),
        output_path=output_manifest,
    )
    console.print(f"Wrote {len(annotated):,} ASR-audited Common Voice records to {output_manifest}")


@manifest_app.command("common-voice-finalize")
def common_voice_finalize(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    prefix: Annotated[str, typer.Option()] = "common-voice-26.0-tr",
    report: Annotated[Path | None, typer.Option()] = None,
) -> None:
    records = list(iter_jsonl(input_manifest))
    summary = finalize_common_voice_splits(records, output_dir=output_dir, prefix=prefix)
    report_path = report or output_dir / f"{prefix}.report.json"
    write_export_summary(report_path, summary)
    console.print(
        f"Accepted {summary['accepted_records']:,} and rejected {summary['rejected_records']:,} Common Voice records"
    )


@manifest_app.command("validate")
def validate_manifest(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    count = 0
    speakers: set[str] = set()
    rights: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    missing_audio = 0
    for record in iter_jsonl(path):
        count += 1
        if record.speaker_id:
            speakers.add(record.speaker_id)
        rights[record.rights_state.value] += 1
        sources[f"{record.source_dataset}@{record.source_version}"] += 1
        if not Path(record.audio_path).is_file():
            missing_audio += 1

    table = Table(title=str(path))
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Records", f"{count:,}")
    table.add_row("Speakers", f"{len(speakers):,}")
    table.add_row("Missing audio", f"{missing_audio:,}")
    table.add_row("Rights", ", ".join(f"{key}={value:,}" for key, value in sorted(rights.items())))
    table.add_row("Sources", ", ".join(f"{key}={value:,}" for key, value in sorted(sources.items())))
    console.print(table)
    if not count:
        raise typer.Exit(1)


@manifest_app.command("probe")
def probe_manifest(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
) -> None:
    records = (enrich_record(record) for record in iter_jsonl(input_manifest))
    count = write_jsonl(output_manifest, records)
    console.print(f"Wrote {count:,} probed records to {output_manifest}")


@label_app.command("segment")



@label_app.command("segment-scripted")
def segment_scripted_campaign(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    output_audio_dir: Annotated[Path, typer.Option()],
    report: Annotated[Path | None, typer.Option()] = None,
    asr_cache_dir: Annotated[Path | None, typer.Option()] = None,
    asr_provider: Annotated[Literal["auto", "faster-whisper", "elevenlabs-scribe"], typer.Option()] = "auto",
    model_name: Annotated[str, typer.Option(help="Faster Whisper model name.")] = "large-v3",
    device: Annotated[str, typer.Option()] = "cuda",
    compute_type: Annotated[str, typer.Option()] = "float16",
    num_workers: Annotated[int, typer.Option(min=1)] = 1,
    scribe_api_key: Annotated[str | None, typer.Option(envvar="ELEVENLABS_API_KEY")] = None,
    scribe_model_name: Annotated[str, typer.Option()] = "scribe_v2",
    scribe_zero_retention: Annotated[bool, typer.Option()] = False,
) -> None:
    records = list(iter_jsonl(input_manifest))
    transcriber: TimedTranscriber
    use_scribe = asr_provider == "elevenlabs-scribe" or (asr_provider == "auto" and bool(scribe_api_key))
    if use_scribe:
        if not scribe_api_key:
            raise typer.BadParameter("Set ELEVENLABS_API_KEY or pass --scribe-api-key.")
        transcriber = ElevenLabsScribeTimedTranscriber(
            api_key=scribe_api_key,
            model_name=scribe_model_name,
            zero_retention=scribe_zero_retention,
        )
    else:
        transcriber = FasterWhisperTimedTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )
    result = segment_scripted_records(
        records,
        output_dir=output_audio_dir,
        transcriber=transcriber,
        transcript_cache_dir=asr_cache_dir or output_manifest.parent / "scripted-asr-cache",
    )
    count = write_jsonl(output_manifest, result.segments)
    write_export_summary(report or output_manifest.with_suffix(".report.json"), result.report)
    console.print(
        f"Wrote {count:,} script-aligned segments ({result.report['accepted_segments']:,} accepted) "
        f"to {output_manifest}"
    )


@label_app.command("calibrate-asr")
def calibrate_asr(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_config: Annotated[Path, typer.Option()],
    sample_size: Annotated[int, typer.Option(min=100)] = 2_000,
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    recognizer = TurkishCtcRecognizer(token=settings.hf_token)
    thresholds, report = calibrate_labeling_thresholds(
        records,
        recognizer=recognizer,
        sample_size=sample_size,
        source_name=f"{records[0].source_dataset}@{records[0].source_version}",
    )
    write_labeling_config(output_config, thresholds=thresholds, report=report)
    console.print(f"Wrote ASR and alignment calibration to {output_config}")


@label_app.command("whisper")
def label_whisper(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 128,
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    labeled = run_whisper_pseudolabels(
        records,
        token=settings.hf_token,
        checkpoint_path=checkpoint or output_manifest.with_suffix(".checkpoint.jsonl"),
        output_path=output_manifest,
        batch_size=batch_size,
    )
    console.print(f"Wrote {len(labeled):,} Whisper pseudo-label records to {output_manifest}")


@label_app.command("ctc-align")
def label_ctc_align(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    checkpoint: Annotated[Path | None, typer.Option()] = None,
    report: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 24,
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    labeled = run_ctc_consensus_and_alignment(
        records,
        thresholds=read_labeling_thresholds(calibration),
        token=settings.hf_token,
        checkpoint_path=checkpoint or output_manifest.with_suffix(".checkpoint.jsonl"),
        output_path=output_manifest,
        batch_size=batch_size,
    )
    write_export_summary(report or output_manifest.with_suffix(".report.json"), labeling_summary(labeled))
    console.print(f"Wrote {len(labeled):,} dual-ASR aligned records to {output_manifest}")


@label_app.command("privacy")
def label_privacy(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    cpu: Annotated[bool, typer.Option()] = False,
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    gated = run_privacy_gate(
        records,
        token=settings.hf_token,
        output_path=output_manifest,
        use_gpu=not cpu,
    )
    console.print(f"Wrote {len(gated):,} privacy-gated records to {output_manifest}")


@label_app.command("calibrate-speaker")
def calibrate_speaker(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_config: Annotated[Path, typer.Option()],
    max_speakers: Annotated[int, typer.Option(min=10)] = 300,
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    embedder = SpeakerEmbedder(token=settings.hf_token)
    thresholds, report = calibrate_speaker_thresholds(
        records,
        embedder=embedder,
        source_name=f"{records[0].source_dataset}@{records[0].source_version}",
        max_speakers=max_speakers,
    )
    write_speaker_config(output_config, thresholds=thresholds, report=report)
    console.print(f"Wrote speaker calibration to {output_config}")


@label_app.command("speaker")
def label_speaker(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    calibration: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    settings = Settings()
    records = list(iter_jsonl(input_manifest))
    gated = run_speaker_consistency_gate(
        records,
        embedder=SpeakerEmbedder(token=settings.hf_token),
        thresholds=read_speaker_thresholds(calibration),
        output_path=output_manifest,
    )
    console.print(f"Wrote {len(gated):,} speaker-consistency records to {output_manifest}")


@label_app.command("fingerprint")
def label_fingerprint(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    report: Annotated[Path | None, typer.Option()] = None,
) -> None:
    records = list(iter_jsonl(input_manifest))
    gated = run_acoustic_fingerprint_gate(records, output_path=output_manifest)
    write_export_summary(report or output_manifest.with_suffix(".report.json"), quality_gate_summary(gated))
    console.print(f"Wrote {len(gated):,} duplicate-gated records to {output_manifest}")


@label_app.command("finalize")
def label_finalize(
    common_voice_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    fleurs_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    voicedata_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    prefix: Annotated[str, typer.Option()] = "turkish-flagship-candidate-a-v1",
    report: Annotated[Path | None, typer.Option()] = None,
    max_public_speaker_seconds: Annotated[float, typer.Option(min=60)] = 30 * 60,
    max_common_voice_seconds: Annotated[float, typer.Option(min=3600)] = 120 * 3600,
) -> None:
    result = build_combined_candidate_dataset(
        list(iter_jsonl(common_voice_manifest)),
        list(iter_jsonl(fleurs_manifest)),
        list(iter_jsonl(voicedata_manifest)),
        output_dir=output_dir,
        prefix=prefix,
        config=CombinedDatasetConfig(
            max_public_speaker_seconds=max_public_speaker_seconds,
            max_common_voice_seconds=max_common_voice_seconds,
        ),
    )
    report_path = report or output_dir / f"{prefix}.report.json"
    write_export_summary(report_path, result.report)
    console.print(
        f"Wrote {len(result.accepted):,} accepted records and "
        f"{len(result.train) + len(result.validation) + len(result.test):,} Candidate A training records"
    )


@baseline_app.command("prepare")
def baseline_prepare(
    combined_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    evaluation_suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    combined_prefix: Annotated[str, typer.Option()] = "turkish-flagship-candidate-a-v1",
) -> None:
    result = prepare_candidate_baseline_data(
        combined_dir=combined_dir,
        combined_prefix=combined_prefix,
        output_dir=output_dir,
        evaluation_suite=evaluation_suite,
    )
    console.print(
        f"Wrote {sum(result.records.values()):,} {BASELINE_DATASET_VERSION} records "
        f"({sum(result.hours.values()):.3f} hours) to {output_dir}"
    )


@baseline_app.command("prepare-target")
def baseline_prepare_target(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    target_speaker_id: Annotated[str, typer.Option()],
    target_alias: Annotated[str, typer.Option()],
    dataset_version: Annotated[str, typer.Option()],
    evaluation_suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    source_dataset: Annotated[str, typer.Option()] = "voicedata-turkish",
) -> None:
    result = prepare_target_speaker_data(
        input_manifest=input_manifest,
        target_speaker_id=target_speaker_id,
        target_alias=target_alias,
        dataset_version=dataset_version,
        output_dir=output_dir,
        evaluation_suite=evaluation_suite,
        source_dataset=source_dataset,
    )
    console.print(
        f"Wrote {sum(result.records.values()):,} {result.dataset_version} records "
        f"({sum(result.hours.values()):.3f} hours) to {output_dir}"
    )


@baseline_app.command("prepare-stage")
def baseline_prepare_stage(
    combined_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    evaluation_suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    stage: Annotated[Literal["foundation"], typer.Option()],
    combined_prefix: Annotated[str, typer.Option()] = "turkish-flagship-candidate-a-v1",
) -> None:
    result = prepare_fastpitch_stage_data(
        combined_dir=combined_dir,
        combined_prefix=combined_prefix,
        output_dir=output_dir,
        evaluation_suite=evaluation_suite,
        stage=stage,
    )
    console.print(
        f"Wrote {sum(result.records.values()):,} {result.dataset_version} records "
        f"({sum(result.hours.values()):.3f} hours) to {output_dir}"
    )


@baseline_app.command("prepare-voicedata")



@baseline_app.command("features")
def baseline_features(
    dataset_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    dataset_version: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    workers: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    report_path = prepare_fastpitch_features(
        dataset_dir=dataset_dir,
        config_path=config,
        feature_dir=dataset_dir / "features-v1",
        dataset_version=dataset_version,
        num_workers=workers,
    )
    console.print(f"Wrote {dataset_version} features and report to {report_path}")


@baseline_app.command("train")
def baseline_train(
    dataset_dir: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    dataset_version: Annotated[str, typer.Option()] = BASELINE_DATASET_VERSION,
    run_version: Annotated[str, typer.Option()] = "fastpitch-candidate-a-v1",
    initial_model: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    feature_workers: Annotated[int, typer.Option(min=1)] = 1,
    max_epochs: Annotated[int | None, typer.Option(min=1)] = None,
    max_steps: Annotated[int | None, typer.Option(min=1)] = None,
    devices: Annotated[int | None, typer.Option(min=1)] = None,
    strategy: Annotated[str | None, typer.Option()] = None,
) -> None:
    model_path = train_fastpitch_candidate(
        dataset_dir=dataset_dir,
        config_path=config,
        output_dir=output_dir,
        dataset_version=dataset_version,
        run_version=run_version,
        initial_model_path=initial_model,
        feature_workers=feature_workers,
        max_epochs=max_epochs,
        max_steps=max_steps,
        devices=devices,
        strategy=strategy,
    )
    console.print(f"Wrote FastPitch model to {model_path}")


@baseline_app.command("synthesize")
def baseline_synthesize(
    model: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    evaluation_suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    bigvgan_source: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    vocoder_dir: Annotated[Path, typer.Option()],
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    device: Annotated[str, typer.Option()] = "cuda",
    speaker_index: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    prompts = load_evaluation_suite(evaluation_suite)
    if limit is not None:
        prompts = prompts[:limit]
    report_path = synthesize_fastpitch_prompts(
        model_path=model,
        prompts=prompts,
        output_dir=output_dir,
        bigvgan_source=bigvgan_source,
        vocoder_dir=vocoder_dir,
        device=device,
        speaker_index=speaker_index,
    )
    console.print(f"Wrote {len(prompts):,} synthesized prompts and report to {report_path}")


@baseline_app.command("evaluate")
def baseline_evaluate(
    synthesis_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    reference_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    settings = Settings()
    report_path = evaluate_fastpitch_synthesis(
        synthesis_report=synthesis_report,
        reference_manifest=reference_manifest,
        output_path=output,
        token=settings.hf_token,
        device=device,
    )
    console.print(f"Wrote synthesis quality report to {report_path}")


@baseline_app.command("validate-evaluation")
def baseline_validate_evaluation(
    evaluation_suite: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    prompts = load_evaluation_suite(evaluation_suite)
    console.print(f"Validated {len(prompts):,} frozen Turkish prompts")


@crossflow_app.command("train")
def crossflow_train(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    resume: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    from turkish_tts.crossflow_train import train_crossflow

    checkpoint = train_crossflow(config, resume_path=resume)
    console.print(f"Wrote CrossFlow checkpoint to {checkpoint}")


@crossflow_app.command("synthesize")
def crossflow_synthesize(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    text: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    vocoder: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    device: Annotated[str, typer.Option()] = "cuda",
    steps: Annotated[int, typer.Option(min=1)] = 32,
    seed: Annotated[int, typer.Option()] = 20260803,
    duration_scale: Annotated[float, typer.Option(min=0.5, max=2.0)] = 1.0,
) -> None:
    from turkish_tts.crossflow_train import synthesize_crossflow

    report = synthesize_crossflow(
        checkpoint_path=checkpoint,
        text=text,
        output_path=output,
        vocoder_dir=vocoder,
        device=device,
        steps=steps,
        seed=seed,
        duration_scale=duration_scale,
    )
    console.print(report)


@crossflow_app.command("benchmark")
def crossflow_benchmark(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    plan: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    vocoder: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    device: Annotated[str, typer.Option()] = "cuda",
    steps: Annotated[int, typer.Option(min=1)] = 32,
    seed: Annotated[int, typer.Option()] = 20260803,
    duration_scale: Annotated[float, typer.Option(min=0.5, max=2.0)] = 1.0,
) -> None:
    from turkish_tts.inference_benchmark import benchmark_crossflow_checkpoint

    report = benchmark_crossflow_checkpoint(
        checkpoint_path=checkpoint,
        benchmark_plan=plan,
        output_dir=output_dir,
        vocoder_dir=vocoder,
        device=device,
        steps=steps,
        seed=seed,
        duration_scale=duration_scale,
    )
    console.print(f"Wrote inference benchmark to {report}")


@crossflow_app.command("inspect")
def crossflow_inspect(
    checkpoint: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    console.print(
        {
            "run_version": payload["run_version"],
            "update": payload["update"],
            "epoch": payload["epoch"],
            "model_config": payload["model_config"],
            "provenance": payload["provenance"],
        }
    )


@app.command("normalize")
def normalize_text(
    text: Annotated[str, typer.Argument()],
    comparison: Annotated[bool, typer.Option()] = False,
) -> None:
    output = normalize_for_asr_comparison(text) if comparison else normalize_orthography(text)
    console.print(output)


@app.command("transcribe-openai")
def transcribe_openai(
    input_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_manifest: Annotated[Path, typer.Option()],
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    settings = Settings()
    if not settings.openai_api_key:
        raise typer.BadParameter("OPENAI_API_KEY is required")
    records = [record for record in iter_jsonl(input_manifest) if record.rights_state == RightsState.ALLOWED]
    transcribed = asyncio.run(
        transcribe_records(
            records,
            api_key=settings.openai_api_key,
            model=settings.transcription_model,
            language=settings.language,
            max_concurrency=settings.max_concurrency,
            overwrite=overwrite,
        )
    )
    count = write_jsonl(output_manifest, transcribed)
    console.print(f"Wrote {count:,} transcribed records to {output_manifest}")


if __name__ == "__main__":
    app()
