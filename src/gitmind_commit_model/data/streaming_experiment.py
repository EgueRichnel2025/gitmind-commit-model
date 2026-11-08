"""Streaming, checkpointing and resumable real-commit experiments."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from gitmind_commit_model.data.real_commit_experiment import (
    RealCommitAccumulator,
    benchmark_real_commit,
    find_sensitive_patterns,
)
from gitmind_commit_model.data.records import (
    build_commit_record,
    build_file_sections,
    is_supported_license,
)


class StatefulStream(Protocol):
    """Minimal interface required from a resumable streaming dataset."""

    def __iter__(self) -> Any: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> Any: ...


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid corrupting checkpoints."""

    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_name, path)

    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def process_commit_row(
    row: Any,
    tokenizer: Any,
    accumulator: RealCommitAccumulator,
    *,
    max_source_tokens: int = 512,
    overlap_tokens: int = 32,
    max_target_tokens: int = 128,
) -> str:
    """Process one dataset row without retaining its raw content."""

    accumulator.rows_examined += 1

    if not isinstance(row, dict):
        accumulator.reject("invalid_row")
        return "rejected"

    if not str(row.get("message") or "").strip():
        accumulator.reject("missing_message")
        return "rejected"

    if not is_supported_license(row.get("license")):
        accumulator.reject("unsupported_license")
        return "rejected"

    if not build_file_sections(row.get("mods")):
        accumulator.reject("no_usable_diff")
        return "rejected"

    record = build_commit_record(row)

    if record is None:
        accumulator.reject("invalid_record")
        return "rejected"

    sensitive_patterns = find_sensitive_patterns(record)

    if sensitive_patterns:
        accumulator.reject(
            "potential_sensitive_content",
            sensitive_patterns=sensitive_patterns,
        )
        return "rejected"

    try:
        metrics = benchmark_real_commit(
            record,
            tokenizer,
            max_source_tokens=max_source_tokens,
            overlap_tokens=overlap_tokens,
            max_target_tokens=max_target_tokens,
        )

    except (AssertionError, ValueError) as error:
        accumulator.reject(
            "chunking_error",
            error_type=type(error).__name__,
        )
        return "rejected"

    accumulator.add_commit(record, metrics)

    return "accepted"


def load_checkpoint(
    path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Load a checkpoint and verify its experiment identity."""

    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema version.")

    if payload.get("metadata") != expected_metadata:
        raise ValueError(
            "Checkpoint configuration does not match this experiment. "
            "Use the same dataset revision and model parameters, "
            "or select another checkpoint directory."
        )

    return payload


def save_checkpoint(
    path: Path,
    stream: StatefulStream,
    accumulator: RealCommitAccumulator,
    metadata: dict[str, Any],
    *,
    completed: bool,
) -> None:
    """Save the stream position and matching statistics atomically."""

    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "stream_state": stream.state_dict(),
        "accumulator": accumulator.state_dict(),
        "completed": completed,
    }

    atomic_write_json(path, payload)


def consume_stream(
    stream: StatefulStream,
    tokenizer: Any,
    *,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    checkpoint_every: int = 5_000,
    progress_every: int = 1_000,
    max_rows: int | None = None,
    reservoir_size: int = 10_000,
    seed: int = 42,
    max_source_tokens: int = 512,
    overlap_tokens: int = 32,
    max_target_tokens: int = 128,
    on_checkpoint: Callable[[RealCommitAccumulator, bool], None] | None = None,
) -> tuple[RealCommitAccumulator, bool]:
    """Consume a stream with bounded memory and resumable checkpoints.

    ``max_rows`` is a cumulative limit for the partition, not an additional
    number of rows. Omitting it allows the stream to run to exhaustion.
    """

    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive.")

    if progress_every < 1:
        raise ValueError("progress_every must be positive.")

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive.")

    if checkpoint_path.exists():
        payload = load_checkpoint(checkpoint_path, metadata)

        accumulator = RealCommitAccumulator.from_state_dict(payload["accumulator"])

        if payload["completed"]:
            print(
                "Partition déjà terminée :",
                accumulator.rows_examined,
                "lignes examinées.",
            )
            return accumulator, True

        stream.load_state_dict(payload["stream_state"])

        print(
            "Reprise depuis le checkpoint :",
            accumulator.rows_examined,
            "lignes déjà examinées.",
        )

    else:
        accumulator = RealCommitAccumulator(
            reservoir_size=reservoir_size,
            seed=seed,
        )

    if max_rows is not None and accumulator.rows_examined >= max_rows:
        return accumulator, False

    iterator = iter(stream)
    started = perf_counter()
    last_progress_time = started
    last_progress_rows = accumulator.rows_examined

    def persist(completed: bool) -> None:
        save_checkpoint(
            checkpoint_path,
            stream,
            accumulator,
            metadata,
            completed=completed,
        )

        if on_checkpoint is not None:
            on_checkpoint(accumulator, completed)

    while True:
        try:
            row = next(iterator)

        except StopIteration:
            persist(completed=True)
            return accumulator, True

        process_commit_row(
            row,
            tokenizer,
            accumulator,
            max_source_tokens=max_source_tokens,
            overlap_tokens=overlap_tokens,
            max_target_tokens=max_target_tokens,
        )

        rows = accumulator.rows_examined

        if rows % progress_every == 0:
            now = perf_counter()
            interval_elapsed = max(now - last_progress_time, 1e-9)
            interval_rows = rows - last_progress_rows
            interval_speed = interval_rows / interval_elapsed
            total_elapsed = now - started

            print(
                f"Lignes : {rows:,} | "
                f"Acceptés : {accumulator.accepted_commits:,} | "
                f"Rejetés : {sum(accumulator.rejected_by_reason.values()):,} | "
                f"Vitesse récente : {interval_speed:.2f} lignes/s | "
                f"Temps de cette session : {total_elapsed:.1f}s"
            )

            last_progress_time = now
            last_progress_rows = rows

        limit_reached = max_rows is not None and rows >= max_rows
        checkpoint_due = rows % checkpoint_every == 0

        if checkpoint_due or limit_reached:
            persist(completed=False)

            print(
                f"Checkpoint enregistré : {rows:,} lignes "
                f"({accumulator.accepted_commits:,} commits acceptés)."
            )

        if limit_reached:
            print("Limite de lignes atteinte. Partition laissée incomplète.")
            return accumulator, False
