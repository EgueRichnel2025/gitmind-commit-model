"""Scalable validation and statistics for real Git commit experiments."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from time import perf_counter
from typing import Any

from gitmind_commit_model.data.chunking import (
    TokenizerProtocol,
    chunk_commit_record,
)
from gitmind_commit_model.data.records import CommitRecord

SENSITIVE_PATTERNS = {
    "email": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


class FastTokenizerAdapter:
    """Use the fast tokenizer backend without model-length warnings."""

    def __init__(self, tokenizer: Any) -> None:
        backend = getattr(tokenizer, "backend_tokenizer", None)

        if backend is None:
            raise TypeError("A fast tokenizer backend is required.")

        self._backend = backend
        self._tokenizer = tokenizer

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self._backend.encode(
            text,
            add_special_tokens=add_special_tokens,
        ).ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self._backend.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tokenizer.num_special_tokens_to_add(pair=pair)


def _state_to_json(value: Any) -> Any:
    """Convert nested tuples into JSON-compatible lists."""

    if isinstance(value, tuple):
        return [_state_to_json(item) for item in value]

    if isinstance(value, list):
        return [_state_to_json(item) for item in value]

    return value


def _state_to_tuple(value: Any) -> Any:
    """Restore random generator states from JSON-compatible lists."""

    if isinstance(value, list):
        return tuple(_state_to_tuple(item) for item in value)

    return value


class OnlineStats:
    """Exact basic statistics with a bounded random reservoir for quantiles."""

    def __init__(
        self,
        *,
        reservoir_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        if reservoir_size < 1:
            raise ValueError("reservoir_size must be positive.")

        self.reservoir_size = reservoir_size
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.reservoir: list[float] = []
        self.random = random.Random(seed)

    def update(self, value: int | float) -> None:
        """Add one numeric value using reservoir sampling."""

        number = float(value)

        if not math.isfinite(number):
            raise ValueError("Statistics cannot contain non-finite values.")

        self.count += 1
        self.total += number

        if self.minimum is None or number < self.minimum:
            self.minimum = number

        if self.maximum is None or number > self.maximum:
            self.maximum = number

        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(number)
            return

        replacement_index = self.random.randrange(self.count)

        if replacement_index < self.reservoir_size:
            self.reservoir[replacement_index] = number

    def summary(self) -> dict[str, Any]:
        """Return exact basic statistics and sampled quantiles."""

        if self.count == 0:
            return {"count": 0}

        ordered_sample = sorted(self.reservoir)

        def percentile(fraction: float) -> float:
            index = round((len(ordered_sample) - 1) * fraction)
            return round(ordered_sample[index], 4)

        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(self.total / self.count, 4),
            "median_sampled": percentile(0.50),
            "p90_sampled": percentile(0.90),
            "p95_sampled": percentile(0.95),
            "reservoir_size": len(self.reservoir),
            "quantiles_approximate": len(self.reservoir) < self.count,
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize the accumulator for checkpointing."""

        return {
            "reservoir_size": self.reservoir_size,
            "count": self.count,
            "total": self.total,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "reservoir": self.reservoir,
            "random_state": _state_to_json(self.random.getstate()),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> OnlineStats:
        """Restore an accumulator without changing its future sampling."""

        instance = cls(
            reservoir_size=state["reservoir_size"],
        )

        instance.count = state["count"]
        instance.total = state["total"]
        instance.minimum = state["minimum"]
        instance.maximum = state["maximum"]
        instance.reservoir = list(state["reservoir"])
        instance.random.setstate(_state_to_tuple(state["random_state"]))

        return instance


class RealCommitAccumulator:
    """Bounded-memory statistics for a complete dataset partition."""

    METRIC_NAMES = (
        "files_per_commit",
        "diff_characters_per_commit",
        "source_tokens_per_commit",
        "source_tokens_per_file",
        "chunks_per_commit",
        "chunks_per_file",
        "chunk_input_tokens",
        "target_tokens",
        "processing_seconds_per_commit",
    )

    def __init__(
        self,
        *,
        reservoir_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.reservoir_size = reservoir_size
        self.seed = seed

        self.rows_examined = 0
        self.accepted_commits = 0

        self.rejected_by_reason: Counter[str] = Counter()
        self.sensitive_pattern_counts: Counter[str] = Counter()
        self.chunking_error_types: Counter[str] = Counter()

        self.language_counts: Counter[str] = Counter()
        self.license_counts: Counter[str] = Counter()
        self.change_type_counts: Counter[str] = Counter()

        self.single_file_commits = 0
        self.multi_file_commits = 0

        self.total_generated_chunks = 0
        self.targets_over_max_length = 0
        self.max_chunk_tokens_observed = 0

        self.statistics = {
            name: OnlineStats(
                reservoir_size=reservoir_size,
                seed=seed + index * 1009,
            )
            for index, name in enumerate(self.METRIC_NAMES)
        }

    def reject(
        self,
        reason: str,
        *,
        sensitive_patterns: tuple[str, ...] = (),
        error_type: str | None = None,
    ) -> None:
        """Register a rejected commit without retaining its contents."""

        self.rejected_by_reason[reason] += 1

        for name in sensitive_patterns:
            self.sensitive_pattern_counts[name] += 1

        if error_type:
            self.chunking_error_types[error_type] += 1

    def add_commit(
        self,
        record: CommitRecord,
        metrics: dict[str, Any],
    ) -> None:
        """Add an accepted commit to the running statistics."""

        self.accepted_commits += 1

        file_count = metrics["file_count"]

        if file_count == 1:
            self.single_file_commits += 1
        else:
            self.multi_file_commits += 1

        self.language_counts[record.language or "unknown"] += 1
        self.license_counts[record.license] += 1
        self.change_type_counts.update(metrics["change_types"])

        self.total_generated_chunks += metrics["generated_chunks"]
        self.targets_over_max_length += int(metrics["target_over_max_length"])

        if metrics["chunk_input_tokens"]:
            self.max_chunk_tokens_observed = max(
                self.max_chunk_tokens_observed,
                max(metrics["chunk_input_tokens"]),
            )

        values = {
            "files_per_commit": [file_count],
            "diff_characters_per_commit": [metrics["diff_characters"]],
            "source_tokens_per_commit": [metrics["total_source_tokens"]],
            "source_tokens_per_file": metrics["source_tokens_per_file"],
            "chunks_per_commit": [metrics["generated_chunks"]],
            "chunks_per_file": metrics["chunks_per_file"],
            "chunk_input_tokens": metrics["chunk_input_tokens"],
            "target_tokens": [metrics["target_tokens"]],
            "processing_seconds_per_commit": [metrics["processing_seconds"]],
        }

        for name, observations in values.items():
            for observation in observations:
                self.statistics[name].update(observation)

    def state_dict(self) -> dict[str, Any]:
        """Serialize all counters and bounded statistics."""

        return {
            "reservoir_size": self.reservoir_size,
            "seed": self.seed,
            "rows_examined": self.rows_examined,
            "accepted_commits": self.accepted_commits,
            "rejected_by_reason": dict(self.rejected_by_reason),
            "sensitive_pattern_counts": dict(self.sensitive_pattern_counts),
            "chunking_error_types": dict(self.chunking_error_types),
            "language_counts": dict(self.language_counts),
            "license_counts": dict(self.license_counts),
            "change_type_counts": dict(self.change_type_counts),
            "single_file_commits": self.single_file_commits,
            "multi_file_commits": self.multi_file_commits,
            "total_generated_chunks": self.total_generated_chunks,
            "targets_over_max_length": self.targets_over_max_length,
            "max_chunk_tokens_observed": self.max_chunk_tokens_observed,
            "statistics": {name: stats.state_dict() for name, stats in self.statistics.items()},
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> RealCommitAccumulator:
        """Restore a complete partition accumulator."""

        instance = cls(
            reservoir_size=state["reservoir_size"],
            seed=state["seed"],
        )

        instance.rows_examined = state["rows_examined"]
        instance.accepted_commits = state["accepted_commits"]

        instance.rejected_by_reason = Counter(state["rejected_by_reason"])
        instance.sensitive_pattern_counts = Counter(state["sensitive_pattern_counts"])
        instance.chunking_error_types = Counter(state["chunking_error_types"])

        instance.language_counts = Counter(state["language_counts"])
        instance.license_counts = Counter(state["license_counts"])
        instance.change_type_counts = Counter(state["change_type_counts"])

        instance.single_file_commits = state["single_file_commits"]
        instance.multi_file_commits = state["multi_file_commits"]

        instance.total_generated_chunks = state["total_generated_chunks"]
        instance.targets_over_max_length = state["targets_over_max_length"]
        instance.max_chunk_tokens_observed = state["max_chunk_tokens_observed"]

        instance.statistics = {
            name: OnlineStats.from_state_dict(stats_state)
            for name, stats_state in state["statistics"].items()
        }

        if set(instance.statistics) != set(instance.METRIC_NAMES):
            raise ValueError("Checkpoint statistics schema is incompatible.")

        return instance

    def summary(self) -> dict[str, Any]:
        """Build an aggregate-only report for one partition."""

        return {
            "counts": {
                "rows_examined": self.rows_examined,
                "accepted_commits": self.accepted_commits,
                "rejected_commits": sum(self.rejected_by_reason.values()),
                "single_file_commits": self.single_file_commits,
                "multi_file_commits": self.multi_file_commits,
            },
            "filtering": {
                "rejected_commits_by_reason": dict(self.rejected_by_reason),
                "sensitive_pattern_counts": dict(self.sensitive_pattern_counts),
                "chunking_error_types": dict(self.chunking_error_types),
            },
            "distributions": {name: stats.summary() for name, stats in self.statistics.items()},
            "chunking": {
                "total_generated_chunks": self.total_generated_chunks,
                "maximum_chunk_tokens_observed": (self.max_chunk_tokens_observed),
                "token_budget_violations": 0,
                "targets_over_max_length": (self.targets_over_max_length),
            },
            "categories": {
                "languages": dict(self.language_counts.most_common(20)),
                "licenses": dict(self.license_counts.most_common(20)),
                "change_types": dict(self.change_type_counts),
            },
        }


def find_sensitive_patterns(
    record: CommitRecord,
) -> tuple[str, ...]:
    """Detect sensitive patterns without retaining matched values."""

    texts = [
        record.target_message,
        *(text for section in record.file_sections for text in (section.path, section.diff)),
    ]

    return tuple(
        name
        for name, pattern in SENSITIVE_PATTERNS.items()
        if any(pattern.search(text) for text in texts)
    )


def benchmark_real_commit(
    record: CommitRecord,
    tokenizer: TokenizerProtocol,
    *,
    max_source_tokens: int = 512,
    overlap_tokens: int = 32,
    max_target_tokens: int = 128,
) -> dict[str, Any]:
    """Chunk a real commit and validate its token and metadata constraints."""

    started = perf_counter()

    chunks = chunk_commit_record(
        record,
        tokenizer,
        max_source_tokens=max_source_tokens,
        overlap_tokens=overlap_tokens,
    )

    if not chunks:
        raise ValueError("The commit produced no chunks.")

    actual_lengths = [chunk.input_token_count for chunk in chunks]

    if any(length < 1 for length in actual_lengths):
        raise AssertionError("An empty chunk was generated.")

    if max(actual_lengths) > max_source_tokens:
        raise AssertionError("A generated chunk exceeds the token budget.")

    if any(chunk.chunk_count != len(chunks) for chunk in chunks):
        raise AssertionError("Inconsistent chunk count metadata.")

    if [chunk.chunk_index for chunk in chunks] != list(range(len(chunks))):
        raise AssertionError("Chunk indexes are not sequential.")

    file_counts = Counter(chunk.file_section_index for chunk in chunks)

    source_counts = {}

    for chunk in chunks:
        section_index = chunk.file_section_index

        if not 0 <= section_index < len(record.file_sections):
            raise AssertionError("Invalid file section index.")

        section = record.file_sections[section_index]

        if chunk.file_path != section.path:
            raise AssertionError("File path metadata was not preserved.")

        if chunk.change_type != section.change_type:
            raise AssertionError("Change type metadata was not preserved.")

        if chunk.target_message != record.target_message:
            raise AssertionError("Commit target was not preserved.")

        if chunk.repository != record.repository or chunk.commit_hash != record.commit_hash:
            raise AssertionError("Commit metadata was not preserved.")

        previous_count = source_counts.setdefault(
            section_index,
            chunk.source_token_count,
        )

        if previous_count != chunk.source_token_count:
            raise AssertionError("Source token counts are inconsistent.")

    expected_sections = set(range(len(record.file_sections)))

    if set(source_counts) != expected_sections:
        raise AssertionError("Some modified files have no generated chunks.")

    source_tokens_per_file = [source_counts[index] for index in range(len(record.file_sections))]

    chunks_per_file = [file_counts[index] for index in range(len(record.file_sections))]

    target_tokens = len(
        tokenizer.encode(
            record.target_message,
            add_special_tokens=True,
        )
    )

    elapsed = perf_counter() - started

    return {
        "file_count": len(record.file_sections),
        "diff_characters": sum(len(section.diff) for section in record.file_sections),
        "source_tokens_per_file": source_tokens_per_file,
        "total_source_tokens": sum(source_tokens_per_file),
        "generated_chunks": len(chunks),
        "chunks_per_file": chunks_per_file,
        "chunk_input_tokens": actual_lengths,
        "maximum_chunk_tokens": max(actual_lengths),
        "target_tokens": target_tokens,
        "target_over_max_length": target_tokens > max_target_tokens,
        "change_types": [section.change_type for section in record.file_sections],
        "processing_seconds": round(elapsed, 6),
    }
