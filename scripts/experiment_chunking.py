"""Benchmark the commit chunking algorithm with the real CodeT5 tokenizer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from gitmind_commit_model.data.chunking import (
    chunk_commit_record,
)
from gitmind_commit_model.data.records import (
    CommitRecord,
    FileSection,
)


def generate_diff(tokenizer: Any, target_tokens: int) -> str:
    """Generate a realistic synthetic diff close to a token target."""

    def render(pair_count: int) -> str:
        lines = ["@@ -1,0 +1,0 @@\n"]

        for index in range(pair_count):
            lines.append(
                f"+    result_{index} = calculate_total(price_{index}, quantity_{index}) + tax\n"
            )
            lines.append(
                f"-    result_{index} = calculate_total(price_{index}, quantity_{index})\n"
            )

        return "".join(lines)

    def count_tokens(text: str) -> int:
        return len(tokenizer.backend_tokenizer.encode(text).ids)

    low = 1
    high = 1

    while count_tokens(render(high)) < target_tokens:
        high *= 2

    while low < high:
        middle = (low + high) // 2

        if count_tokens(render(middle)) < target_tokens:
            low = middle + 1
        else:
            high = middle

    candidates = {max(1, low - 1), low}

    best_count = min(
        candidates,
        key=lambda value: abs(count_tokens(render(value)) - target_tokens),
    )

    return render(best_count)


def build_record(
    tokenizer: Any,
    *,
    repository: str,
    commit_hash: str,
    file_targets: dict[str, int],
) -> CommitRecord:
    """Build a synthetic multi-file commit."""

    sections = []

    for path, target_tokens in file_targets.items():
        sections.append(
            FileSection(
                path=path,
                change_type="MODIFY",
                diff=generate_diff(tokenizer, target_tokens),
            )
        )

    return CommitRecord(
        repository=repository,
        commit_hash=commit_hash,
        language="Python",
        license="MIT License",
        target_message="feat(core): improve calculation processing",
        file_sections=tuple(sections),
    )


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.backend_tokenizer.encode(text).ids)


def benchmark_record(
    tokenizer: Any,
    record: CommitRecord,
    *,
    expected_file_count: int,
) -> dict[str, Any]:
    """Run chunking and validate every generated sequence."""

    source_tokens = {
        section.path: count_tokens(tokenizer, section.diff) for section in record.file_sections
    }

    started = perf_counter()

    chunks = chunk_commit_record(
        record,
        tokenizer,
        max_source_tokens=512,
        overlap_tokens=32,
    )

    elapsed = perf_counter() - started

    if not chunks:
        raise AssertionError("No chunks were generated.")

    actual_lengths = [count_tokens(tokenizer, chunk.input_text) for chunk in chunks]

    if max(actual_lengths) > 512:
        raise AssertionError(f"Token budget exceeded: {max(actual_lengths)}")

    if any(
        length != chunk.input_token_count
        for length, chunk in zip(actual_lengths, chunks, strict=True)
    ):
        raise AssertionError("Stored token counts do not match.")

    if any(chunk.chunk_count != len(chunks) for chunk in chunks):
        raise AssertionError("Inconsistent chunk metadata.")

    if [chunk.chunk_index for chunk in chunks] != list(range(len(chunks))):
        raise AssertionError("Chunk indexes are not sequential.")

    if len({chunk.file_path for chunk in chunks}) != expected_file_count:
        raise AssertionError("Some modified files have no generated chunks.")

    if any(chunk.target_message != record.target_message for chunk in chunks):
        raise AssertionError("Commit target was not preserved.")

    chunks_per_file = Counter(chunk.file_path for chunk in chunks)

    total_source_tokens = sum(source_tokens.values())
    total_window_tokens = sum(
        count_tokens(
            tokenizer,
            chunk.input_text.split("Diff:\n", 1)[-1],
        )
        for chunk in chunks
    )

    return {
        "repository": record.repository,
        "file_count": expected_file_count,
        "source_tokens_per_file": source_tokens,
        "total_source_tokens": total_source_tokens,
        "generated_chunks": len(chunks),
        "chunks_per_file": dict(chunks_per_file),
        "minimum_chunk_tokens": min(actual_lengths),
        "maximum_chunk_tokens": max(actual_lengths),
        "mean_chunk_tokens": round(
            sum(actual_lengths) / len(actual_lengths),
            2,
        ),
        "total_window_body_tokens": total_window_tokens,
        "processing_seconds": round(elapsed, 4),
        "validation": "PASSED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Salesforce/codet5-small",
    )

    parser.add_argument(
        "--output",
        default="artifacts/chunking_experiment.json",
    )

    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("Chargement du tokenizer :", args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    print("Tokenizer :", type(tokenizer).__name__)
    print("Limite du modèle :", tokenizer.model_max_length)

    report = {
        "model": args.model,
        "tokenizer_limit": 512,
        "chunk_overlap": 32,
        "experiments": {},
    }

    single_file_targets = [512, 5_000, 10_000, 20_000]

    for target in single_file_targets:
        print(f"\nExpérience : {target} tokens, un fichier")

        record = build_record(
            tokenizer,
            repository="synthetic/single-file",
            commit_hash=f"single-{target}",
            file_targets={
                "src/calculation.py": target,
            },
        )

        result = benchmark_record(
            tokenizer,
            record,
            expected_file_count=1,
        )

        report["experiments"][f"single_file_{target}"] = result

        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nExpérience : 20 000 tokens, cinq fichiers")

    multi_targets = {f"src/module_{index}.py": 4_000 for index in range(1, 6)}

    record = build_record(
        tokenizer,
        repository="synthetic/multi-file",
        commit_hash="multi-20000",
        file_targets=multi_targets,
    )

    result = benchmark_record(
        tokenizer,
        record,
        expected_file_count=5,
    )

    report["experiments"]["multi_file_20000"] = result

    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nRapport enregistré :", output_path)
    print("Toutes les validations sont réussies.")


if __name__ == "__main__":
    main()
