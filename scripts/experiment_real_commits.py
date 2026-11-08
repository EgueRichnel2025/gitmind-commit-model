"""Run the complete CommitChronicle chunking experiment."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from gitmind_commit_model.data.real_commit_experiment import FastTokenizerAdapter
from gitmind_commit_model.data.records import SUPPORTED_LICENSES
from gitmind_commit_model.data.streaming_experiment import (
    RealCommitAccumulator,
    atomic_write_json,
    consume_stream,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark chunking on real CommitChronicle commits."
    )

    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=None,
    )

    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Optional cumulative limit. Omit to scan the complete partition.",
    )

    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--reservoir-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/real_commits",
    )

    parser.add_argument(
        "--output",
        default="artifacts/real_commits_chunking.json",
    )

    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    data_config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)

    primary = data_config["primary"]
    model_settings = model_config["model"]
    tokenization = model_config["tokenization"]
    chunking = model_config["chunking"]

    dataset_id = primary["repository"]
    dataset_configuration = primary["configuration"]
    available_splits = primary["splits"]

    approved_licenses = set(primary["license_policy"]["approved_values"])

    if approved_licenses != set(SUPPORTED_LICENSES):
        raise ValueError("License configuration differs from the approved licenses in records.py.")

    model_name = args.model or model_settings["pretrained_checkpoint"]

    max_source_tokens = tokenization["source_max_length"]
    max_target_tokens = tokenization["target_max_length"]
    overlap_tokens = chunking["overlap_tokens"]

    seed = args.seed if args.seed is not None else data_config["exploration"]["random_seed"]

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")

    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive.")

    if args.reservoir_size < 1:
        raise ValueError("--reservoir-size must be positive.")

    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        raise ValueError("--max-rows-per-split must be positive.")

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

    except ImportError as error:
        raise SystemExit(
            'Install experiment dependencies with: python -m pip install -e ".[experiment]"'
        ) from error

    print("=" * 72)
    print("GITMIND COMMIT MODEL — REAL COMMIT EXPERIMENT")
    print("=" * 72)

    print("\nDataset :", dataset_id)
    print("Configuration :", dataset_configuration)

    print("\nRésolution de la version du dataset...")

    dataset_info = HfApi().dataset_info(
        repo_id=dataset_id,
        revision=args.revision,
    )

    dataset_revision = dataset_info.sha

    if not dataset_revision:
        raise RuntimeError("Unable to resolve a dataset revision SHA.")

    print("Révision épinglée :", dataset_revision)

    print("\nChargement du tokenizer :", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    tokenizer_limit = tokenizer.model_max_length

    print("Tokenizer :", type(tokenizer).__name__)
    chunk_tokenizer = FastTokenizerAdapter(tokenizer)
    print("Backend rapide :", type(chunk_tokenizer).__name__)
    print("Limite annoncée :", tokenizer_limit)
    print("Budget configuré :", max_source_tokens)
    print("Overlap :", overlap_tokens)

    if tokenizer_limit < max_source_tokens:
        raise ValueError("The configured source budget exceeds the tokenizer limit.")

    selected_splits = args.splits or list(available_splits)

    unknown_splits = set(selected_splits) - set(available_splits)

    if unknown_splits:
        raise ValueError(f"Unknown splits: {sorted(unknown_splits)}")

    report: dict[str, Any] = {
        "experiment": "real_commit_chunking",
        "dataset": dataset_id,
        "dataset_configuration": dataset_configuration,
        "dataset_revision": dataset_revision,
        "datasets_version": version("datasets"),
        "huggingface_hub_version": version("huggingface-hub"),
        "model": model_name,
        "tokenizer": type(tokenizer).__name__,
        "tokenizer_limit": tokenizer_limit,
        "max_source_tokens": max_source_tokens,
        "max_target_tokens": max_target_tokens,
        "chunk_overlap": overlap_tokens,
        "sampling_strategy": "complete_stream_without_shuffle",
        "raw_commit_content_stored": False,
        "selected_splits": selected_splits,
        "max_rows_per_split": args.max_rows_per_split,
        "started_at_utc": utc_now(),
        "splits": {},
    }

    checkpoint_directory = Path(args.checkpoint_dir)
    output_path = Path(args.output)

    all_splits_complete = True

    for split_index, split_name in enumerate(available_splits):
        if split_name not in selected_splits:
            continue

        source_split = available_splits[split_name]
        split_seed = seed + split_index

        print("\n" + "=" * 72)
        print(f"PARTITION : {split_name.upper()}")
        print("=" * 72)

        print("Initialisation du flux Hugging Face...")

        stream = load_dataset(
            dataset_id,
            dataset_configuration,
            split=source_split,
            streaming=True,
            revision=dataset_revision,
        )

        metadata = {
            "dataset": dataset_id,
            "dataset_configuration": dataset_configuration,
            "dataset_revision": dataset_revision,
            "split": split_name,
            "source_split": source_split,
            "model": model_name,
            "max_source_tokens": max_source_tokens,
            "max_target_tokens": max_target_tokens,
            "overlap_tokens": overlap_tokens,
            "reservoir_size": args.reservoir_size,
            "seed": split_seed,
        }

        checkpoint_path = checkpoint_directory / f"{split_name}.checkpoint.json"

        def update_report(
            accumulator: RealCommitAccumulator,
            completed: bool,
            partition_name: str = split_name,
        ) -> None:
            report["splits"][partition_name] = {
                "completed": completed,
                "summary": accumulator.summary(),
            }

            report["updated_at_utc"] = utc_now()
            report["complete"] = all(
                report["splits"].get(name, {}).get("completed", False) for name in selected_splits
            )

            atomic_write_json(output_path, report)

        accumulator, completed = consume_stream(
            stream,
            chunk_tokenizer,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            max_rows=args.max_rows_per_split,
            reservoir_size=args.reservoir_size,
            seed=split_seed,
            max_source_tokens=max_source_tokens,
            overlap_tokens=overlap_tokens,
            max_target_tokens=max_target_tokens,
            on_checkpoint=update_report,
        )

        update_report(accumulator, completed)

        print("\nRésumé de la partition :")
        print(
            json.dumps(
                accumulator.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nPartition terminée :", completed)

        if not completed:
            all_splits_complete = False

    report["complete"] = all_splits_complete
    report["updated_at_utc"] = utc_now()

    atomic_write_json(output_path, report)

    print("\n" + "=" * 72)
    print("EXPÉRIENCE TERMINÉE")
    print("=" * 72)
    print("Toutes les partitions terminées :", all_splits_complete)
    print("Rapport :", output_path)
    print("Checkpoints :", checkpoint_directory)
    print("Aucun diff brut ni message de commit n'a été enregistré.")


if __name__ == "__main__":
    main()
