from pathlib import Path
from typing import Any

import pytest

from gitmind_commit_model.data.real_commit_experiment import (
    RealCommitAccumulator,
)
from gitmind_commit_model.data.streaming_experiment import (
    consume_stream,
    process_commit_row,
)


class FakeTokenizer:
    """Deterministic tokenizer for streaming integration tests."""

    def __init__(self) -> None:
        self.vocabulary = {}
        self.reverse_vocabulary = {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        tokens = []

        for word in text.split():
            if word not in self.vocabulary:
                token_id = len(self.vocabulary) + 10
                self.vocabulary[word] = token_id
                self.reverse_vocabulary[token_id] = word

            tokens.append(self.vocabulary[word])

        if add_special_tokens:
            return [1, *tokens, 2]

        return tokens

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return " ".join(
            self.reverse_vocabulary[token_id]
            for token_id in token_ids
            if token_id in self.reverse_vocabulary
        )

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2


class FakeStream:
    """Small iterable dataset with resumable state."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.cursor = 0

    def __iter__(self):
        while self.cursor < len(self.rows):
            row = self.rows[self.cursor]
            self.cursor += 1
            yield row

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cursor = state["cursor"]


def make_row(index: int) -> dict[str, Any]:
    return {
        "repo": "example/project",
        "hash": f"commit-{index}",
        "language": "Python",
        "license": "MIT License",
        "message": f"fix(core): improve calculation {index}",
        "original_message": "unused original message",
        "mods": [
            {
                "old_path": "src/calculation.py",
                "new_path": "src/calculation.py",
                "change_type": "MODIFY",
                "diff": (f"@@ -1 +1 @@\n-old_value_{index}\n+new_value_{index}"),
            }
        ],
    }


def test_process_commit_row_accepts_valid_commit() -> None:
    accumulator = RealCommitAccumulator(seed=42)

    status = process_commit_row(
        make_row(1),
        FakeTokenizer(),
        accumulator,
    )

    assert status == "accepted"
    assert accumulator.rows_examined == 1
    assert accumulator.accepted_commits == 1


def test_process_commit_row_rejects_unsupported_license() -> None:
    row = make_row(1)
    row["license"] = "Unknown License"

    accumulator = RealCommitAccumulator(seed=42)

    status = process_commit_row(
        row,
        FakeTokenizer(),
        accumulator,
    )

    assert status == "rejected"
    assert accumulator.accepted_commits == 0
    assert accumulator.rejected_by_reason["unsupported_license"] == 1


def test_process_commit_row_rejects_sensitive_content() -> None:
    row = make_row(1)
    row["mods"][0]["diff"] += "\n+CONTACT = alice@example.com"

    accumulator = RealCommitAccumulator(seed=42)

    status = process_commit_row(
        row,
        FakeTokenizer(),
        accumulator,
    )

    assert status == "rejected"
    assert accumulator.rejected_by_reason["potential_sensitive_content"] == 1
    assert accumulator.sensitive_pattern_counts["email"] == 1


def test_stream_checkpoint_resumes_without_duplicate_processing(
    tmp_path: Path,
) -> None:
    rows = [make_row(index) for index in range(5)]
    metadata = {
        "dataset": "synthetic/test",
        "revision": "test-revision",
        "split": "train",
    }

    checkpoint_path = tmp_path / "train.checkpoint.json"

    first_accumulator, first_complete = consume_stream(
        FakeStream(rows),
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=2,
        progress_every=100,
        max_rows=3,
        seed=42,
    )

    assert first_complete is False
    assert first_accumulator.rows_examined == 3
    assert first_accumulator.accepted_commits == 3
    assert checkpoint_path.exists()

    second_stream = FakeStream(rows)

    final_accumulator, final_complete = consume_stream(
        second_stream,
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=2,
        progress_every=100,
        seed=42,
    )

    assert final_complete is True
    assert final_accumulator.rows_examined == 5
    assert final_accumulator.accepted_commits == 5
    assert second_stream.cursor == 5


def test_completed_checkpoint_is_not_processed_twice(
    tmp_path: Path,
) -> None:
    rows = [make_row(index) for index in range(3)]
    metadata = {"dataset": "synthetic/test", "split": "train"}
    checkpoint_path = tmp_path / "complete.json"

    consume_stream(
        FakeStream(rows),
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=2,
        progress_every=100,
        seed=42,
    )

    fresh_stream = FakeStream(rows)

    accumulator, completed = consume_stream(
        fresh_stream,
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=2,
        progress_every=100,
        seed=42,
    )

    assert completed is True
    assert accumulator.accepted_commits == 3
    assert fresh_stream.cursor == 0


def test_checkpoint_rejects_different_experiment_metadata(
    tmp_path: Path,
) -> None:
    rows = [make_row(index) for index in range(2)]
    checkpoint_path = tmp_path / "checkpoint.json"

    consume_stream(
        FakeStream(rows),
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata={"revision": "abc"},
        checkpoint_every=1,
        progress_every=100,
    )

    with pytest.raises(ValueError, match="configuration"):
        consume_stream(
            FakeStream(rows),
            FakeTokenizer(),
            checkpoint_path=checkpoint_path,
            metadata={"revision": "different"},
            checkpoint_every=1,
            progress_every=100,
        )


def test_huggingface_stream_supports_multiple_successive_resumes(
    tmp_path: Path,
) -> None:
    import pytest

    datasets = pytest.importorskip("datasets")
    Dataset = datasets.Dataset

    rows = []

    for index in range(6):
        row = make_row(index)
        row["message"] = "fix(core): " + "update " * (index + 1) + "calculation"
        rows.append(row)

    def create_stream():
        return Dataset.from_list(rows).to_iterable_dataset(num_shards=2)

    checkpoint_path = tmp_path / "multiple-resumes.json"

    metadata = {
        "dataset": "synthetic/huggingface-multiple-resume",
        "revision": "test-revision",
        "split": "train",
    }

    first, first_complete = consume_stream(
        create_stream(),
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=1,
        progress_every=100,
        max_rows=2,
        seed=42,
    )

    assert first.rows_examined == 2
    assert first.accepted_commits == 2
    assert first_complete is False

    second, second_complete = consume_stream(
        create_stream(),
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=1,
        progress_every=100,
        max_rows=4,
        seed=42,
    )

    assert second.rows_examined == 4
    assert second.accepted_commits == 4
    assert second_complete is False

    third, third_complete = consume_stream(
        create_stream(),
        FakeTokenizer(),
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        checkpoint_every=1,
        progress_every=100,
        seed=42,
    )

    assert third.rows_examined == 6
    assert third.accepted_commits == 6
    assert third_complete is True

    target_stats = third.summary()["distributions"]["target_tokens"]

    assert target_stats["count"] == 6
    assert target_stats["mean"] == 7.5
