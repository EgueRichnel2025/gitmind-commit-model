from gitmind_commit_model.data.real_commit_experiment import (
    OnlineStats,
    RealCommitAccumulator,
    benchmark_real_commit,
    find_sensitive_patterns,
)
from gitmind_commit_model.data.records import (
    CommitRecord,
    FileSection,
)


class FakeTokenizer:
    """A deterministic tokenizer for real-commit experiment tests."""

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


def make_record() -> CommitRecord:
    return CommitRecord(
        repository="example/project",
        commit_hash="abc123",
        language="Python",
        license="MIT License",
        target_message="fix(core): improve processing",
        file_sections=(
            FileSection(
                path="src/first.py",
                change_type="MODIFY",
                diff=" ".join(f"first_{index}" for index in range(70)),
            ),
            FileSection(
                path="src/second.py",
                change_type="ADD",
                diff=" ".join(f"second_{index}" for index in range(50)),
            ),
        ),
    )


def test_benchmark_real_commit_respects_token_budget() -> None:
    result = benchmark_real_commit(
        make_record(),
        FakeTokenizer(),
        max_source_tokens=20,
        overlap_tokens=3,
        max_target_tokens=128,
    )

    assert result["file_count"] == 2
    assert result["generated_chunks"] > 2
    assert result["maximum_chunk_tokens"] <= 20
    assert result["total_source_tokens"] == 120
    assert result["target_over_max_length"] is False
    assert len(result["chunks_per_file"]) == 2
    assert result["change_types"] == ["MODIFY", "ADD"]


def test_benchmark_real_commit_preserves_file_section_counts() -> None:
    result = benchmark_real_commit(
        make_record(),
        FakeTokenizer(),
        max_source_tokens=20,
        overlap_tokens=3,
    )

    assert sum(result["chunks_per_file"]) == result["generated_chunks"]
    assert len(result["source_tokens_per_file"]) == 2


def test_sensitive_patterns_are_detected_without_exposing_values() -> None:
    record = CommitRecord(
        repository="example/project",
        commit_hash="abc123",
        language="Python",
        license="MIT License",
        target_message="fix: remove leaked credential",
        file_sections=(
            FileSection(
                path="src/config.py",
                change_type="MODIFY",
                diff=(
                    "+EMAIL = alice@example.com\n+TOKEN = ghp_123456789012345678901234567890123456"
                ),
            ),
        ),
    )

    patterns = find_sensitive_patterns(record)

    assert "email" in patterns
    assert "github_token" in patterns
    assert "alice@example.com" not in patterns
    assert all("ghp_" not in pattern for pattern in patterns)


def test_clean_commit_has_no_sensitive_patterns() -> None:
    assert find_sensitive_patterns(make_record()) == ()


def test_online_stats_have_bounded_memory() -> None:
    stats = OnlineStats(
        reservoir_size=10,
        seed=42,
    )

    for value in range(1_000):
        stats.update(value)

    summary = stats.summary()

    assert summary["count"] == 1_000
    assert summary["min"] == 0
    assert summary["max"] == 999
    assert summary["mean"] == 499.5
    assert summary["reservoir_size"] == 10
    assert summary["quantiles_approximate"] is True


def test_online_stats_resume_preserves_future_results() -> None:
    first = OnlineStats(
        reservoir_size=20,
        seed=42,
    )

    for value in range(500):
        first.update(value)

    restored = OnlineStats.from_state_dict(first.state_dict())

    for value in range(500, 1_000):
        first.update(value)
        restored.update(value)

    assert restored.summary() == first.summary()


def test_accumulator_checkpoint_preserves_statistics() -> None:
    accumulator = RealCommitAccumulator(
        reservoir_size=20,
        seed=42,
    )

    tokenizer = FakeTokenizer()
    record = make_record()

    result = benchmark_real_commit(
        record,
        tokenizer,
        max_source_tokens=20,
        overlap_tokens=3,
    )

    accumulator.rows_examined = 1
    accumulator.add_commit(record, result)

    restored = RealCommitAccumulator.from_state_dict(accumulator.state_dict())

    assert restored.rows_examined == 1
    assert restored.accepted_commits == 1
    assert restored.total_generated_chunks == result["generated_chunks"]

    assert restored.summary() == accumulator.summary()
