from gitmind_commit_model.data.chunking import (
    chunk_commit_record,
    split_token_ids,
)
from gitmind_commit_model.data.records import (
    CommitRecord,
    FileSection,
)


class FakeTokenizer:
    """A deterministic tokenizer for unit tests."""

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
        target_message="fix(api): correct user lookup",
        file_sections=(
            FileSection(
                path="src/user.py",
                change_type="MODIFY",
                diff=" ".join(f"token{i}" for i in range(100)),
            ),
        ),
    )


def test_split_token_ids_preserves_all_tokens() -> None:
    tokens = list(range(100))

    chunks = split_token_ids(
        tokens,
        max_tokens=20,
        overlap_tokens=5,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)

    covered = {token for chunk in chunks for token in chunk}

    assert covered == set(tokens)


def test_split_token_ids_rejects_invalid_overlap() -> None:
    try:
        split_token_ids(
            [1, 2, 3],
            max_tokens=5,
            overlap_tokens=5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid overlap should raise ValueError.")


def test_chunk_commit_record_respects_token_budget() -> None:
    tokenizer = FakeTokenizer()

    chunks = chunk_commit_record(
        make_record(),
        tokenizer,
        max_source_tokens=20,
        overlap_tokens=3,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        actual_length = len(
            tokenizer.encode(
                chunk.input_text,
                add_special_tokens=True,
            )
        )

        assert actual_length <= 20
        assert chunk.input_token_count == actual_length


def test_chunks_keep_commit_metadata_and_target() -> None:
    tokenizer = FakeTokenizer()

    chunks = chunk_commit_record(
        make_record(),
        tokenizer,
        max_source_tokens=20,
        overlap_tokens=3,
    )

    for chunk in chunks:
        assert chunk.repository == "example/project"
        assert chunk.commit_hash == "abc123"
        assert chunk.target_message == "fix(api): correct user lookup"
        assert chunk.file_path == "src/user.py"

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)
