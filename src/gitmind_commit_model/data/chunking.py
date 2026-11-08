"""Token-budget chunking for multi-file Git commits."""

from dataclasses import dataclass
from typing import Protocol

from gitmind_commit_model.data.records import CommitRecord, FileSection


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface required by the chunker."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


@dataclass(frozen=True)
class PreparedChunk:
    """One token-limited piece of a commit."""

    repository: str
    commit_hash: str
    language: str
    license: str

    target_message: str

    file_path: str
    change_type: str
    file_section_index: int
    source_token_count: int

    chunk_index: int
    chunk_count: int

    input_text: str
    input_token_count: int


def split_token_ids(
    token_ids: list[int],
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[list[int]]:
    """Split token IDs into overlapping windows without losing tokens."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    if not token_ids:
        return []

    stride = max_tokens - overlap_tokens
    chunks = []

    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start : start + max_tokens]

        if not chunk:
            break

        chunks.append(chunk)

        if start + max_tokens >= len(token_ids):
            break

    return chunks


def chunk_commit_record(
    record: CommitRecord,
    tokenizer: TokenizerProtocol,
    *,
    max_source_tokens: int = 512,
    overlap_tokens: int = 32,
) -> list[PreparedChunk]:
    """Create file-aware token windows for every modification in a commit.

    Every chunk receives the global commit message as its target.
    This is an experimental weak-supervision strategy.
    """

    if max_source_tokens < 8:
        raise ValueError("max_source_tokens is too small.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    prepared: list[tuple[FileSection, int, int, str, int]] = []

    for file_section_index, section in enumerate(record.file_sections):
        header = f"File: {section.path}\nChange type: {section.change_type}\nDiff:\n"

        header_ids = tokenizer.encode(
            header,
            add_special_tokens=False,
        )

        body_ids = tokenizer.encode(
            section.diff,
            add_special_tokens=False,
        )

        if not body_ids:
            continue

        source_token_count = len(body_ids)

        body_budget = max_source_tokens - len(header_ids) - special_tokens

        if body_budget < 1:
            raise ValueError(f"File header exceeds token budget: {section.path}")

        current_budget = body_budget

        while current_budget >= 1:
            current_overlap = min(
                overlap_tokens,
                current_budget - 1,
            )

            token_windows = split_token_ids(
                body_ids,
                max_tokens=current_budget,
                overlap_tokens=current_overlap,
            )

            rendered = []
            overflow = 0

            for token_window in token_windows:
                body_text = tokenizer.decode(
                    token_window,
                    skip_special_tokens=True,
                )

                input_text = header + body_text

                actual_count = len(
                    tokenizer.encode(
                        input_text,
                        add_special_tokens=True,
                    )
                )

                if actual_count > max_source_tokens:
                    overflow = max(
                        overflow,
                        actual_count - max_source_tokens,
                    )
                    continue

                rendered.append(
                    (
                        input_text,
                        actual_count,
                    )
                )

            if overflow == 0 and len(rendered) == len(token_windows):
                prepared.extend(
                    (
                        section,
                        file_section_index,
                        source_token_count,
                        input_text,
                        token_count,
                    )
                    for input_text, token_count in rendered
                )
                break

            current_budget -= max(overflow, 1)

        else:
            raise ValueError(f"Unable to fit file section into token budget: {section.path}")

    total_chunks = len(prepared)

    return [
        PreparedChunk(
            repository=record.repository,
            commit_hash=record.commit_hash,
            language=record.language,
            license=record.license,
            target_message=record.target_message,
            file_path=section.path,
            change_type=section.change_type,
            file_section_index=file_section_index,
            source_token_count=source_token_count,
            chunk_index=index,
            chunk_count=total_chunks,
            input_text=input_text,
            input_token_count=token_count,
        )
        for index, (
            section,
            file_section_index,
            source_token_count,
            input_text,
            token_count,
        ) in enumerate(prepared)
    ]
