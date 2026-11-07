"""Commit record normalization and file-level diff extraction."""

from dataclasses import dataclass
from typing import Any

LICENSE_ALIASES = {
    "apache license 2.0": "Apache License 2.0",
    "mit license": "MIT License",
    "bsd 3-clause new or revised license": "BSD 3-Clause New or Revised License",
}

SUPPORTED_LICENSES = frozenset(LICENSE_ALIASES.values())


@dataclass(frozen=True)
class FileSection:
    """A single file modification extracted from a Git commit."""

    path: str
    change_type: str
    diff: str


@dataclass(frozen=True)
class CommitRecord:
    """A normalized commit ready for token-based preprocessing."""

    repository: str
    commit_hash: str
    language: str
    license: str
    target_message: str
    file_sections: tuple[FileSection, ...]


def normalize_license(value: Any) -> str:
    """Normalize known dataset license labels."""
    raw_value = str(value or "").strip()
    normalized = " ".join(raw_value.casefold().split())

    return LICENSE_ALIASES.get(normalized, raw_value)


def is_supported_license(value: Any) -> bool:
    """Check whether a dataset example uses an accepted license."""
    return normalize_license(value) in SUPPORTED_LICENSES


def build_file_sections(
    modifications: Any,
) -> tuple[FileSection, ...]:
    """Extract non-empty file diffs from a commit."""
    if not isinstance(modifications, (list, tuple)):
        return ()

    sections = []

    for modification in modifications:
        if not isinstance(modification, dict):
            continue

        diff = str(modification.get("diff") or "").strip("\n")

        if not diff.strip():
            continue

        old_path = str(modification.get("old_path") or "").strip()
        new_path = str(modification.get("new_path") or "").strip()

        if old_path and new_path and old_path != new_path:
            path = f"{old_path} -> {new_path}"
        else:
            path = new_path or old_path or "unknown"

        change_type = str(modification.get("change_type") or "UNKNOWN").upper()

        sections.append(
            FileSection(
                path=path,
                change_type=change_type,
                diff=diff,
            )
        )

    return tuple(sections)


def build_commit_record(
    row: dict[str, Any],
    *,
    require_supported_license: bool = True,
) -> CommitRecord | None:
    """Build a normalized training record from a dataset row.

    The processed `message` field is the training target.
    `original_message` is intentionally not used as a target.
    """
    if not isinstance(row, dict):
        return None

    target_message = str(row.get("message") or "").strip()

    if not target_message:
        return None

    license_name = normalize_license(row.get("license"))

    if require_supported_license and license_name not in SUPPORTED_LICENSES:
        return None

    file_sections = build_file_sections(row.get("mods"))

    if not file_sections:
        return None

    return CommitRecord(
        repository=str(row.get("repo") or "").strip(),
        commit_hash=str(row.get("hash") or "").strip(),
        language=str(row.get("language") or "unknown").strip(),
        license=license_name,
        target_message=target_message,
        file_sections=file_sections,
    )
