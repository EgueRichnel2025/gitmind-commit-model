from gitmind_commit_model.data.records import (
    SUPPORTED_LICENSES,
    build_commit_record,
    build_file_sections,
    is_supported_license,
    normalize_license,
)


def test_normalize_apache_license() -> None:
    assert normalize_license("Apache License 2.0") == ("Apache License 2.0")


def test_normalize_mit_license() -> None:
    assert normalize_license("MIT License") == "MIT License"


def test_supported_licenses_are_recognized() -> None:
    assert len(SUPPORTED_LICENSES) == 3

    for license_name in SUPPORTED_LICENSES:
        assert is_supported_license(license_name)


def test_unknown_license_is_rejected() -> None:
    assert not is_supported_license("Unknown License")


def test_file_sections_extract_non_empty_diffs() -> None:
    modifications = [
        {
            "old_path": "old.py",
            "new_path": "new.py",
            "change_type": "RENAME",
            "diff": "@@ -1 +1 @@\n-old\n+new",
        },
        {
            "old_path": None,
            "new_path": "empty.py",
            "change_type": "ADD",
            "diff": "",
        },
    ]

    sections = build_file_sections(modifications)

    assert len(sections) == 1
    assert sections[0].path == "old.py -> new.py"
    assert sections[0].change_type == "RENAME"


def test_commit_record_uses_processed_message() -> None:
    row = {
        "repo": "example/project",
        "hash": "abc123",
        "language": "Python",
        "license": "MIT License",
        "message": "fix(api): correct user lookup",
        "original_message": (
            "fix(api): correct user lookup\nSigned-off-by: Example <example@email.com>"
        ),
        "mods": [
            {
                "old_path": "user.py",
                "new_path": "user.py",
                "change_type": "MODIFY",
                "diff": "@@ -1 +1 @@\n-old\n+new",
            }
        ],
    }

    record = build_commit_record(row)

    assert record is not None
    assert record.target_message == "fix(api): correct user lookup"
    assert record.repository == "example/project"
    assert len(record.file_sections) == 1


def test_commit_without_supported_license_is_rejected() -> None:
    row = {
        "message": "fix: correct calculation",
        "license": "Unknown License",
        "mods": [
            {
                "new_path": "calculation.py",
                "change_type": "MODIFY",
                "diff": "+return 10",
            }
        ],
    }

    assert build_commit_record(row) is None


def test_commit_without_diff_is_rejected() -> None:
    row = {
        "message": "docs: update documentation",
        "license": "MIT License",
        "mods": [],
    }

    assert build_commit_record(row) is None
