from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_data_config() -> dict:
    config_path = ROOT / "configs" / "data.yaml"
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_primary_dataset_is_commit_chronicle() -> None:
    config = load_data_config()
    assert config["primary"]["repository"] == ("JetBrains-Research/commit-chronicle")


def test_primary_dataset_requires_license_review() -> None:
    config = load_data_config()
    policy = config["primary"]["license_policy"]

    assert policy["require_review"] is True
    assert policy["allow_training_before_review"] is False


def test_commitbench_is_allowed_for_noncommercial_training() -> None:
    config = load_data_config()
    dataset = config["secondary"]

    assert dataset["allow_training"] is True
    assert dataset["usage_policy"] == "non-commercial-only"
    assert dataset["attribution_required"] is True
    assert dataset["commercial_use_allowed"] is False
    assert dataset["trained_weights_release_review_required"] is True


def test_dataset_exploration_uses_streaming() -> None:
    config = load_data_config()
    assert config["exploration"]["mode"] == "streaming"
