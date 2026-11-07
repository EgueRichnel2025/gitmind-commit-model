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


def test_primary_dataset_preserves_license_metadata() -> None:
    config = load_data_config()

    assert config["primary"]["fields"]["license"] == "license"


def test_commitbench_is_excluded() -> None:
    config = load_data_config()

    assert config["secondary"]["allow_training"] is False


def test_dataset_exploration_uses_streaming() -> None:
    config = load_data_config()

    assert config["exploration"]["mode"] == "streaming"
