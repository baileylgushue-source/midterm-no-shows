"""Check that model, dependencies and Space settings agree."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read_space_settings():
    """Read the simple key-value settings at the top of the README."""
    readme = ROOT / "README.md"
    assert readme.is_file(), "README.md is missing."

    text = readme.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines and lines[0] == "---", "README must start with ---."

    closing = next(
        (index for index in range(1, len(lines)) if lines[index] == "---"),
        None,
    )
    assert closing is not None, "README settings need a closing ---."

    settings = {}
    for line in lines[1:closing]:
        key, separator, value = line.partition(":")
        if separator:
            settings[key.strip()] = value.strip().strip("\"'")
    return settings


@pytest.mark.parametrize(
    "package",
    ["scikit-learn", "pandas", "joblib"],
)
def test_runtime_package_matches_pinned_version(package):
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(package)}==([^\s#]+)\s*$",
        requirements,
        flags=re.MULTILINE,
    )
    assert match is not None, f"{package} needs an exact version pin."
    assert version(package) == match.group(1)


def test_sklearn_matches_saved_model_version():
    info = json.loads((ROOT / "models" / "model_info.json").read_text(encoding="utf-8"))
    assert version("scikit-learn") == info["scikit_learn"]


def test_gradio_is_not_in_runtime_requirements():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not re.search(
        r"^\s*gradio\b",
        requirements,
        flags=re.MULTILINE | re.IGNORECASE,
    ), "Gradio belongs in requirements-dev.txt; Spaces installs the SDK."


def test_space_uses_gradio():
    settings = read_space_settings()
    assert settings.get("sdk") == "gradio"


def test_space_gradio_version_matches_installed_version():
    settings = read_space_settings()
    assert settings.get("sdk_version") == version("gradio")


def test_space_app_file_exists():
    settings = read_space_settings()
    app_file = settings.get("app_file")
    assert app_file, "README settings must include app_file."
    assert (ROOT / app_file).is_file(), f"App file is missing: {app_file}"


def test_saved_pipeline_exists():
    assert (ROOT / "models" / "no_show.joblib").is_file()
