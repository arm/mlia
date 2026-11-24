# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Provider builds deliver ordinary wheels and retain them in source distributions."""

import shutil
import zipfile
from pathlib import Path

import pytest

from mlia.runtime.build import runtime_resource_files
from mlia.runtime.catalog import RuntimeDefinition


@pytest.fixture
def package_project(tmp_path: Path) -> Path:
    """A standard worker package using the optional MLIA build helper."""
    project = tmp_path / "runtime/worker"
    project.mkdir(parents=True)
    (project.parent / "package_hook.py").write_text(
        "from mlia.runtime.build import RuntimePackageBuildHook\n"
    )
    helpers = tmp_path / "src/mlia/helpers"
    helpers.mkdir(parents=True)
    (helpers / "entry.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from . import types_only\n"
        "def run():\n    from .relative import value\n"
        "    from mlia import extra\n"
    )
    (helpers / "relative.py").write_text("value = 1\n")
    (helpers / "types_only.py").write_text("value = 2\n")
    (tmp_path / "src/mlia/extra.py").write_text("value = 3\n")
    (project / "worker.py").write_text("def run(request): return {}, []\n")
    (project / "pyproject.toml").write_text("""[project]
name = "test-worker"
version = "1.0"
dependencies = ["sample-dependency==2"]
[tool.hatch.build.targets.wheel]
only-include = ["worker.py"]
[tool.hatch.build.hooks.custom]
path = "../package_hook.py"
modules = ["mlia.helpers.entry"]
[tool.mlia.runtime]
python = "3.11"
id = "test/public"
label = "Test runtime"
architectures = ["x86_64", "aarch64"]
[tool.mlia.runtime.operations.echo]
label = "Test operation"
entrypoint = "worker:run"
result_schema = "echo/1"
""")
    return project


@pytest.mark.parametrize("architecture", [None, "aarch64", "x86_64"])
def test_build_and_sdist_rebuild(
    tmp_path: Path, package_project: Path, architecture: str | None
) -> None:
    """Build worker wheels from current code, then rebuild without source or git."""
    # Hatch owns the actual wheel contents and dependency metadata.
    included = runtime_resource_files(
        tmp_path, "test-provider", "1.0", "sdist", architecture=architecture
    )
    manifests = [
        Path(source) for source in included if source.endswith(".runtime.json")
    ]
    assert len(manifests) == (1 if architecture else 2)
    for manifest in manifests:
        definition = RuntimeDefinition.load(manifest)
        definition.verify_sources()
        assert definition.data["package"] == "test-worker==1.0"
        assert definition.data["label"] == "Test runtime"
        assert definition.data["operations"]["echo"]["label"] == "Test operation"
        assert "artifacts" not in definition.data
        assert (
            f"py311-none-linux_{definition.data['architecture']}"
            in definition.data["wheels"][0]["source"]
        )
        with zipfile.ZipFile(
            manifest.parent / definition.data["wheels"][0]["source"]
        ) as stream:
            assert stream.read("worker.py").startswith(b"def run")
            assert stream.read("mlia/helpers/relative.py") == b"value = 1\n"
            assert stream.read("mlia/extra.py") == b"value = 3\n"
            assert "mlia/helpers/entry.py" in stream.namelist()
            assert "mlia/helpers/types_only.py" not in stream.namelist()
            assert "mlia/__init__.py" not in stream.namelist()
            metadata = stream.read("test_worker-1.0.dist-info/METADATA").decode()
            assert "Requires-Dist: sample-dependency==2" in metadata
    extracted = tmp_path / "sdist"
    for source, destination in included.items():
        target = extracted / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (extracted / "PKG-INFO").write_text("test")
    rebuilt = runtime_resource_files(
        extracted, "test-provider", "1.0", "wheel", architecture=architecture
    )
    assert sorted(
        destination.removeprefix("src/") for destination in included.values()
    ) == sorted(rebuilt.values())


@pytest.mark.parametrize(
    "field,value",
    [("id", "../escape"), ("id", "/outside"), ("architectures", "../escape")],
)
def test_build_rejects_escaping_destination(
    tmp_path: Path, package_project: Path, field: str, value: str
) -> None:
    project = package_project / "pyproject.toml"
    data = project.read_text()
    data = (
        data.replace('id = "test/public"', f'id = "{value}"')
        if field == "id"
        else data.replace('["x86_64", "aarch64"]', f'["{value}"]')
    )
    project.write_text(data)
    with pytest.raises(ValueError, match="Invalid runtime"):
        runtime_resource_files(tmp_path, "test-provider", "1.0", "wheel")
    assert not (tmp_path / "build/escape").exists()


def test_worker_release_must_match_provider(
    tmp_path: Path, package_project: Path
) -> None:
    with pytest.raises(ValueError, match="must match provider release"):
        runtime_resource_files(tmp_path, "test-provider", "2.0", "wheel")


@pytest.mark.parametrize(
    "architectures", ['"x86_64"', "[]", '["x86_64", "x86_64"]', "[1]"]
)
def test_build_rejects_malformed_architecture_selection(
    tmp_path: Path, package_project: Path, architectures: str
) -> None:
    project = package_project / "pyproject.toml"
    project.write_text(
        project.read_text().replace('["x86_64", "aarch64"]', architectures)
    )
    with pytest.raises(ValueError, match="architectures"):
        runtime_resource_files(tmp_path, "test-provider", "1.0", "wheel")
    assert not list((tmp_path / "build").rglob("*.whl"))
