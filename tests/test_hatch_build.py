# SPDX-FileCopyrightText: Copyright 2022,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Test for module setup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from cyclonedx.schema import OutputFormat, SchemaVersion
from cyclonedx.validation import make_schemabased_validator

from hatch_build import CustomBuildHook, MetadataHook, create_sbom


def test_create_sbom() -> None:
    """Test generation of the package SBOM."""
    sbom = create_sbom(
        name="Example_Package",
        normalized_name="example-package",
        version="1.2.3+local",
        description="Example package",
        license_expression="MIT",
        urls={
            "Homepage": "https://example.com",
            "Repository": "https://example.com/repository.git",
        },
    )

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert "serialNumber" not in sbom
    assert "timestamp" not in sbom["metadata"]
    assert sbom["metadata"]["component"] == {
        "type": "library",
        "bom-ref": "pkg:pypi/example-package@1.2.3%2Blocal",
        "name": "Example_Package",
        "version": "1.2.3+local",
        "purl": "pkg:pypi/example-package@1.2.3%2Blocal",
        "description": "Example package",
        "licenses": [{"license": {"id": "MIT"}}],
        "externalReferences": [
            {
                "type": "vcs",
                "url": "https://example.com/repository.git",
                "comment": "Repository",
            },
            {
                "type": "website",
                "url": "https://example.com",
                "comment": "Homepage",
            },
        ],
    }
    assert "tools" not in sbom["metadata"]
    assert sbom["dependencies"] == [{"ref": "pkg:pypi/example-package@1.2.3%2Blocal"}]
    validator = make_schemabased_validator(OutputFormat.JSON, SchemaVersion.V1_6)
    assert validator.validate_str(json.dumps(sbom)) is None


def test_build_hook_generates_wheel_sbom(tmp_path: Path) -> None:
    """Test that wheel builds receive a generated SBOM file."""
    metadata = Mock()
    metadata.version = "1.2.3"
    metadata.core.raw_name = "example-package"
    metadata.core.name = "example-package"
    metadata.core.description = "Example package"
    metadata.core.license_expression = "MIT"
    metadata.core.urls = {"Homepage": "https://example.com"}
    hook = CustomBuildHook(
        str(tmp_path), {}, Mock(), metadata, str(tmp_path / "dist"), "wheel"
    )
    build_data: dict = {"sbom_files": []}

    hook.initialize("standard", build_data)
    expected_sbom = create_sbom(
        name="example-package",
        normalized_name="example-package",
        version="1.2.3",
        description="Example package",
        license_expression="MIT",
        urls={"Homepage": "https://example.com"},
    )

    sbom_path = Path(build_data["sbom_files"][0])
    assert sbom_path.name == "example-package.cdx.json"
    assert json.loads(sbom_path.read_text(encoding="utf-8")) == expected_sbom

    hook.finalize("standard", build_data, "dist/mlia-1.2.3.whl")
    assert not sbom_path.exists()


def test_metadata_hook_update_uses_commit_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """README metadata links should target the current commit."""
    (tmp_path / "README.md").write_text(
        "[Docs](docs.md)\n![Image](image.png)\n[Section](#section)",
        encoding="utf-8",
    )
    (tmp_path / "docs.md").touch()
    (tmp_path / "image.png").touch()
    monkeypatch.setattr(
        "hatch_build.subprocess.run",
        Mock(
            side_effect=[
                Mock(stdout=f"{tmp_path}\n"),
                Mock(stdout="0123456789abcdef\n"),
            ]
        ),
    )
    hook = MetadataHook(str(tmp_path), {})
    metadata = {"version": "0.1.0"}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": "[Docs](https://github.com/arm/mlia/blob/"
        "0123456789abcdef/docs.md)\n"
        "![Image](https://raw.githubusercontent.com/arm/mlia/"
        "0123456789abcdef/image.png)\n"
        "[Section](https://github.com/arm/mlia/blob/"
        "0123456789abcdef/README.md#section)",
    }


@pytest.mark.parametrize(
    ("version", "revision"),
    [
        ("0.12.2", "v0.12.2"),
        ("0.12.3.dev9+g40a5004", "g40a5004"),
    ],
)
def test_metadata_hook_update_falls_back_to_version_or_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: str,
    revision: str,
) -> None:
    """README links should use the version or its hash without Git metadata."""
    (tmp_path / "README.md").write_text("[Docs](docs.md)", encoding="utf-8")
    (tmp_path / "docs.md").touch()
    monkeypatch.setattr("hatch_build.subprocess.run", Mock(side_effect=OSError))
    hook = MetadataHook(str(tmp_path), {})
    metadata = {"version": version}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": f"[Docs](https://github.com/arm/mlia/blob/{revision}/docs.md)",
    }
