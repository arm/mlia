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

from hatch_build import CustomBuildHook, create_sbom, replace_markdown_relative_paths


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


@pytest.mark.parametrize(
    "linked_file_found, file_content, expected_result",
    [
        [
            True,
            "[Test](README.md)",
            "[Test](https://github.com/arm/mlia/blob/0.1.0/README.md)",
        ],
        [
            True,
            "![Test](image.png)",
            "![Test](https://raw.githubusercontent.com/arm/mlia/0.1.0/image.png)",
        ],
        [
            False,
            "[Test](https://github.com/arm/mlia)",
            "[Test](https://github.com/arm/mlia)",
        ],
        [False, "[Test](README.md)", "[Test](README.md)"],
        [
            True,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        ],
    ],
)
def test_replace_markdown_relative_paths(
    linked_file_found: bool,
    file_content: str,
    expected_result: str,
) -> None:
    """Test replacement of relative md paths with links to GitHub."""
    path_mock = Mock()
    tag = "0.1.0"
    path_mock.read_text.return_value = file_content
    path_mock.exists.return_value = linked_file_found
    path_mock.joinpath.return_value = path_mock

    result = replace_markdown_relative_paths(path_mock, "test.md", tag)
    assert result == expected_result
