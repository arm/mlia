# SPDX-FileCopyrightText: Copyright 2022,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hatchling hooks used to customize package metadata and wheel contents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from string import Template
from tempfile import TemporaryDirectory
from typing import Any, cast

from cyclonedx.contrib.license.factories import LicenseFactory
from cyclonedx.model import ExternalReference, ExternalReferenceType, XsUri
from cyclonedx.model.bom import Bom, BomMetaData
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output import make_outputter
from cyclonedx.schema import OutputFormat, SchemaVersion
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface
from packageurl import PackageURL


def _external_reference_type(label: str) -> ExternalReferenceType:
    """Map common project URL labels to CycloneDX reference types."""
    normalized_label = re.sub(r"[^a-z]", "", label.casefold())
    return {
        "documentation": ExternalReferenceType.DOCUMENTATION,
        "homepage": ExternalReferenceType.WEBSITE,
        "issues": ExternalReferenceType.ISSUE_TRACKER,
        "issuetracker": ExternalReferenceType.ISSUE_TRACKER,
        "repository": ExternalReferenceType.SCM,
        "source": ExternalReferenceType.SCM,
    }.get(normalized_label, ExternalReferenceType.OTHER)


def create_sbom(
    *,
    name: str,
    normalized_name: str,
    version: str,
    description: str,
    license_expression: str,
    urls: dict[str, str],
) -> dict[str, Any]:
    """Create a CycloneDX SBOM describing the software contained in the wheel."""
    purl = PackageURL(type="pypi", name=normalized_name, version=version)
    component = Component(
        type=ComponentType.LIBRARY,
        name=name,
        version=version,
        bom_ref=str(purl),
        purl=purl,
        description=description or None,
        licenses=(
            [LicenseFactory().make_from_string(license_expression)]
            if license_expression
            else None
        ),
        external_references=[
            ExternalReference(
                type=_external_reference_type(label),
                url=XsUri(url),
                comment=label,
            )
            for label, url in urls.items()
        ],
    )
    bom = Bom(
        metadata=BomMetaData(
            component=component,
        )
    )

    # The wheel version and contents determine this SBOM, so omit generated
    # identifiers and timestamps to keep identical local builds reproducible.
    cast(Any, bom).serial_number = None
    cast(Any, bom.metadata).timestamp = None

    outputter = make_outputter(
        bom=bom,
        output_format=OutputFormat.JSON,
        schema_version=SchemaVersion.V1_6,
    )
    return cast(dict[str, Any], json.loads(outputter.output_as_string()))


class CustomBuildHook(BuildHookInterface):
    """Generate and add an SBOM to every wheel build."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Generate an artifact-specific SBOM and give it to Hatchling."""
        project = self.metadata.core
        self._sbom_directory = TemporaryDirectory(prefix="mlia-sbom-")
        sbom_path = Path(self._sbom_directory.name) / f"{project.name}.cdx.json"
        sbom_path.write_text(
            json.dumps(
                create_sbom(
                    name=project.raw_name,
                    normalized_name=project.name,
                    version=self.metadata.version,
                    description=project.description,
                    license_expression=project.license_expression,
                    urls=project.urls,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        build_data["sbom_files"].append(str(sbom_path))

    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact_path: str,
    ) -> None:
        """Remove the temporary SBOM after Hatchling has packaged it."""
        self._sbom_directory.cleanup()


def replace_markdown_relative_paths(
    path: Path, file_name: str, revision_tag: str
) -> str:
    """Replace relative paths in md file with links to GitHub."""
    md_url = Template("https://github.com/arm/mlia/blob/$tag/$link")
    img_url = Template("https://raw.githubusercontent.com/arm/mlia/$tag/$link")
    md_link_pattern = r"(!?\[.+?\]\((.+?)\))"

    content = path.joinpath(file_name).read_text()
    for match, link in re.findall(md_link_pattern, content):
        if link.startswith("#") or path.joinpath(link).exists():
            if link.startswith("#"):
                new_url = md_url.substitute(tag=revision_tag, link=file_name + link)
            else:
                template = img_url if match[0] == "!" else md_url
                new_url = template.substitute(tag=revision_tag, link=link)
            md_link = match.replace(link, new_url)
            content = content.replace(match, md_link)
    return content


class MetadataHook(MetadataHookInterface):
    """Rewrite README links and optionally append a version suffix."""

    def update(self, metadata: dict) -> None:
        """Mutate Hatch metadata with rewritten README and optional suffix."""
        root = Path(self.root)
        version = str(metadata.get("version", ""))
        tag = version
        custom_tag_suffix = os.getenv("MLIA_CUSTOM_TAG_SUFFIX")
        if custom_tag_suffix:
            tag = f"{tag}.{custom_tag_suffix}"
            metadata["version"] = tag

        pypi_md = replace_markdown_relative_paths(root, "README.md", tag)
        if os.getenv("MLIA_DEBUG"):
            (root / "PYPI.md").write_text(pypi_md)
        metadata["readme"] = {"content-type": "text/markdown", "text": pypi_md}
