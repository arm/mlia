# SPDX-FileCopyrightText: Copyright 2022,2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hatchling metadata hook used to customize package version and README."""

from __future__ import annotations

import os
import re
from pathlib import Path
from string import Template

from hatchling.metadata.plugin.interface import MetadataHookInterface


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
