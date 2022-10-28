# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module to setup the python package."""
from __future__ import annotations

import os
import re
from pathlib import Path
from string import Template

from setuptools import setup


def replace_markdown_relative_paths(
    path: Path, file_name: str, revision_tag: str
) -> str:
    """Replace relative paths in md file with links to review.mlplatform.org."""
    md_url = Template(
        "https://review.mlplatform.org/plugins/gitiles/ml/mlia/+/refs/tags/$tag/$link"
    )
    img_url = Template("https://git.mlplatform.org/ml/mlia.git/plain/$link?h=$tag")
    md_link_pattern = r"(!?\[.+?\]\((.+?)\))"

    content = path.joinpath(file_name).read_text()
    # Find all md links with these formats: [title](url) or ![title](url)
    for match, link in re.findall(md_link_pattern, content):
        if path.joinpath(link).exists():
            # Choose appropriate url template depending on wheteher the
            # orginal link points to a file or an image.
            template = img_url if match[0] == "!" else md_url
            new_url = template.substitute(tag=revision_tag, link=link)
            md_link = match.replace(link, new_url)
            # Replace existing links with new ones
            content = content.replace(match, md_link)
    if os.getenv("MLIA_DEBUG"):
        # Generate an md file to verify the changes
        path.joinpath("PYPI.md").write_text(content)
    return content


if __name__ == "__main__":
    from setuptools_scm import get_version  # pylint: disable=import-error

    tag = get_version()
    long_description = replace_markdown_relative_paths(Path.cwd(), "README.md", tag)
    setup(long_description=long_description)
