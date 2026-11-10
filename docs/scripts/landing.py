# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Render the shared landing content with MkDocs Material."""

import json
import re
from pathlib import Path
from shutil import copyfile
from typing import Any
from urllib.parse import urlsplit

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.nav import Navigation
from mkdocs.structure.pages import Page
from mkdocs.utils import get_relative_url


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Keep the existing Markdown homepage available as documentation."""
    source = Path(config.docs_dir) / "index.md"
    files.append(File.generated(config, "documentation.md", content=source.read_text()))
    return files


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str:
    """Choose the landing template only for the homepage."""
    if page.file.src_uri == "index.md":
        page.meta["template"] = "landing.html"
        page.meta["hide"] = ["navigation", "toc"]
        return ""
    return markdown


def on_page_context(
    context: dict[str, Any], page: Page, config: MkDocsConfig, nav: Navigation
) -> dict[str, Any]:
    """Expose shared content and links that also work in offline builds."""
    if page.file.src_uri != "index.md":
        return context
    source = Path(config.docs_dir) / "landing-page"
    context["landing"] = json.loads((source / "content.json").read_text())
    context["is_core"] = "capabilities" in context["landing"]

    page_files = {
        item.file.src_uri.removesuffix(".md"): item.file for item in nav.pages
    }
    page_files[""] = page.file

    def landing_url(url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            return url
        target = page_files[parts.path.strip("/")]
        relative = str(get_relative_url(target.url, page.file.url))
        return relative + ("#" + parts.fragment if parts.fragment else "")

    context["landing_url"] = landing_url
    return context


def _prefix_selector(match: re.Match[str]) -> str:
    """Scope CSS module selectors without rewriting declaration values."""
    selector = match.group()
    if selector.lstrip().startswith("@"):
        return selector

    def prefix_class(token: re.Match[str]) -> str:
        name = token.group(1)
        return ".mlia-" + name if name else token.group()

    return re.sub(
        r"/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\.([a-zA-Z_][\w-]*)",
        prefix_class,
        selector,
        flags=re.DOTALL,
    )


def on_post_build(config: MkDocsConfig) -> None:
    """Publish logos and a scoped version of the shared landing stylesheet."""
    source = Path(config.docs_dir) / "landing-page"
    output = Path(config.site_dir) / "assets" / "mlia"
    output.mkdir(parents=True, exist_ok=True)
    for image in source.glob("mlia-logo-stacked-*.png"):
        copyfile(image, output / image.name)
    css = (source / "index.module.css").read_text()
    css = css.replace(
        ':global(html[data-theme="dark"])', '[data-md-color-scheme="slate"]'
    )
    css = re.sub(r"[^{}]+(?=\{)", _prefix_selector, css)
    overrides = Path(__file__).resolve().parents[1] / "overrides" / "landing.css"
    css += overrides.read_text()
    (output / "landing.css").write_text(css)
