<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development Notes

## Building docs

From the repository root:

```bash
uv sync --group docs
uv run mkdocs build --strict
```

For local preview:

```bash
uv run mkdocs serve
```

## Keeping docs aligned

When adding or updating documentation in this repository:

- Keep user-facing installation and CLI material aligned with `README.md`
- Add navigable Markdown pages here for repository-level concepts that outgrow
  the README
- Link plugin-specific details from the plugin repositories rather than copying
  them into the core repo
- Update `mkdocs.yml` whenever you add or rename a page so navigation stays in
  sync

## GitHub readiness

The docs workflow should validate that the MkDocs site builds cleanly on pull
requests. Keep the docs tree Markdown-first so GitHub browsing and generated
site output stay aligned.

## Suggested future additions

As the repo evolves, this documentation set can be expanded with pages covering:

- Backend management internals.
- Plugin discovery lifecycle.
- Python API patterns.
- Output schema and reporting concepts.
