<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Documentation

This directory contains the MkDocs content for the `mlia` repository.

## Included pages

- `source/index.md`: documentation landing page
- `source/overview.md`: role of the core repo in the split MLIA ecosystem
- `source/plugin_architecture.md`: how the core repo relates to plugin repos
- `source/cli.md`: command-line entry points and common workflows
- `source/backends.md`: backend discovery, installation, and responsibility boundaries
- `source/metrics.md`: output formats, schema shape, and metrics guidance
- `source/development.md`: docs and development maintenance notes

## Build

Install the documentation dependencies in your environment, then build from the
repository root:

```bash
uv sync --no-install-project --only-group docs
uv run mkdocs build --strict
```

For local preview:

```bash
uv run mkdocs serve
```

The generated site will be written to `.mkdocs/site/`.

## Scope

The top-level `README.md` remains the main user guide for installation and CLI
usage. This docs tree explains the core architecture, CLI, output shape, and
backend-management model.

## Relationship to plugins

Plugin packages own the target-specific, backend-specific, and converter-specific
detail. Keep that material in the plugin repos rather than copying it into the
core repo.
