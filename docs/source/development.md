<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development Notes

## Building documentation

From the repository root:

```bash
uv sync --no-install-project --only-group docs
uv run mkdocs build --strict
```

For local preview:

```bash
uv run mkdocs serve
```

## Running tests

Install the development dependencies and run the relevant test scope:

```bash
uv sync --group dev
uv run pytest --no-success-flaky-report -m "not slow" tests/
uv run pytest --no-success-flaky-report tests/
```

Entity graph and projection changes should include behavioural tests against the
runtime implementation. Relevant coverage lives in
`tests/test_core_entity_collapse.py` and
`tests/test_core_output_projection.py`.

## Keeping documentation aligned

When changing this repository:

- Update the architecture and execution-flow pages when workflow ownership or
  output ordering changes.
- Document public CLI and Python API parameters in their respective pages.
- Update [Outputs](metrics.md) when the standardized schema, entity semantics,
  metric rules, or post-processing behaviour changes.
- Update [Plugin Architecture](plugin_architecture.md) when adding an entry-point
  group or changing a plugin contract.
- Keep user-facing installation and CLI material aligned with `README.md`.
- Link plugin-specific details rather than copying them into the core docs.
- Update `mkdocs.yml` whenever a page is added or renamed.

## Extending standardized output

Schema changes should keep the Python schema classes, JSON schema resources,
validation, rendering, and tests aligned. Entity-producing plugins should use
canonical identities where core defines them and declare relationships for
backend-specific entity kinds.

Core always performs basic structure and entity-graph validation around output
post-processing. Optional full JSON Schema validation is an additional layer,
not a replacement for those checks.

## Extending MLIA through plugins

New plugin integrations should use the existing target, backend, transformer,
or analysis entry-point categories rather than adding target-specific branches
to core. Analysis plugins should consume completed standardized output; they
should not reconstruct results from console text or internal workflow state.

Do not use the dormant `mlia.plugin.cli` group for top-level command extensions.
Although its loader helper remains in the source tree, it is not connected to
the Typer application, so those entry points are not registered as commands.

See [Plugin Architecture](plugin_architecture.md) for the category boundaries
and configuration model.
