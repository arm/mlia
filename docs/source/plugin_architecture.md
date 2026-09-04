<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Plugin Architecture

## Why plugins exist

MLIA keeps shared orchestration, validation, standardized output, and user
experience in the core package while target- and framework-specific behaviour
is delivered through plugins. This allows hardware support, backend tooling,
model conversion, and optional downstream analysis to evolve independently.

## Core responsibilities

The core `mlia` package provides:

- The `mlia` CLI, including the `target` and `backend` command groups.
- Plugin discovery, compatibility checking, and registration.
- Shared workflow abstractions and execution context.
- Standardized output collection, post-processing, validation, and rendering.
- Backend discovery and installation management.
- Dynamic registration of backend and post-analysis CLI options.

## Plugin categories

Plugin packages may provide one or more of the following:

| Category | Entry-point role | Responsibility |
| --- | --- | --- |
| Target | Register target metadata and advisor factories | Define supported profiles and construct target-specific workflows. |
| Backend | Register backend configuration | Describe installation, CLI options, supported advice, estimation/profiling modes, and optional default collapse rules. |
| Transformer | Register model transformations | Convert or prepare model artifacts for another workflow stage. |
| Analysis | Register post-analysis plugins | Add dynamic `mlia check` options and consume completed standardized output. |

Bundled target profiles and backend resources may accompany these plugin
registrations.

The source tree still defines the `mlia.plugin.cli` entry-point group and a
`load_cli_plugins()` helper, but the Typer application does not invoke that
loader or register plugin-provided commands. Installing an entry point in this
group therefore does not extend the `mlia` command tree. Top-level commands are
currently core-owned, and CLI command plugins are not a supported extension
category.

## Post-analysis plugins

Analysis plugins run after a successful `mlia check` analysis. They do not add
collectors or analyzers to the target workflow and do not own the canonical
result. Instead, each enabled plugin receives:

- the processed standardized output, when one was produced;
- the `ExecutionContext` and output directory;
- parsed command arguments and resolved command parameters; and
- only the configuration table scoped to that plugin's registered name.

An analysis plugin supplies its CLI options before command parsing, decides
whether it is enabled from the parsed option values, and implements its
post-analysis action. Installed analysis-plugin options therefore appear in:

```bash
mlia check --help
```

Analysis plugins use the `mlia.plugin.analysis` entry-point group and currently
require plugin interface version `0.0.2`.

## Configuration ownership

The user configuration file can contain plugin-owned tables beneath `plugins`:

```toml
[plugins.example]
output_format = "custom"
```

Core passes only `plugins.example` to the plugin registered as `example`.
Plugins are responsible for validating the contents of their own table.

See the individual plugin packages for target-specific, backend-specific,
transformer-specific, and analysis-plugin documentation.
