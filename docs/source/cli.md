<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

## Overview

The core `mlia` repository owns the main command-line experience. Even when the
actual analysis is implemented by split plugin repositories, users generally
interact through the CLI entry points provided by this repo.

These CLI docs therefore focus on the shared workflow: discovering available capabilities, selecting a target profile, running analysis, and following results to the component that provides detailed behavior.

## Main commands

The most important entry points are:

- Use `mlia` for model analysis and current workflows.
- Use `mlia target` for discovering target profiles.
- Use `mlia backend` for discovering, installing, and managing backends.

Together, these commands define the basic MLIA flow: discovery first, then
analysis.

## A good starting sequence

If you are in a fresh environment, a practical first sequence is:

```bash
mlia --help
mlia target list
mlia backend list
```

That gives you a quick picture of what the environment can actually do before
you start copying example commands from documentation.

## Common workflows

Show top-level help:

```bash
mlia --help
```

List targets:

```bash
mlia target list
```

List backends:

```bash
mlia backend list
```

Run a compatibility check:

```bash
mlia check model.tflite --target-profile <target-profile> --compatibility
```

Run a performance check:

```bash
mlia check model.tflite --target-profile <target-profile> --performance
```

Request JSON output:

```bash
mlia check model.tflite --target-profile <target-profile> --performance --json
```

Use measured profiling data alongside a source model:

```bash
mlia check model.tflite --target-profile <target-profile> --performance \
  --profiling-data <profiling-data-path> --json --out-dir <report-dir>
```

The model is optional when the profiling data can be analyzed independently:

```bash
mlia check --target-profile <target-profile> --performance \
  --profiling-data <profiling-data-path>
```

`--profiling-data` changes the source of the analysis facts; it does not start a
separate reporting workflow. The selected target advisor still produces normal
standardized output, which is rendered and passed to post-analysis plugins in
the same way as model-only analysis.

`--profiling-data` is repeatable and MLIA preserves the supplied order. Core
passes profiling inputs to the selected backend as a list even when only one
path is supplied. Each path may be a file or directory; the selected backend
defines which forms and contents it accepts. Backends advertise whether they can
consume profiling data. Use `--backend` to select one explicitly. When it is omitted,
MLIA selects the target's unique profiling-capable backend. MLIA reports an
error if none is installed, if several require explicit selection, or if an
estimator-only backend is requested.

## Color output

MLIA enables colored CLI output when writing to an interactive terminal.

To disable colors explicitly for `mlia`, `mlia target`, and `mlia backend`, set
the `NO_COLOR` environment variable to any non-empty value before running the
command:

```bash
NO_COLOR=1 mlia check model.tflite --target-profile <target-profile> --performance
```

Color is also disabled automatically when standard output is not a TTY, such as
when redirecting output to a file or piping it to another command.

## Dynamic backend and analysis options

The `check` command can expose additional options from installed backend and
post-analysis plugins. These options are registered before command-line parsing,
so the available command surface depends on the current environment.

Use:

```bash
mlia check --help
```

to inspect the actual backend and post-analysis options available. Backend
option values are passed to their owning backend. Analysis-plugin options decide
whether the plugin runs after successful analysis.

The CLI first renders the processed standardized output and then invokes enabled
post-analysis plugins with that same output, the execution context, resolved
command parameters, and plugin-scoped configuration. These plugins extend what
happens after analysis; they do not replace target collectors or analyzers.

## Configuration

MLIA supports `core`, `filtering`, `backend_options`, and plugin-scoped settings
in addition to the top-level color setting. For example:

```toml
[[filtering.collapse]]
kind = "code_stack"
attribute = "file"
globs = ["*/generated/*"]

[plugins.example]
output_format = "custom"
```

Collapse rules are applied during standardized-output post-processing. A plugin
registered as `example` receives only the `plugins.example` table.

## Plugin relationship

The core repo provides commands, workflow orchestration, standardized-output
processing, and rendering. Plugin packages can provide targets, backends,
transformers, CLI commands, and post-analysis actions.

A useful mental model is:

1. `mlia` is the front door.
2. `mlia target` and `mlia backend` expose installed capabilities.
3. Target and backend plugins determine how analysis runs.
4. Core validates, post-processes, and renders their standardized output.
5. Enabled analysis plugins consume the completed result.
6. Plugin docs explain the detailed behaviour owned by each plugin package.

## README versus docs

Use `README.md` when you want the broad getting-started path. Use this page when
you want a slightly more guided description of the core CLI responsibilities in
the plugin-based architecture without jumping straight into plugin-specific detail.
