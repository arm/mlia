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

## Backend-specific options

The CLI can expose backend-specific options depending on which backends are
installed.

If you are unsure which options are available in the current environment, run:

```bash
mlia check --help
```

This is often the safest way to confirm what a real environment supports before
assuming a plugin-specific example applies to your setup.

## Plugin relationship

The core repo provides the commands and workflow orchestration. Plugin packages
provide target plugins, backend plugins, and transformer plugins that are
installed into that experience.

A useful mental model is:

1. `mlia` is the front door.
2. `mlia target` and `mlia backend` help discover available capabilities.
3. Plugin repos extend what those commands can do.
4. Plugin docs explain the detailed behaviour once you know which plugin path
   your run is using.

## README versus docs

Use `README.md` when you want the broad getting-started path. Use this page when
you want a slightly more guided description of the core CLI responsibilities in
the plugin-based architecture without jumping straight into plugin-specific detail.
