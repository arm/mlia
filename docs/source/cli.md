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
- Use `mlia-target` for discovering target profiles.
- Use `mlia-backend` for discovering, installing, and managing backends.

Together, these commands define the basic MLIA flow: discovery first, then
analysis.

## A good starting sequence

If you are in a fresh environment, a practical first sequence is:

```bash
mlia --help
mlia-target list
mlia-backend list
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
mlia-target list
```

List backends:

```bash
mlia-backend list
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

These examples stay intentionally generic. The core repo explains the shared
command shape, while plugin repos explain the target- or backend-specific
variants.

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
provide target plugins, backend plugins, and converter plugins that are
installed into that experience.

A useful mental model is:

1. `mlia` is the front door.
2. `mlia-target` and `mlia-backend` help discover available capabilities.
3. Plugin repos extend what those commands can do.
4. Plugin docs explain the detailed behaviour once you know which plugin path
   your run is using.

## README versus docs

Use `README.md` when you want the broad getting-started path. Use this page when
you want a slightly more guided description of the core CLI responsibilities in
the plugin-based architecture without jumping straight into plugin-specific detail.
