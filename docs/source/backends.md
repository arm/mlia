<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Backend Guide

## Overview

The core `mlia` repository provides backend discovery and backend-management
commands, while many backend implementations now live in split plugin
repositories.

This page exists so the top-level `README.md` does not need to carry the full
backend catalog and reference detail.

## Backend management commands

Use `mlia-backend` to inspect and manage installed backends:

```bash
mlia-backend list
mlia-backend install <backend>
mlia-backend uninstall <backend>
mlia-backend --help
```

Depending on the backend, installation may be automatic or may require an
explicit local path or extra setup.

## How to discover backend options

Backend-specific CLI options are exposed through the core command-line
experience when the relevant backend is installed.

A practical discovery sequence is:

```bash
mlia-backend list
mlia check --help
```

This lets you see both the installed backend set and any backend-specific
command-line options that MLIA is currently exposing.

## Backend families

### Hardware-specific analysis backends

Some plugin repos provide backends that perform compatibility or performance
analysis for a hardware family or platform family.

### Dependency backends

Some backends exist mainly to support larger MLIA pipelines by converting models
or preparing intermediate artifacts. Users may see them in `mlia-backend list`
without needing to reason about them as the primary analysis backend for a run.

## Where detailed backend docs live now

For backend-specific installation, metrics, troubleshooting, and CLI examples,
use the split plugin repo that owns the backend you are working with.

## Why this split helps

Moving detailed backend reference material out of the core README keeps the
front page shorter and easier to scan, while allowing plugin repos to own the
detail that changes most often.
