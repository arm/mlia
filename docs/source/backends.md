<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Backend Guide

## Overview

The core `mlia` repository provides backend discovery, capability selection, and
backend management commands. Backend implementations commonly live in split
plugin repositories, which own their detailed installation and metric guidance.

## Backend management commands

Use `mlia backend` to inspect and manage installed backends:

```bash
mlia backend list
mlia backend install <backend>
mlia backend uninstall <backend>
mlia backend --help
```

Depending on the backend, installation may be automatic or may require an
explicit local path, acceptance of a licence, or additional setup.

## Analysis modes

A backend advertises its support separately for two sources of analysis data:

- **Estimation** analyses a model using a tool, simulator, compiler, or service.
- **Profiling** analyses measured data supplied through `--profiling-data` or the
  Python API.

A backend may support estimation, profiling, or both. Its supported advice
categories are evaluated within the selected mode, so a backend being generally
available does not imply that it can satisfy every profiling request.

When profiling data is supplied and `--backend` is omitted, MLIA first considers
profiling-capable default backends for the target. If none are configured, it
considers all profiling-capable target backends. The selected candidate set must
contain exactly one backend; otherwise explicit selection is required. An
estimation-only backend cannot be selected for a profiling-data run.

## Discovering capabilities and options

A practical discovery sequence is:

```bash
mlia target list
mlia backend list
mlia check --help
```

Target discovery lists installed target plugins and profiles. The Python
`list_targets()` helper exposes the backend hierarchy and mode-specific
capabilities. The check help includes dynamically registered options from
installed backends and post-analysis plugins.

Backend-specific CLI options are collected by core and passed to the backend
under its own configuration entry. Use `--backend` when an option or profiling
input needs a particular backend rather than relying on automatic selection.

## Entity-collapse defaults

A backend can provide default rules for collapsing entities in its standardized
output. Core combines those defaults with explicit application
`filtering.collapse` rules before deriving source-line entities and projecting
breakdowns. This lets a backend suppress implementation-detail entities while
keeping the collapse operation and graph validation centralized.

## Backend families

### Analysis backends

Analysis backends produce compatibility or performance results from models,
profiling data, or both.

### Dependency backends

Some backends mainly support larger pipelines by converting models or preparing
intermediate artifacts. They may appear in `mlia backend list` without being a
primary selectable analysis backend.

## Detailed backend documentation

For backend-specific installation, accepted profiling-data formats, metrics,
troubleshooting, and CLI examples, use the plugin repository that owns the
backend.
