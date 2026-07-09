<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Plugin Architecture

## Why plugins exist

MLIA is designed so that the core package can stay responsible for orchestration
while target- and framework-specific functionality is delivered through plugins.
This makes it possible to evolve support for different hardware families and
conversion stacks.

## Core responsibilities

The core `mlia` package provides:

- The `mlia` CLI, including the `target` and `backend` command groups.
- Plugin discovery and registration.
- Shared workflow abstractions.
- Common reporting and output schema logic.
- Backend installation management.

## Plugin responsibilities

Plugin packages provide one or more of:

- Target plugins.
- Backend plugins.
- Transformer plugins.
- Bundled target profiles and backend resources.

Examples include target-focused plugins, backend-focused plugins, and
transformer plugins that extend the larger MLIA workflow.

See the individual plugin packages for in-depth target-specific,
backend-specific, and transformer-specific documentation.
