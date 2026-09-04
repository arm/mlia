<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Overview

The `mlia` package provides the shared MLIA CLI, Python API, backend management,
workflow contracts, and standardized output. Installable plugins register
target, backend, transformer, and post-analysis capabilities.

## What lives in this package

Use this package to run MLIA, discover installed targets and backends, integrate
MLIA through Python, or consume standardized JSON output.

Use the owning plugin documentation for the detailed behaviour of a specific
target, backend family, or model transformation path.

## What this documentation covers

This documentation focuses on the shared MLIA experience: how to run the core
workflows, how results are structured, and how plugins are discovered and used.

For in-depth target-specific, backend-specific, or transformer-specific detail,
see the individual plugin packages.
