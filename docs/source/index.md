---
# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
sidebar_position: 1
slug: /documentation
---

# ML Inference Advisor

Use ML Inference Advisor (MLIA) to check model compatibility and estimate
performance for supported targets. The core package provides the CLI, Python
API, backend management, and standardized results. Plugins add targets,
conversion paths, and visualization.

## Get started

1. [Install MLIA](https://github.com/arm/mlia#installation) and the plugins
   needed for your model and target.
2. [Discover targets and backends](cli.md#a-good-starting-sequence).
3. [Run an analysis](cli.md#common-workflows) and
   [understand the results](metrics.md).

## Find the right guide

| I want to… | Start here |
| --- | --- |
| Understand the core package | [Overview](overview.md) |
| Run MLIA from the terminal | [CLI guide](cli.md) |
| Integrate model analysis in Python | [Python API](api.md) |
| Discover and manage analysis backends | [Backends](backends.md) |
| Read JSON results and metrics | [Outputs](metrics.md) |
| Understand the runtime components | [High-level architecture](high_level_architecture.md) |
| Build or extend a plugin | [Plugin architecture](plugin_architecture.md) |
| Follow an analysis from input to output | [Execution flow](execution_flow.md) |
| Contribute to the project | [Development](development.md) |

## Targets, converters, and visualization

Choose the plugin documentation for target-specific configuration, supported
model formats, and installation requirements.

| Plugin | Documentation |
| --- | --- |
| Neural Technology targets and backends | [Neural Technology](https://developer.arm.com/dev-docs/mlia-neural-technology/) |
| Ethos-U targets, Vela, and Corstone | [Ethos-U](https://developer.arm.com/dev-docs/mlia-ethos-u/) |
| LiteRT model conversion | [LiteRT converters](https://developer.arm.com/dev-docs/mlia-converters-litert/) |
| PyTorch and PTE conversion | [PyTorch converters](https://developer.arm.com/dev-docs/mlia-converters-pytorch/) |
| Model and analysis visualization | [Model Explorer](https://developer.arm.com/dev-docs/mlia-model-explorer/) |
