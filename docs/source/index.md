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

Use Python 3.10 or newer on Ubuntu 22.04 LTS or a compatible Linux environment.
Install `libpython3.10-dev` if your environment requires it. We recommend a
virtual environment.

Install a target plugin for your model and target. It installs a compatible
version of MLIA as a dependency. For example, for Arm Ethos-U:

```bash
pip install mlia-ethos-u
```

If you also need PyTorch model conversion, install the converter alongside the
target plugin:

```bash
pip install mlia-ethos-u mlia-converters-pytorch
```

If you do not need a target plugin, install the core package with
`pip install mlia`. See the [plugin guides](#targets-converters-and-visualization)
for other targets, model formats, and additional setup.

Check your installation with `mlia --help`, then
[discover targets and backends](cli.md#a-good-starting-sequence). When you have
chosen a target, [run an analysis](cli.md#common-workflows) and
[understand the results](metrics.md).

## Find the right guide

| I want to… | Start here |
| --- | --- |
| Understand the core package | [Overview](overview.md) |
| Run MLIA from the terminal | [CLI guide](cli.md) |
| Integrate model analysis in Python | [Python API](api.md) |
| Discover and manage analysis backends | [Backends](backends.md) |
| Read JSON results and metrics | [Outputs](metrics.md) |
| Follow a guided tutorial | [Learning Paths](learning-paths.md) |
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
