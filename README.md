<!---
SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# ML Inference Advisor

ML Inference Advisor (MLIA) helps AI developers evaluate model compatibility
and performance for supported inference targets. The core `mlia` package is the
shared foundation of the wider MLIA ecosystem: it provides the common CLI,
plugin discovery, backend management, reporting model, and Python API used
across the split target, backend, and converter repos.

In practice, that means the core repo is the place to start even when the final
analysis is performed by plugin-owned functionality. The core package gives you
one entry point and one output model, while plugin packages extend what is
available in the environment.

## Table of Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Plugin model](#plugin-model)
- [Quick start](#quick-start)
- [Python API](#python-api)
- [Development](#development)

## Inclusive language commitment

This product conforms to Arm's inclusive language policy and, to the best of
our knowledge, does not contain any non-inclusive language.

If you find something that concerns you, email <terms@arm.com>.

## Releases

Release notes can be found in [MLIA releases](RELEASES.md).

## Documentation

Structured repository documentation lives in [docs/README.md](docs/README.md).
Use the core docs for:

- shared CLI guidance: [docs/source/cli.md](docs/source/cli.md)
- backend discovery and installation model: [docs/source/backends.md](docs/source/backends.md)
- output structure and JSON results: [docs/source/metrics.md](docs/source/metrics.md)
- architecture and repo boundaries:
  [docs/source/overview.md](docs/source/overview.md),
  [docs/source/high_level_architecture.md](docs/source/high_level_architecture.md),
  and [docs/source/execution_flow.md](docs/source/execution_flow.md)

Target-specific, backend-specific, and converter-specific detail belongs in the
plugin repo that owns that functionality.

## Installation

It is recommended to use a virtual environment for MLIA installation.
A typical setup requires:

- Ubuntu 22.04 LTS or another compatible Linux environment
- Python 3.10 or newer
- `libpython3.10-dev` when required by your environment

Install the core package with:

```bash
pip install mlia
```

## Plugin model

`mlia` is the core package. Targets, backends, and converters are provided
through separate plugin wheels.

A useful way to think about the split is:

- `mlia` provides the command-line experience and shared output model
- plugin repos add target definitions, backend implementations, and converter
  paths
- MLIA discovers those plugins at runtime and exposes them through the same CLI

Install only the plugin packages you need, then use the discovery commands to
see what is available in the current environment:

```bash
mlia-target list
mlia-backend list
```

## Quick start

Check that MLIA is installed correctly:

```bash
mlia --help
```

A typical run looks like this:

```bash
mlia check model.tflite --target-profile <target-profile> --performance
```

Useful discovery commands:

```bash
mlia-target list
mlia-backend list
mlia check --help
```

Use custom target profiles by passing a TOML file path:

```bash
mlia check model.tflite --target-profile ./my_target_profile.toml
```

If you are new to the plugin-based model, the safest first pattern is:

1. install `mlia` and the plugin packages you actually need
2. confirm target and backend discovery with `mlia-target list` and
   `mlia-backend list`
3. run one simple `mlia check` command before introducing backend-specific
   options

## Python API

MLIA also provides a Python API for programmatic compatibility and performance
analysis. The main entry point is `run_advisor()`, which mirrors the CLI
`check` workflow and returns standardized output as a Python `dict`.

```python
from mlia import run_advisor

result = run_advisor(
    advice_category="performance",
    target_profile="<target-profile>",
    model="model.tflite",
)

print(result["schema_version"])
print(result["results"])
```

Other public helpers include:

- `list_targets()`
- `list_target_profiles()`
- `list_backends()`
- `list_backend_options()`
- `supported_backends(target_profile)`

If you need `torch.nn.Module` inputs, install the optional extra:

```bash
pip install mlia[torch]
```

The Python API follows the same plugin-based architecture as the CLI: the core
package provides the entry points and shared output structure, while installed
plugins extend what targets and backends are available.

## Development

Install `uv`, then sync dependencies for local development:

```bash
uv sync --group dev
```

Common commands:

```bash
uv run pre-commit run --all-files --hook-stage=push
uv run pytest --no-success-flaky-report -m "not slow" tests/
uv run pytest --no-success-flaky-report tests/
uv build
```

## Getting support

In case you need support or want to report an issue, give us feedback or simply
ask a question about MLIA, please send an email to <mlia@arm.com>.

Alternatively, use the
[AI and ML forum](https://community.arm.com/support-forums/f/ai-and-ml-forum)
to get support by marking your post with the **MLIA** tag, or tag the @mlia
team directly for assistance.

## Reporting vulnerabilities

Information on reporting security issues can be found in
[Reporting vulnerabilities](SECURITY.md).

## License

ML Inference Advisor is licensed under [Apache License 2.0](LICENSES/Apache-2.0.txt)
unless otherwise indicated. This project contains software under a range of
permissive licenses, see [LICENSES](LICENSES/).

## Trademarks and copyrights

- Arm, Arm Ethos-U, Arm Cortex-A, Arm Cortex-M, and Arm Corstone are registered trademarks or trademarks of Arm Limited (or its subsidiaries) in the U.S. and/or elsewhere.
- TensorFlow is a trademark of Google LLC.
- Keras is a trademark of Francois Chollet.
- Linux is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
- Python is a registered trademark of the PSF.
- Ubuntu is a registered trademark of Canonical.
- Microsoft and Windows are trademarks of the Microsoft group of companies.
