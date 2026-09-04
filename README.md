<!---
SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# ML Inference Advisor

ML Inference Advisor (MLIA) helps AI developers evaluate model compatibility
and performance for supported inference targets. The core `mlia` package
provides the shared CLI, public Python API, backend management, and standardized
output model. Installed plugins register target, backend, converter, and
post-analysis capabilities.

## Table of Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Plugin model](#plugin-model)
- [Quick start](#quick-start)
- [Python API](#python-api)
- [Reporting bugs](#reporting-bugs)
- [Configuration](#configuration)
- [Development](#development)
- [Getting support](#getting-support)
- [Reporting vulnerabilities](#reporting-vulnerabilities)
- [License](#license)
- [Trademarks and copyrights](#trademarks-and-copyrights)

## Inclusive language commitment

This product conforms to Arm's inclusive language policy and, to the best of
our knowledge, does not contain any non-inclusive language.

If you find something that concerns you, email <terms@arm.com>.

## Releases

Latest changes and release history can be found in [MLIA releases](https://github.com/arm/mlia/releases).

## Documentation

Structured repository documentation lives in [docs/README.md](docs/README.md).
Use the core docs for:

- Shared CLI guidance: [docs/source/cli.md](docs/source/cli.md).
- Backend discovery and installation:
  [docs/source/backends.md](docs/source/backends.md).
- Output structure and JSON results:
  [docs/source/metrics.md](docs/source/metrics.md).
- Architecture and package boundaries:
  [docs/source/overview.md](docs/source/overview.md),
  [docs/source/high_level_architecture.md](docs/source/high_level_architecture.md),
  and [docs/source/execution_flow.md](docs/source/execution_flow.md).

Target-specific, backend-specific, and converter-specific detail belongs in the
documentation for the plugin package that owns that functionality.

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

Install the target, backend, and converter plugins required for your workflow.
Use the owning plugin documentation for package names and any additional setup.

## Plugin model

`mlia` is the core package. Targets, backends, and converters are provided
through separate plugin wheels.

MLIA uses the following plugin model:

- `mlia` provides the command-line experience and shared output model.
- Plugin packages add target definitions, backend implementations, and converter
  paths.
- MLIA discovers those plugins at runtime and exposes them through the same CLI.

Install only the plugin packages you need, then use the discovery commands to
see what is available in the current environment:

```bash
mlia target list
mlia backend list
```

## Quick start

Check that MLIA is installed correctly:

```bash
mlia --help
```

A typical run looks like this:

```bash
mlia check my_model.tflite --target-profile <target-profile> --performance
```

Useful discovery commands:

```bash
mlia target list
mlia backend list
mlia check --help
```

Use custom target profiles by passing a TOML file path:

```bash
mlia check my_model.tflite --target-profile ./my_target_profile.toml
```

If you are new to the plugin-based model, the safest first pattern is:

1. Install `mlia` and the plugin packages you need.
2. Confirm target and backend discovery with `mlia target list` and
   `mlia backend list`.
3. Run one simple `mlia check` command, then add backend-specific options as
   needed.

## Python API

MLIA also provides a Python API for programmatic compatibility and performance
analysis. The main entry point is `run_advisor()`, which mirrors the CLI
`check` workflow and returns standardized output as a Python `dict`.

```python
from pathlib import Path

from mlia import run_advisor

result = run_advisor(
    advice_category="performance",
    target_profile="<target-profile>",
    model=Path("my_model.tflite"),
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

The Python API uses the same installed target and backend plugins as the CLI.

## Reporting bugs

Report bugs by creating a GitHub issue. Use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues) by default,
including when you are not sure which repo owns the problem.

Only file an issue in a plugin repository if the bug is clearly and
specifically isolated to that plugin.

## Configuration

MLIA reads user configuration from the platform-specific configuration
directory provided by `platformdirs`. On Linux this is normally
`~/.config/mlia/config.toml`.

Supported top-level TOML keys are:

- `color` (`bool`): enable or disable colored terminal output.
- `backend_options` (`table`): configuration passed to installed backends.
- `core` (`table`): settings owned by the core application.
- `filtering` (`table`): standardized-output filtering settings. Its optional
  `collapse` value is an array of tables.
- `plugins` (`table`): settings keyed by post-analysis plugin name. Each plugin's
  value is a table owned and validated by that plugin.

Entity collapse can be configured with an array of rules:

```toml
[[filtering.collapse]]
kind = "code_stack"
attribute = "file"
globs = ["*/generated/*", "*/site-packages/*"]
```

Each rule selects an entity kind, a string attribute, and one or more glob
patterns. Core combines these rules with defaults supplied by the backend before
deriving source-line entities and projecting breakdowns.

Post-analysis plugins receive only their own settings table:

```toml
[plugins.example]
output_format = "custom"
```

The plugin registered as `example` receives the contents of `plugins.example`
and is responsible for validating them.

MLIA follows common CLI environment variables where they map cleanly to existing
options:

- `DEBUG`: enable verbose output by default. Passing `--debug` has the same
  effect.
- `NO_COLOR`: disable colored terminal output.
- `MLIA_NO_COLOR`: disable colored terminal output for MLIA specifically.
- `COLUMNS`: set the terminal width used for interactive output. Redirected
  output is rendered wide so the receiving terminal or tool can wrap it.

MLIA can read environment variables from `.env` files in parent directories.

When an option is available through more than one source, MLIA applies values
in this order:

1. Command-line flags.
2. The running shell's environment variables.
3. Project-level environment variables.
4. User-level configuration.
5. Built-in defaults.

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

Wheel builds include a CycloneDX SBOM in the wheel's
`mlia-<version>.dist-info/sboms/` directory.

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
