<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Python API

## Overview

The core `mlia` package exposes a Python API for programmatic compatibility and
performance analysis. It uses the same target and backend plugins as the CLI and
returns MLIA's processed standardized output directly as a JSON-compatible
`dict`.

## Model-based analysis

The main convenience entry point is `run_advisor()`:

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

For a `torch.nn.Module`, provide the module and its example inputs. MLIA exports
a temporary PT2 artifact for analysis. Quantization is available only for this
module-input path.

## Profiling-data analysis

Measured profiling data can be supplied as one path or an ordered sequence:

```python
result = run_advisor(
    advice_category="performance",
    target_profile="<target-profile>",
    profiling_data=["capture-part-1", "capture-part-2"],
    backends=["<profiling-backend>"],
)
```

The model may be omitted when profiling data is present. Paths may identify
files or directories, and MLIA verifies that each path exists before creating
the advisor. Their accepted format is defined by the selected backend.

When `backends` is omitted, MLIA follows the CLI selection rules. It first
considers profiling-capable default backends for the target and, when there are
none, all profiling-capable target backends. The resulting candidate set must
contain exactly one backend. Estimation-only backends cannot satisfy this
request.

## Output and validation

`run_advisor()` returns the standardized output produced by the workflow after
core post-processing. That processing:

1. validates the basic output and entity graph;
2. applies entity-collapse rules;
3. derives source-line entities;
4. projects missing entity breakdowns; and
5. validates the processed output again.

Basic and entity-graph validation is mandatory. The `validation` argument
controls additional JSON Schema validation:

- `"strict"` raises when additional schema validation fails;
- `"warn"` logs the problem and returns the output; and
- `"off"` skips only the additional JSON Schema validation.

API mode does not render MLIA reports to stdout or stderr. Use
`write_output_files=True` with `output_dir` to retain generated artifacts and
write the canonical result to `mlia-output/mlia-output.json`. Supplying
`logs_dir` enables per-run file logging; its handlers are closed when the
invocation finishes.

## Lower-level API

`get_advice()` is the lower-level library entry point used by the CLI. It accepts
an optional `ExecutionContext`, model or profiling inputs, backend options, and
resolved application settings, and returns the advisor's processed standardized
output.

Most integrations should prefer `run_advisor()` because it owns input
validation, temporary output handling, backend installation, logging capture,
and optional JSON Schema validation.

## Prepared checks for interactive frontends

`mlia.check` exposes `CheckRequest`, `CheckSettings`, `CheckResult`, `CheckError`,
`prepare_check()` and `execute_check()`. Use these when an application needs the
check workflow, including persisted output and post-analysis plugins, after a
separate configuration step. `run_advisor()` remains the library convenience API.

`prepare_check()` captures resolved values without running analysis or creating
output. `execute_check()` validates completeness and runs the shared workflow.
A profiling-only request can omit the model. The result distinguishes completed
and skipped checks; execution failures propagate as exceptions, with `CheckError`
carrying an exit status for command callers.

`CheckRequest.output_dir` is the base directory; when omitted, it defaults to
the current working directory. Execution recreates the `mlia-output` directory
inside that base, replacing any previous contents. Choose a fresh base or ask
the user to confirm replacement before executing another check there.
`CheckResult.output_dir` identifies the actual `mlia-output` directory.

Standalone frontends obtain CLI defaults and typed plugin options through the
public `mlia.cli.preparation.prepare_check_defaults()` context manager:

```python
from mlia.cli.preparation import prepare_check_defaults

with prepare_check_defaults() as request:
    # Configure the request and run the frontend inside this context.
    print(request.target_profile)
```

The context retains callback resources until frontend shutdown. Preparation
uses the actual CLI declarations, environment/configuration, conversions and
callbacks. Required plugin options without usable defaults produce an actionable
error. Declarative prompts are suppressed only for standalone preparation;
callbacks that require terminal input are unsupported on that route.

Requests copy MLIA-owned mappings and preserve plugin-owned values, including
live resources. Any process transport restrictions belong to the frontend.
The optional execution `prompt` callback handles backend installation consent;
ordinary CLI installation policy is preserved when no callback is supplied.
The executor closes MLIA-configured logging handlers on every exit.

Use `execute_check(request, output_stream=stream)` to capture rendered output
while retaining MLIA's configured theme and named styles. The caller owns the
stream; execution leaves it open. An explicit `console` takes precedence over
`output_stream`. Without either, rendered output is returned in `CheckResult.report`.

Frontend integrations use these documented entry points and the discovery API.
CLI command callbacks, internal validators, rendering helpers and context
metadata are implementation details, not integration interfaces.

## Discovery helpers

The public API includes helpers such as:

- `list_targets()`
- `list_target_profiles()`
- `list_backends()`
- `list_backend_options()`
- `supported_backends(target_profile)`
- `target_for_profile(name_or_path)`

`list_targets()` includes backend metadata and mode-specific capabilities for
estimation and profiling, allowing applications to select a viable backend
before starting analysis.

## Relationship to plugins

The core API defines the shared invocation and output contracts. Installed
plugins determine the available targets, profiles, transformations, backends,
and detailed result contents. Post-analysis plugins are a CLI extension and are
not automatically invoked by `run_advisor()`.

## Cross-links

- See [CLI](cli.md) for equivalent command-line workflows.
- See [Backends](backends.md) for estimation and profiling capabilities.
- See [Outputs](metrics.md) for schema and entity-processing semantics.
- See [Plugin Architecture](plugin_architecture.md) for extension categories.
