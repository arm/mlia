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
    model=Path("model.tflite"),
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
`write_output_files=True` with `output_dir` when generated artifacts should be
retained. Supplying `logs_dir` enables per-run file logging; its handlers are
closed when the invocation finishes.

## Lower-level API

`get_advice()` is the lower-level library entry point used by the CLI. It accepts
an optional `ExecutionContext`, model or profiling inputs, backend options, and
resolved application settings. It now returns the advisor's processed
standardized output rather than communicating results through workflow event
handlers.

Most integrations should prefer `run_advisor()` because it owns input
validation, temporary output handling, backend installation, logging capture,
and optional JSON Schema validation.

## Discovery helpers

The public API includes helpers such as:

- `list_targets()`
- `list_target_profiles()`
- `list_backends()`
- `list_backend_options()`
- `supported_backends(target_profile)`

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
