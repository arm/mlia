<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Python API

## Overview

The core `mlia` package exposes a Python API for programmatic compatibility and
performance analysis. This API makes use of plugins identically to the CLI.
This makes it easier to embed MLIA in scripts, notebooks, automation, and
larger Python applications.

## Main entry point

The main entry point is `run_advisor()`, which mirrors the CLI `mlia check`
workflow and returns a standardized Python `dict`.

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

A useful way to think about `run_advisor()` is that it gives you the shared MLIA
workflow without requiring a shell command. You still choose the advice
category, target profile, and model input, and you still receive the same
general result structure that the CLI can emit as JSON.

## Discovery Helpers

The core package also exposes helper functions for discovering what the current
environment supports.

Common helpers include:

- `list_targets()`
- `list_target_profiles()`
- `list_backends()`
- `list_backend_options()`
- `supported_backends(target_profile)`

These helpers are especially useful when a script needs to inspect the current
plugin environment before deciding which target or backend path to use.

## Relationship to plugins

The API surface in core `mlia` stays intentionally generic. The available
results still depend on plugin packages.

That means:

- The core API defines the shared calling pattern.
- Plugin packages extend the available targets and backends.

## Working with results

The result returned by `run_advisor()` follows the same high-level structure as
MLIA JSON output. In practice, that means you can use the API when you want the
same kind of information as the CLI, but you want to inspect or transform it in
Python instead of parsing command output.

A practical pattern is:

1. Call `run_advisor()` with the target profile and advice category you want.
2. Inspect top-level context such as target and backend information.
3. Read the result metrics at the model level first.
4. Move to the plugin repo docs when you need the exact meaning of
   backend-specific fields.

## Cross-links

- See [CLI](cli.md) for the equivalent command-line workflow.
- See [Outputs](metrics.md) for the shared output structure.
- See [Overview](overview.md) for how the core API fits into the split MLIA
  architecture.
