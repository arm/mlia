<!--
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

# `mlia.testing.e2e`

`mlia.testing.e2e` is the shared, opinionated pytest pattern for MLIA end-to-end
tests.

## Model

- Each parametrized case represents one distinct `mlia` CLI invocation.
- Each case is expected to appear only once in the pytest parametrization.
- Because each case is unique, each case should run exactly once.
- There is no single combined MLIA invocation for the whole suite.
- There is no need for a cross-case result cache just to avoid duplicate
  execution.

In practice, the intended shape is:

```python
from mlia.testing import e2e as mlia_e2e
from pathlib import Path


@mlia_e2e.parametrize(mlia_e2e.E2E_PERFORMANCE)
def test_e2e_performance(case: mlia_e2e.E2ECase, tmp_path: Path) -> None:
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    chunky_assertions(result, case)
```

## Shared responsibilities

The shared helper should contain only generic behavior that can be reused across
MLIA repos:

- reading suite definitions from `MLIA_E2E_*` environment variables
- reading backend names from `MLIA_E2E_BACKENDS`
- normalizing suite cases into `E2ECase` objects
- staging a prepared artifact tree into a work directory
- expanding globs relative to that work directory
- running one `mlia` command for one case
- providing small pytest helpers such as `parametrize(...)`
- providing small setup helpers such as `install_requested_backends()` and
  `prepared_artifact_path(...)` for tests that need the shared e2e environment
  but do not run through `run_case(...)`

## Repo responsibilities

Repo-specific test code should stay outside `mlia.testing.e2e`.

That includes:

- target-specific expectations
- backend-specific expectations
- command-specific output assertions that are not generic across MLIA repos

Those assertions should live in the consuming test module so the test flow stays
readable.

## Artifacts and backends

`mlia.testing.e2e` assumes:

- backends are installed globally once, outside per-case execution concerns
- prepared artifacts already exist in a known location before the test runs

The shared helper may stage prepared artifacts into a case work directory, but
it should not own artifact download policy or repo-specific layout rules.
