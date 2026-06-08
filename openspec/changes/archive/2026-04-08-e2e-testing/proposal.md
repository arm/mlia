## Why

MLIA repositories need end-to-end tests that exercise real CLI invocations
against prepared artifacts, but the mechanics for case expansion, staging,
backend setup, and subprocess execution are repetitive and easy to implement
inconsistently. A shared helper reduces that duplication and makes e2e suites
behave the same way across repositories.

## What Changes

- Add a shared `mlia.testing.e2e` helper for pytest-native end-to-end tests.
- Define environment-driven execution configuration for commands, artifacts,
  backends, and optional sharding.
- Add reusable case loading, artifact staging, backend installation, and command
  execution helpers.
- Add pytest parametrization helpers for compatibility and performance suites.

## Capabilities

### New Capabilities
- `e2e-testing`: Defines the shared MLIA pytest e2e execution model, including
  environment-driven case discovery, artifact staging, sharding, and per-case
  command execution.

### Modified Capabilities

## Impact

- Affects `mlia.testing.e2e` and the repository's e2e tests.
- Establishes a consistent e2e execution model across MLIA repositories.
- Moves suite mechanics into shared code while keeping assertions in consuming
  test modules.
