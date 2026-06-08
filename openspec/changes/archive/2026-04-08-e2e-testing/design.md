## Context

`mlia.testing.e2e` is a small shared layer for pytest-based end-to-end tests
across MLIA repositories. Its job is to standardize how e2e cases are defined,
expanded, prepared, and executed, while leaving repo-specific assertions in the
tests that consume it.

## Goals / Non-Goals

**Goals:**

- Provide one concrete-command-per-case execution model for pytest e2e suites.
- Let repositories define suites through environment-driven configuration
  instead of custom harness code.
- Keep shared mechanics in one place while leaving behavioral assertions in
  consuming test modules.

**Non-Goals:**

- Define target-specific expected output patterns.
- Define backend-specific assertions or repository-local test policy.
- Own artifact preparation or download workflows before pytest starts.

## Decisions

### Model one e2e case as one CLI invocation

The helper treats one e2e case as one `mlia` CLI invocation.

That choice drives the rest of the design:

- parametrized pytest cases are concrete commands, not abstract scenarios
- each command should appear only once after expansion
- each case runs exactly once
- there is no suite-wide execution cache for duplicate suppression

This keeps the test model easy to reason about. When a test fails, the failing
pytest case directly corresponds to a single CLI command line.

Alternative considered: treat cases as higher-level scenarios that may map to
multiple invocations.

Why not chosen: it weakens debuggability and makes parametrized pytest cases
less direct.

### Drive the helper entirely from environment variables

The helper is configured through:

- `MLIA_E2E_EXECUTIONS` for command blocks and parameter combinations
- `MLIA_E2E_ARTIFACTS` for the prepared artifact tree
- `MLIA_E2E_BACKENDS` for requested backend installation
- `MLIA_E2E_SHARD_INDEX` and `MLIA_E2E_SHARD_COUNT` for sharding

This keeps the shared layer generic. It does not need repo-specific code to
know which models, targets, or backends a repository wants to exercise.

Alternative considered: encode case definitions directly in Python test modules.

Why not chosen: it would reduce reuse across MLIA repositories and duplicate
configuration logic.

### Expand and validate commands before execution

Case loading works in four steps:

1. Parse `MLIA_E2E_EXECUTIONS` into execution blocks.
2. Compute the Cartesian product of parameter groups inside each block.
3. Expand any `e2e_config/...*` glob arguments relative to the prepared
   artifacts directory.
4. Turn each resolved command into an `E2ECase` and validate it with the MLIA
   CLI parser.

The duplicate-command check happens after expansion. That matters because two
different execution blocks may still resolve to the same final command line.

Alternative considered: defer validation to subprocess execution.

Why not chosen: early validation produces clearer errors and avoids starting
broken commands.

### Stage only referenced artifacts into the case workdir

The helper assumes test artifacts are already prepared before pytest starts. It
does not download models, generate repo-specific layouts, or decide artifact
preparation policy.

Instead, it copies only the prepared-root-relative files referenced by a case
into that case's work directory before running `mlia`. This gives each case an
isolated filesystem view without making the helper responsible for artifact
production.

Alternative considered: let all cases run directly against a shared artifact
tree.

Why not chosen: per-case staging gives cleaner isolation and fewer hidden
couplings between tests.

### Install requested backends outside the per-case model

Backends are installed outside the per-case model. The helper reads
`MLIA_E2E_BACKENDS`, installs those backends once per Python process, and then
reuses that installed set across case executions.

Alternative considered: install backends for every case.

Why not chosen: it would be slower and would mix environment setup with case
execution mechanics.

### Keep assertions out of the shared helper

The shared helper owns reusable mechanics only:

- environment-driven case discovery
- command normalization and validation
- sharding
- artifact staging
- backend installation
- subprocess execution
- pytest parametrization helpers

Consuming repositories own behavior assertions:

- target-specific expectations
- backend-specific expectations
- repo-specific output checks
- test naming and organization

This boundary is deliberate. Once the shared helper starts owning domain
assertions, it stops being reusable across MLIA repositories.

Alternative considered: bundle common output assertions into the helper.

Why not chosen: those checks quickly become repo- and target-specific.

## Risks / Trade-offs

- [Environment-driven configuration is less discoverable than in-code test data]
  → Document the variables and expected flow in this design.
- [Per-case staging adds filesystem work] → Accept the cost in exchange for
  better test isolation.
- [The helper is intentionally narrow] → Keep repo-specific expectations in
  consuming test modules rather than widening the shared API.
