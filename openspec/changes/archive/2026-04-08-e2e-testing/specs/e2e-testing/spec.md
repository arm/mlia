## ADDED Requirements

### Requirement: Model e2e suites as unique CLI invocations
The `mlia.testing.e2e` helper SHALL model each parametrized case as one concrete
`mlia` CLI invocation identified by its full command line. The helper SHALL
reject duplicate commands after parameter expansion and SHALL expose
suite-specific parametrization for compatibility and performance cases.

#### Scenario: Duplicate case definitions are rejected
- **WHEN** configured execution blocks resolve to the same `mlia` command line
- **THEN** case loading SHALL fail with a duplicate-command runtime error

#### Scenario: Suite parametrization filters cases by check category
- **WHEN** a consumer parametrizes a test for the `compatibility` suite
- **THEN** the helper SHALL yield only `check` cases that include
  `--compatibility`

#### Scenario: Performance parametrization filters cases by check category
- **WHEN** a consumer parametrizes a test for the `performance` suite
- **THEN** the helper SHALL yield only `check` cases that include
  `--performance`

### Requirement: Load shared e2e configuration from environment
The shared helper SHALL derive its generic execution model from environment
variables instead of repo-specific code. It SHALL load execution definitions
from `MLIA_E2E_EXECUTIONS`, prepared artifact roots from `MLIA_E2E_ARTIFACTS`,
backend names from `MLIA_E2E_BACKENDS`, and optional sharding controls from
`MLIA_E2E_SHARD_INDEX` and `MLIA_E2E_SHARD_COUNT`.

#### Scenario: Configured executions require a prepared artifacts root
- **WHEN** `MLIA_E2E_EXECUTIONS` is set and `MLIA_E2E_ARTIFACTS` is missing
- **THEN** case loading or case execution SHALL fail because the prepared
  artifacts root is required

#### Scenario: Invalid shard configuration is rejected
- **WHEN** only one shard variable is set, the shard values are not integers, or
  the shard index falls outside `[0, shard_count)`
- **THEN** case loading SHALL fail with a shard-configuration runtime error

#### Scenario: Sharding selects a stable subset of cases
- **WHEN** both shard variables are set to valid values
- **THEN** the helper SHALL keep only cases whose positional index satisfies
  `index % shard_count == shard_index`

### Requirement: Expand prepared artifact inputs relative to the artifacts root
The helper SHALL interpret prepared model arguments relative to the prepared
artifacts root. It SHALL expand wildcard paths under `e2e_config/` before case
creation and SHALL validate each resulting command with the MLIA CLI parser.

#### Scenario: Model globs create one case per resolved artifact
- **WHEN** an execution argument under `e2e_config/` contains a wildcard
- **THEN** the helper SHALL resolve the glob relative to the prepared artifacts
  root and create one case for each resolved artifact path

#### Scenario: Invalid commands are rejected before execution
- **WHEN** a resolved case does not parse as a valid `mlia` CLI invocation
- **THEN** the helper SHALL fail with an invalid-command runtime error instead
  of invoking the command

### Requirement: Run each case in an isolated work directory
The shared helper SHALL stage prepared artifacts for one case into the supplied
work directory and SHALL execute exactly one `mlia` command for that case. The
helper SHALL install any requested backends before execution and SHALL proxy
captured stdout and stderr back to pytest when requested.

#### Scenario: Referenced prepared artifacts are staged into the case workdir
- **WHEN** a case argument names a relative file path that exists under the
  configured artifacts root
- **THEN** the helper SHALL copy those prepared artifacts from the configured
  artifacts root into the case work directory before launching `mlia`

#### Scenario: Requested backends are installed once before case execution
- **WHEN** `MLIA_E2E_BACKENDS` contains backend names
- **THEN** the helper SHALL invoke backend installation for each requested
  backend before running cases and SHALL reuse that installed set across case
  executions in the same process

#### Scenario: One case produces one subprocess invocation
- **WHEN** a consumer calls `run_case(case, workdir=...)`
- **THEN** the helper SHALL execute exactly one `mlia <command> ...` subprocess
  in that work directory and return its completed result

### Requirement: Keep repo-specific assertions outside the shared helper
The shared helper SHALL stop at generic case discovery, artifact staging, and
command execution. Target-specific, backend-specific, and command-specific
output assertions that are not reusable across MLIA repositories SHALL remain
in the consuming test modules.

#### Scenario: Consuming tests own domain-specific assertions
- **WHEN** a repository needs to assert target-specific or backend-specific e2e
  behavior
- **THEN** those assertions SHALL be implemented in the repository's pytest
  modules rather than in `mlia.testing.e2e`
