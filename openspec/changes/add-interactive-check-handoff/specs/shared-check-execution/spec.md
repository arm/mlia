## ADDED Requirements

### Requirement: Execute prepared checks through one core lifecycle
MLIA SHALL expose `execute_check` and public result/error types. The executor
SHALL own completeness validation before output creation, existing category and
backend selection rules including profiling data, installation/EULA handling,
advisor execution, standardized persistence, analysis-plugin dispatch, and
logging cleanup. Compatibility SHALL remain the default category when neither
analysis flag is selected; both flags SHALL remain supported together.

#### Scenario: Incomplete request reaches execution
- **WHEN** required model or target inputs are still missing at Run
- **THEN** execution SHALL reject the request without creating its output directory
- **AND** existing output SHALL remain unchanged, including when no model and an empty profiling-data sequence are supplied

#### Scenario: Equivalent configured CLI and frontend requests
- **WHEN** equivalent complete requests are executed through either frontend
- **THEN** the shared executor SHALL apply the same validation, selection, and analysis rules

#### Scenario: Profiling data affects selection
- **WHEN** a request supplies profiling data
- **THEN** execution SHALL preserve its input order and apply the existing profiling-data selection and validation rules rather than a frontend approximation
- **AND** a model SHALL NOT be required when the core supports the supplied profiling-only input combination

### Requirement: Preserve plugin and prompt ownership
Execution SHALL support frontend presentation of installation questions through
a prompt callback and SHALL honor noninteractive and EULA policy. Core SHALL
preserve ordinary CLI support for resource-valued plugin options. Worker
transport restrictions SHALL NOT become global plugin-interface restrictions.

#### Scenario: Frontend presents an installation question
- **WHEN** installation requires a question and a prompt callback is supplied
- **THEN** execution SHALL use that callback and honor its response

#### Scenario: Noninteractive installation policy
- **WHEN** a request disallows installation prompts
- **THEN** execution SHALL apply existing noninteractive policy without requesting terminal input

#### Scenario: Plugin invocation follows persisted analysis
- **WHEN** an analysis plugin fails after canonical output was persisted
- **THEN** execution SHALL report failure while retaining the persisted output for diagnosis

### Requirement: Represent completion and skipping explicitly
`CheckResult` SHALL provide canonical output when present, actual output location,
report, and completed/skipped outcome with a skip reason when available.
`CheckError` SHALL carry the appropriate command exit status. A failed check
verdict SHALL remain a completed analysis when execution itself succeeded.

#### Scenario: Validation elects to skip analysis
- **WHEN** existing validation policy skips a check
- **THEN** execution SHALL return a skipped result without presenting a successful analysis payload

#### Scenario: Compatibility findings report unsupported operators
- **WHEN** analysis completes with incompatibility findings
- **THEN** execution SHALL return those findings as a completed result

### Requirement: Map integrated session outcome to the latest attempt
After the TUI restores the terminal, the CLI SHALL return the latest accepted
attempt outcome. No attempt and skipped/completed attempts SHALL exit 0. A
failure SHALL preserve its `CheckError` status or use 1. Cancellation or quitting
an active attempt SHALL exit 130. Later success SHALL supersede earlier failure;
startup, rendering, or cleanup failure SHALL NOT fall back to an earlier success.

#### Scenario: A successful run is followed by failed worker startup
- **WHEN** a later accepted Run cannot start
- **THEN** closing the session SHALL fail rather than return the earlier successful result

#### Scenario: Configuration interaction does not begin an attempt
- **WHEN** the user edits configuration or cancels a chooser or overwrite confirmation
- **THEN** the prior attempt outcome SHALL remain the session outcome

#### Scenario: User cancels an active attempt
- **WHEN** the latest accepted attempt is cancelled
- **THEN** closing the integrated session SHALL return status 130 unless a later attempt supersedes it

### Requirement: Emit integrated JSON only after successful terminal restoration
The integrated path SHALL suppress the normal plain-text result report while the
TUI owns the terminal. After closure, requested JSON SHALL contain only the latest
completed canonical payload when present. No attempt, skipped, failed, or
cancelled outcomes SHALL emit a success payload. Skipped outcomes SHALL report
the reason on stderr.

#### Scenario: JSON requested for a completed attempt
- **WHEN** the session closes with a completed canonical payload and JSON was requested
- **THEN** MLIA SHALL emit that payload after terminal restoration and exit 0

#### Scenario: Later failure after success with JSON enabled
- **WHEN** the latest attempt failed after an earlier completed attempt
- **THEN** MLIA SHALL emit no JSON success payload and SHALL return the failure status
