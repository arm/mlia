## ADDED Requirements

### Requirement: Prepare resolved requests without execution
MLIA SHALL expose public request and settings types and `prepare_check` to capture
resolved check inputs without executing analysis, installing backends, selecting
automatic backends, or creating output directories. MLIA-owned mappings SHALL be
copied and plugin-owned values SHALL retain their identity and lifetime rules.

#### Scenario: Preparing incomplete editable inputs
- **WHEN** preparation receives a request without a model or target profile
- **THEN** it SHALL retain the supplied options without running a check or creating output

#### Scenario: Preserving resource-valued plugin options
- **WHEN** a resolved plugin option contains an open resource
- **THEN** preparation SHALL preserve that resource identity rather than deep-copying or stringifying it

### Requirement: Resolve standalone defaults through the CLI declarations
MLIA SHALL expose `mlia.cli.preparation.prepare_check_defaults()` as a context manager using
the actual root/check parser and preparation path. Configuration, environment,
dynamic backend/analysis options, callbacks, and type conversions SHALL follow
those declarations. Owned contexts SHALL remain open through frontend shutdown
and close on every exit path, including failures during parsing.

#### Scenario: Typed and environment-derived defaults
- **WHEN** a plugin declares a typed default or environment-derived option
- **THEN** standalone preparation SHALL produce the same converted value as the equivalent CLI preparation

#### Scenario: Callback resources survive frontend use
- **WHEN** an option callback registers a resource with its context
- **THEN** the resource SHALL remain usable inside the preparation context and close when that context exits

#### Scenario: Preparation fails after acquiring resources
- **WHEN** a later parser callback fails
- **THEN** contexts and their registered resources SHALL be released

### Requirement: Prepare standalone defaults without declarative prompts
Standalone default preparation SHALL suppress declarative option prompts without
changing normal CLI prompt behavior. Missing required values or callbacks that
abort while requesting terminal input SHALL produce an actionable configuration
error with exit status 2 directing the user to `mlia check --tui` with options.

#### Scenario: Required plugin option has no usable default
- **WHEN** standalone preparation cannot resolve a required plugin option
- **THEN** it SHALL name the missing option and fail before launching the frontend

#### Scenario: Prompted option has an environment value
- **WHEN** a declaratively prompted option has a usable environment value
- **THEN** standalone preparation SHALL convert that value without requesting terminal input

### Requirement: Dispatch optional TUI after shared option preparation
`mlia check --tui` SHALL pass the resolved request to public
`mlia_tui.launch(request)` and SHALL permit missing editable model/target fields.
It SHALL preserve all supplied core and dynamic options and SHALL await an
explicit Run action before execution. Invalid syntax SHALL remain a parser error.

#### Scenario: Backend supplied without a target profile
- **WHEN** `--tui` and a backend are supplied without a target profile
- **THEN** the frontend SHALL receive that backend and the incomplete request without starting analysis

#### Scenario: Complete CLI configuration
- **WHEN** a complete request is supplied with `--tui`
- **THEN** the frontend SHALL receive that configuration and wait for Run

#### Scenario: Optional package is absent
- **WHEN** `--tui` is requested and `mlia_tui` is not installed
- **THEN** MLIA SHALL fail with an actionable instruction to install `mlia-tui` in the current environment

### Requirement: Preserve ordinary CLI behavior
Without `--tui`, MLIA SHALL retain strict input validation, ordinary execution,
JSON/text reporting, dynamic option handling, installation policy, and existing
help/completion behavior. Core installation SHALL NOT require the TUI package.

#### Scenario: Ordinary CLI lacks a required target
- **WHEN** an ordinary check invocation lacks its required target profile
- **THEN** MLIA SHALL reject it with status 2 before resolving unspecified non-eager plugin prompts or executing analysis

#### Scenario: Empty invocation and completion
- **WHEN** the ordinary root/check command has no arguments or shell completion is requested
- **THEN** existing help exit behavior and lightweight completion SHALL be preserved without importing the TUI

### Requirement: Resolve a target from a profile through a public API
MLIA SHALL expose `mlia.api.target_for_profile` for named and file-backed profiles
using core profile resolution. The query SHALL NOT execute analysis or promise
which backends automatic execution will select.

#### Scenario: File-backed target profile
- **WHEN** a supported profile file is queried
- **THEN** the API SHALL return its resolved target or the core profile validation error
