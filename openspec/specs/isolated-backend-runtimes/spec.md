# isolated-backend-runtimes Specification

## Purpose

Define how MLIA selects, installs and executes versioned backend packages while
keeping their dependencies separate from the frontend.

## Requirements

### Requirement: Isolated package installation
Core SHALL install runtime packages and their dependencies in separate persistent
Linux environments, verify supplied wheel hashes, obtain the requested Python when
needed, and mark installations ready only after validation. Core SHALL provide
runtime support without an optional extra; migrated operations SHALL require it.

#### Scenario: First use
- **WHEN** a check selects an available runtime that is not installed
- **THEN** MLIA provisions it without installing worker dependencies in the frontend
- **AND** later checks reuse the validated environment

### Requirement: Explicit toolchain selection
Execution profiles SHALL bind provider-owned operations to exact runtime IDs.
Defaults SHALL select supported backend/input combinations; explicit selections
SHALL not silently fall back. Preview SHALL validate selections without provisioning.
Stages MAY restrict themselves to original model input formats. Only matching
stages SHALL require provider plugins or provision runtimes for a check.

#### Scenario: Select another version
- **WHEN** a user supplies a profile by name, file or environment setting
- **THEN** its selected versions are used, or an actionable configuration error is reported
- **AND** missing frontend providers are identified rather than installed automatically

### Requirement: Generic supervised transport
Core SHALL exchange versioned JSON envelopes and files through subprocesses,
validate job identity and artifact containment, and terminate subprocess groups
on timeout or cancellation. Providers SHALL own operation payload/result semantics.

#### Scenario: Invalid worker output
- **WHEN** a worker fails or returns malformed, mismatched or invalid artifact references
- **THEN** execution fails with an MLIA error and the output is not consumed as success

### Requirement: Provider ownership and complete operations
Workers SHALL perform complete conversions, backend executions or model processing.
Target plugins SHALL retain result analysis and advice. Caller-side live-model
export SHALL remain available; incompatible representations SHALL fail explicitly.

#### Scenario: Shared target collection
- **WHEN** compatibility and performance need the same NX model facts and raw outputs
- **THEN** one worker collection serves both frontend analyses

### Requirement: Versioned delivery
Provider wheels SHALL include inert worker wheels, manifests and default profiles.
Current runtime versions SHALL derive from their owning provider wheel metadata.
Runtime dependencies SHALL use ordinary Python package metadata; installed versions
SHALL be inspectable. Partner catalogs MAY supply private wheels alongside online
public dependencies, with distinct identities for public and partner variants.

#### Scenario: Cross-provider selection
- **WHEN** a target default profile selects a converter
- **THEN** that binding uses the installed converter provider's version

### Requirement: Consent and lifecycle management
CLI, API and interactive frontends SHALL share verified full-text EULA handling,
installation, forced replacement and removal. Backend aliases SHALL select their
declared defaults; another version SHALL be addressed by its qualified runtime ID.
Installation, execution and removal SHALL coordinate through runtime locks.

#### Scenario: Consent is declined
- **WHEN** required terms are declined or noninteractive acceptance is absent
- **THEN** no selected runtime installation starts

#### Scenario: Forced replacement fails
- **WHEN** a replacement fails and the runtime store remains writable
- **THEN** the prior environment is restored and available diagnostics are retained

### Requirement: Local package retention
Explicit local installation SHALL validate the requested runtime selection and
retain verified package inputs for later use. Registration failures after successful
installation SHALL explain how to retry without rebuilding the environment.

#### Scenario: Original package files are removed
- **WHEN** a registered runtime is selected or reinstalled later
- **THEN** its retained inputs remain usable without changing profile selections

### Requirement: Lightweight and consistent discovery
Metadata-only discovery SHALL avoid loading plugin entry points. Check help SHALL
expose installed backend/analysis options without importing execution libraries.
A single operation SHALL reuse its runtime catalog; later calls SHALL see changes.

#### Scenario: Preview and execute through the API
- **WHEN** backend resolution and execution inspect runtime declarations in one call
- **THEN** they use the same catalog snapshot
