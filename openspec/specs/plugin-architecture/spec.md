## Purpose

Provide a plugin system that lets extension packages add targets, backends, CLI
commands, and converters to MLIA through well-defined discovery,
compatibility-checking, and registration flows.

## Requirements

### Requirement: Discover plugins through dedicated entry point groups
MLIA SHALL discover extensibility plugins through Python package entry points.
It SHALL use dedicated groups for target plugins, backend plugins, CLI plugins,
and converter plugins so each integration surface can load only the plugins
intended for that surface.

#### Scenario: Backend plugins are loaded from the backend group
- **WHEN** MLIA loads backend extensions
- **THEN** it SHALL enumerate entry points in `mlia.plugin.backend` and invoke
  each accepted plugin with the backend registry

#### Scenario: Target plugins are loaded from the target group
- **WHEN** MLIA loads target extensions
- **THEN** it SHALL enumerate entry points in `mlia.plugin.target` and invoke
  each accepted plugin with the target registry

#### Scenario: CLI plugins are loaded from the CLI group
- **WHEN** MLIA initializes the CLI plugin surface
- **THEN** it SHALL enumerate entry points in `mlia.plugin.cli` and invoke each
  accepted plugin with the mutable command configuration

#### Scenario: Converter plugins are loaded from the converter group
- **WHEN** MLIA loads converter extensions
- **THEN** it SHALL enumerate entry points in `mlia.plugin.converter` and invoke
  each accepted plugin with the registry object supplied by the caller

### Requirement: Keep backend and target extension modules importable across distributions
MLIA SHALL extend the `mlia.backend` and `mlia.target` package paths so backend
and target modules can be provided by multiple distributions, while keeping
entry point groups as the authoritative plugin-discovery mechanism.

#### Scenario: Backend extension modules are importable from multiple distributions
- **WHEN** multiple installed distributions provide subpackages under
  `mlia.backend`
- **THEN** importing `mlia.backend` SHALL include those subpackages in the
  package search path

#### Scenario: Target extension modules are importable from multiple distributions
- **WHEN** multiple installed distributions provide subpackages under
  `mlia.target`
- **THEN** importing `mlia.target` SHALL include those subpackages in the
  package search path

#### Scenario: Plugin loading uses entry points rather than namespace scanning
- **WHEN** MLIA decides which plugins to load for a plugin surface
- **THEN** it SHALL use the corresponding entry point group rather than scanning
  importable backend or target subpackages

### Requirement: Accept only compatible plugin implementations
MLIA SHALL require each loaded plugin module to expose a supported
`plugin_interface_version`. External plugins SHALL also declare an `mlia`
package requirement compatible with the installed core package version. Plugins
that fail compatibility checks, import, or registration SHALL be skipped and
reported through logging instead of aborting the host process.

#### Scenario: Internal plugins bypass external core-version gating
- **WHEN** an entry point comes from the same distribution as the `mlia` console
  script
- **THEN** MLIA SHALL treat it as an internal plugin and SHALL not require the
  external core-version compatibility check

#### Scenario: External plugins must declare a compatible core requirement
- **WHEN** an external plugin does not declare an `mlia` requirement or its
  declared version range excludes the installed `mlia` version
- **THEN** MLIA SHALL skip that plugin and log a compatibility error

#### Scenario: Unsupported plugin interface versions are rejected
- **WHEN** a plugin module exposes a `plugin_interface_version` other than
  `"0.0.1"`
- **THEN** MLIA SHALL skip that plugin and log an incompatible-version error

#### Scenario: Import and registration failures are isolated to the failing plugin
- **WHEN** loading or registering a plugin raises an exception
- **THEN** MLIA SHALL log the failure and continue processing the remaining
  plugins in that entry point group

### Requirement: Initialize plugin-backed surfaces consistently
MLIA SHALL connect the plugin loader to package initialization and public APIs
so plugin-backed surfaces are populated before callers use them. Backend,
target, and CLI package initialization SHALL load the plugins for those
surfaces, and target-facing flows that depend on backend registrations SHALL
load backend plugins before target plugins.

#### Scenario: Importing the backend package loads backend plugins
- **WHEN** callers import the `mlia.backend` package
- **THEN** MLIA SHALL load backend plugins into the backend registry during
  package initialization

#### Scenario: Importing the target package loads backend plugins before target plugins
- **WHEN** callers import the `mlia.target` package
- **THEN** MLIA SHALL load backend plugins before target plugins during package
  initialization

#### Scenario: Importing the CLI package loads CLI plugins
- **WHEN** callers import the `mlia.cli` package
- **THEN** MLIA SHALL load CLI plugins into the command configuration used by
  CLI command construction during package initialization

#### Scenario: Backend registry access loads backend plugins once
- **WHEN** callers query supported backends through the backend registry module
- **THEN** MLIA SHALL ensure backend plugins have been loaded before returning
  registry-backed data

#### Scenario: Target registry access loads backend plugins before target plugins
- **WHEN** callers query targets, target profiles, or target-backed constants
- **THEN** MLIA SHALL ensure backend plugins are loaded before target plugins so
  target registrations can rely on registered backend names

#### Scenario: CLI command construction uses the initialized command configuration
- **WHEN** MLIA builds the main command list
- **THEN** it SHALL return the command configuration that already includes
  plugin-provided commands

#### Scenario: Converter plugins load only when the converter loader is invoked
- **WHEN** MLIA calls the converter plugin loader
- **THEN** it SHALL enumerate converter plugin entry points and register
  accepted converters into the supplied registry

### Requirement: Expose plugin inventory and preserve registry stability
MLIA SHALL expose entry point metadata for CLI reporting and SHALL preserve the
stability guarantees of the registries that plugins register into.

#### Scenario: Plugin inventory is available for CLI reporting
- **WHEN** MLIA lists backend or target plugins
- **THEN** it SHALL expose each discovered entry point's name, value,
  distribution name, and distribution version

#### Scenario: Converter registration preserves the existing entry on duplicates
- **WHEN** a converter plugin registers a name that already exists in the
  supplied registry
- **THEN** the existing entry SHALL be preserved and a warning SHALL be logged
