## ADDED Requirements

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
- **WHEN** MLIA builds the command list for the main CLI
- **THEN** it SHALL enumerate entry points in `mlia.plugin.cli` and invoke each
  accepted plugin with the mutable command configuration

#### Scenario: Converter plugins are loaded from the converter group
- **WHEN** MLIA loads converter extensions
- **THEN** it SHALL enumerate entry points in `mlia.plugin.converter` and
  invoke each accepted plugin with a converter registry

### Requirement: Accept only compatible plugin implementations
MLIA SHALL require each loaded plugin module to expose a supported
`plugin_interface_version`. External plugins SHALL also declare an `mlia`
package requirement compatible with the installed core package version. Plugins
that fail compatibility checks, import, or registration SHALL be skipped and
reported through logging instead of aborting the host process.

#### Scenario: Internal plugins bypass external core-version gating
- **WHEN** an entry point comes from the same distribution as the `mlia`
  console script
- **THEN** MLIA SHALL treat it as an internal plugin and SHALL not require the
  external core-version compatibility check

#### Scenario: External plugins must declare a compatible core requirement
- **WHEN** an external plugin does not declare an `mlia` requirement or its
  declared version range excludes the installed `mlia` version
- **THEN** MLIA SHALL skip that plugin and log a compatibility error

#### Scenario: Unsupported plugin interface versions are rejected
- **WHEN** a plugin module exposes a `plugin_interface_version` other than
  `0.0.1`
- **THEN** MLIA SHALL skip that plugin and log an incompatible-version error

#### Scenario: Import and registration failures are isolated to the failing plugin
- **WHEN** loading or registering a plugin raises an exception
- **THEN** MLIA SHALL log the failure and continue processing the remaining
  plugins in that entry point group

### Requirement: Load plugins into registries at the point of use
MLIA SHALL connect the plugin loader to the backend registry, target registry,
dynamic constants, and CLI command construction so installed plugins become
visible through the normal public APIs. Target-facing flows that depend on
backend registrations SHALL load backend plugins before target plugins.

#### Scenario: Backend registry access loads backend plugins once
- **WHEN** callers query supported backends through the backend registry module
- **THEN** MLIA SHALL ensure backend plugins have been loaded before returning
  registry-backed data

#### Scenario: Target registry access loads backend plugins before target plugins
- **WHEN** callers query targets, target profiles, or target-backed constants
- **THEN** MLIA SHALL ensure backend plugins are loaded before target plugins so
  target registrations can rely on registered backend names

#### Scenario: CLI command construction includes plugin-provided commands
- **WHEN** MLIA builds the main command list
- **THEN** it SHALL load CLI plugins before returning the final command
  configuration

### Requirement: Expose plugin inventory and converter registration behavior
MLIA SHALL expose entry point metadata for CLI reporting and SHALL provide a
converter registry that stores named converter callables for converter plugins.

#### Scenario: Plugin inventory is available for CLI reporting
- **WHEN** MLIA lists backend or target plugins
- **THEN** it SHALL expose each discovered entry point's name, value,
  distribution name, and distribution version

#### Scenario: Converter registries provide stable named lookups
- **WHEN** a converter plugin registers a converter under a new name
- **THEN** the converter registry SHALL make that converter retrievable by name
  and include the name in sorted listings

#### Scenario: Duplicate converter registrations preserve the existing converter
- **WHEN** a converter plugin registers a name that already exists in the
  converter registry
- **THEN** the registry SHALL keep the existing converter and log a warning
