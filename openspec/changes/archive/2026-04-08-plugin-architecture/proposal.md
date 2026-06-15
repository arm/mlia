## Why

MLIA needs an extension mechanism that allows targets, backends, CLI commands,
and converters to be added without modifying the core package. That mechanism
must support third-party distributions, reject incompatible plugins safely, and
make plugin-provided capabilities visible through the normal MLIA APIs.

## What Changes

- Add entry-point-based discovery for backend, target, CLI, and converter
  plugins.
- Extend the `mlia.backend` and `mlia.target` package paths so extension
  distributions can contribute importable modules under those namespaces.
- Add compatibility checks for external plugins and plugin interface versions.
- Load accepted plugins into backend, target, and CLI package initialization
  flows and expose them through the relevant public APIs.
- Expose plugin metadata for CLI reporting and provide converter registration
  behavior.

## Capabilities

### New Capabilities
- `plugin-architecture`: Defines how MLIA discovers, validates, and integrates
  backend, target, CLI, and converter plugins.

### Modified Capabilities

## Impact

- Affects plugin loading, registry initialization, CLI package initialization,
  and CLI command discovery.
- Affects backend and target package import behavior for extension
  distributions.
- Enables extension packages to integrate with MLIA without direct core changes.
- Adds compatibility gates so invalid plugins are skipped rather than aborting
  the host process.
