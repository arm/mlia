## 1. Discovery and compatibility

- [x] 1.1 Add entry-point discovery for backend, target, CLI, and converter
  plugin groups.
- [x] 1.2 Extend the backend and target package paths so extension
  distributions can contribute importable modules under those namespaces.
- [x] 1.3 Add compatibility checks for external plugin distributions and plugin
  interface versions.
- [x] 1.4 Log and skip plugins that fail compatibility or import checks.

## 2. Integration points

- [x] 2.1 Load backend and target plugins through package imports and
  registry-access paths.
- [x] 2.2 Introduce package-owned CLI command configuration and load CLI plugins
  during `mlia.cli` package initialization.
- [x] 2.3 Provide converter registration through the registry object supplied to
  the converter plugin loader.

## 3. Observability and stability

- [x] 3.1 Expose plugin metadata for CLI reporting.
- [x] 3.2 Ensure plugin failures are isolated to the failing plugin.
- [x] 3.3 Preserve existing converter registrations on duplicate names.
