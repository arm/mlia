## 1. Case loading and suite selection

- [x] 1.1 Add execution-payload parsing for environment-defined e2e suites.
- [x] 1.2 Generate concrete cases by expanding parameter combinations and model
  globs.
- [x] 1.3 Add compatibility and performance suite filtering for pytest
  parametrization.

## 2. Case execution

- [x] 2.1 Add prepared-artifact staging into per-case work directories.
- [x] 2.2 Add backend installation from the configured backend list.
- [x] 2.3 Execute each case as one `mlia` subprocess and return captured
  results.

## 3. Validation and test integration

- [x] 3.1 Validate resolved commands before execution and reject duplicate
  commands.
- [x] 3.2 Add sharding support for distributed e2e runs.
- [x] 3.3 Keep repo-specific assertions in consuming test modules.
