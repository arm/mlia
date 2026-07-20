## 1. Narrow Generic Completion Imports

- [x] 1.1 Preserve the top-level MLIA API through lazy exports only for measured
  runtime-heavy and plugin-backed surfaces.
- [x] 1.2 Keep cheap CLI helpers, logging, and Rich UI imports conventional.
- [x] 1.3 Move backend option metadata construction from the runtime-heavy API
  to the backend options module, retaining the direct API re-export and
  plugin-provided behavior.
- [x] 1.4 Preserve backend and target package initialization as the existing
  plugin-loading boundaries.

## 2. Implement Completion Inventories

- [x] 2.1 Complete all backend-valued CLI positions from registered selectable
  backend names without using installation-state APIs.
- [x] 2.2 Move the pure selectable-name query to the backend registry while
  retaining the existing manager import path.
- [x] 2.3 Complete target profile names by scanning active MLIA namespace
  resources without importing the target runtime.
- [x] 2.4 Sort and deduplicate completion inventories where applicable.
- [x] 2.5 Return no sentinel candidate when MLIA has no matching value.

## 3. Replace Rejected-Architecture Tests

- [x] 3.1 Add fresh-process tests for lazy public exports and generic completion
  import boundaries.
- [x] 3.2 Test controlled plugin-provided backend registration through the
  package import boundary without claiming plugin transitive imports stay absent.
- [x] 3.3 Prove core backend completion does not inspect installation state.
- [x] 3.4 Test namespace target-profile discovery and its target-runtime import
  boundary.
- [x] 3.5 Test all backend CLI positions, plugin-option completion and help, and
  Bash and Zsh no-match fallback behavior.

## 4. Validate

- [x] 4.1 Validate the OpenSpec change in strict mode.
- [x] 4.2 Run focused completion, CLI, import-boundary, and backend tests.
- [x] 4.3 Run the non-slow test suite and configured pre-commit checks.
- [x] 4.4 Compare fresh-process completion timings on core and installed-plugin
  environments without adding wall-clock assertions to tests.
- [x] 4.5 Review the final diff for scope, public/private boundaries, and stale
  rejected plugin-lifecycle claims.
