## 1. Tests

- [x] 1.1 Add a failing backend installation test that installs a primary
  payload folder and an additional supporting payload folder from one vendored
  backend package and verifies both directories are present in the repository.
- [x] 1.2 Add a failing backend installation test where the supporting payload
  folder contains a nested directory tree and verifies nested files are copied
  recursively.
- [x] 1.3 Add focused tests for `PackagePathChecker` compatibility with no
  `backend_subfolder`, one string `backend_subfolder`, and multiple configured
  payload folders.
- [x] 1.5 Add a failing test for rejection when one configured backend payload
  folder is missing or is not a directory.
- [x] 1.6 Add a `CompoundPathChecker` regression test showing first-match
  behavior is preserved and the first matched checker can still return multiple
  payload folders.

## 2. Core Implementation

- [x] 2.1 Extend backend path resolution metadata so one matched checker can
  describe multiple selected payload folders without breaking existing
  `BackendInfo.backend_path` callers.
- [x] 2.2 Extend `PackagePathChecker` to accept and normalize multiple backend
  payload folders while preserving string `backend_subfolder` behavior.
- [x] 2.4 Update `BackendInstallation._install_from()` and repository copy logic
  to copy multiple selected payload folders into the installed backend directory
  while preserving their relative names and nested contents.
- [x] 2.5 Ensure missing configured payload folders cause path checking to fail
  before installation.
- [x] 2.6 Leave downstream backend plugin configuration to the owning package;
  this core change only exposes the supporting payload folder API.

## 3. Validation

- [x] 3.1 Run the focused MLIA backend installation tests.
- [x] 3.2 Skip backend plugin installation tests because downstream plugin
  changes are outside this package.
- [x] 3.3 Run lint/type checks required for the touched files.
- [x] 3.4 Run the quick non-slow pytest suite if the focused checks pass.
