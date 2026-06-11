## Why

Some backend packages have one primary payload plus additional supporting
folders under the same validated root. Today the path checker can validate that
layout but returns only one install source, so supporting folders are not copied.

## What Changes

- Allow a matched path checker to return a primary install source plus
  additional supporting folders from the same package layout.
- Extend `PackagePathChecker` for this metadata while preserving existing
  no-subfolder and single-subfolder behavior.
- Keep `CompoundPathChecker` first-match semantics; it selects one complete
  layout rather than merging separate checker results.
- Copy supporting folders recursively without requiring per-file configuration.

## Capabilities

### New Capabilities

- `backend-package-installation`: Defines how backend package layouts are
  validated and how primary and supporting payload folders are selected for
  repository installation.

### Modified Capabilities

None.

## Impact

- Core code: `src/mlia/backend/install.py`, backend repository copy logic, and
  backend installation tests.
- Plugin configuration: callers that compose `PackagePathChecker` and
  `CompoundPathChecker`.
- No new runtime dependencies are expected.
