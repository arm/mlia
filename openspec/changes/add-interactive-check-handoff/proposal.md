## Why

An optional terminal interface needs the same resolved options and execution
behavior as `mlia check`. A shared preparation and execution boundary lets users
complete configuration before running, without duplicating CLI defaults or
backend selection in the frontend.

## What Changes

- Add `mlia check --tui` with deferred execution and an actionable error when
  the optional `mlia-tui` package is absent.
- Provide public prepared-request, result, error, execution, standalone-default
  preparation, and profile-to-target lookup interfaces.
- Resolve standalone defaults through the actual CLI declarations and parser,
  retaining configuration, dynamic options, conversions, and resource ownership.
- Preserve ordinary CLI validation, results, optional installation behavior,
  resource-valued plugin options, and lightweight completion.
- Define the integrated session's latest-attempt result and exit semantics.

## Capabilities

### New Capabilities

- `interactive-check-preparation`: Shared resolved requests, standalone defaults,
  optional TUI dispatch, incomplete editable inputs, and profile lookup.
- `shared-check-execution`: Common execution ownership, installation questions,
  plugin dispatch, canonical output, and integrated session outcomes.

## Impact

Core owns these APIs and the CLI adapter; the `mlia-tui` repository owns Textual,
worker isolation, and package delivery.

No standardized result schema change or backend-selection preview is proposed.
Existing CLI behavior without `--tui` remains supported. Publication of the
core API is required before the TUI can set a published dependency floor and
complete clean-package validation.
