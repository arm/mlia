## Why

Shell completion runs on every interactive tab press, but constructing the MLIA
command tree currently imports runtime-only API, installation, target, advice,
and reporting code. That makes generic completion pay for work it does not use.

## What Changes

- Keep root command and option-name completion away from the runtime-heavy API
  and target paths while retaining plugin-provided check options through the
  existing backend package import boundary.
- Complete backend-valued arguments from selectable names registered by
  available backend plugins through the existing backend package import
  boundary, without checking installation state.
- Complete MLIA-packaged target profile names by scanning namespace resources
  without importing the target runtime or target plugins.
- Return no MLIA candidate when a value has no match so the shell can apply its
  normal fallback behavior.
- Preserve plugin-provided backend options in shell completion and normal
  `mlia check --help` output.
- Test the backend package boundary separately from API, installation, and target
  runtime imports.

## Capabilities

### New Capabilities

- `cli-completion`: shell completion provides fast command, option, backend,
  and target-profile suggestions without unrelated runtime discovery.

## Impact

- `src/mlia/__init__.py`
- `src/mlia/backend/config.py`
- `src/mlia/backend/options.py`
- `src/mlia/backend/registry.py`
- `src/mlia/backend/manager.py`
- `src/mlia/cli/commands.py`
- `src/mlia/cli/completion.py`
- CLI completion, import-boundary, and backend registry tests.
