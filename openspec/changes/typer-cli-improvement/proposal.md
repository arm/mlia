## Why

MLIA's CLI parser and command wiring were split across argparse helpers, command metadata, and CLI-only backend install code. This made help text, exit codes, deprecated entry points, and public API reuse harder to keep consistent.

## User-Visible CLI Changes

- CLI help, option handling, and empty-command exit behavior now follow Typer conventions.
- `mlia backend install|uninstall|list` and `mlia target list` are available as root subcommands.
- Show deprecation warnings for the legacy `mlia-backend` and `mlia-target` script entry points, pointing users to `mlia backend` and `mlia target`.
- `mlia backend install` replaces the hidden `--i-agree-to-the-contained-eula` flag with the explicit `--accept-eula` option.
- Add plugin discovery guidance to top-level CLI help.
- Format backend and target list output to match the Typer-based CLI style.
- Expose backend-specific options discovered from backend metadata on
  `mlia check` and forward provided values to advice generation.

## Structural Changes

- Add Typer as a runtime dependency.
- Replace the argparse command/option registration layer with Typer apps and command functions.
- Remove the old `CommandInfo`-based CLI command registration path with the argparse layer.
- Remove the dedicated CLI options module and keep backend option discovery available from the API layer.
- Connect API backend option discovery to the Typer `check` command through
  dynamic Click options.
- Move backend install and uninstall operations into public API functions so the CLI calls the same surface as other callers.
- Rework CLI context and logging setup around the new Typer command path.

## Impact

- `pyproject.toml`
- `src/mlia/__init__.py`
- `src/mlia/cli/main.py`
- `src/mlia/cli/commands.py`
- `src/mlia/cli/options.py`
- `src/mlia/api.py`
- `src/mlia/cli/command_validators.py`
- `src/mlia/cli/helpers.py`
- `src/mlia/core/logging.py`
- CLI, API option discovery, logging, and e2e tests.
