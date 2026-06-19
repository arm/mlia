## 1. CLI Entry Points

- [x] 1.1 Define Typer apps for the root, backend, and target command groups.
- [x] 1.2 Wire `mlia check`, `mlia backend install|uninstall|list`, and `mlia target list`.
- [x] 1.3 Preserve `mlia-backend` and `mlia-target` as deprecated wrappers with warnings.
- [x] 1.4 Use `--accept-eula` for backend install EULA acceptance.
- [x] 1.5 Add plugin discovery guidance to root help.

## 2. Structural Changes

- [x] 2.1 Replace argparse command and option registration with Typer command definitions.
- [x] 2.2 Add Typer as a runtime dependency.
- [x] 2.3 Remove the old `CommandInfo`-based CLI command registration path.
- [x] 2.4 Remove the dedicated CLI options module.
- [x] 2.5 Add public backend install and uninstall API functions.
- [x] 2.6 Keep backend option discovery available without the removed CLI options module.
- [x] 2.7 Rework CLI context and logging setup for the Typer command path.
- [x] 2.8 Format backend and target list output to match the Typer-based CLI style.
- [x] 2.9 Wire discovered backend options into the Typer `check` command.

## 3. Tests

- [x] 3.1 Update CLI entry-point tests for Typer help, flags, dispatch, and exit codes.
- [x] 3.2 Update backend command tests for install, uninstall, list, and deprecated wrappers.
- [x] 3.3 Update API option discovery and logging tests.
- [x] 3.4 Update affected e2e expectations.
- [x] 3.5 Run OpenSpec validation for `typer-cli-improvement`.
- [x] 3.6 Add CLI coverage for dynamic backend option help and forwarding.
