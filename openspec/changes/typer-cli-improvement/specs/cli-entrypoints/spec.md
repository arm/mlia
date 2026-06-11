## ADDED Requirements

### Requirement: Typer command tree
MLIA SHALL expose a Typer-based root CLI with `check`, `backend`, and `target` commands.

#### Scenario: Root command has no arguments
- **WHEN** a user runs `mlia` without arguments
- **THEN** the CLI shows help
- **AND** exits with status code 2

#### Scenario: Check command has no arguments
- **WHEN** a user runs `mlia check` without arguments
- **THEN** the CLI shows help for the check command
- **AND** exits with status code 2

### Requirement: Check command options
The `mlia check` command SHALL keep the user-facing check workflow available through Typer option parsing.

#### Scenario: Check command runs with required target profile
- **WHEN** a user runs `mlia check MODEL --target-profile TARGET`
- **THEN** MLIA validates the requested checks for the target profile
- **AND** invokes advice generation when at least one requested check can run

#### Scenario: Requested checks cannot run
- **WHEN** target profile validation reports that no requested checks can run
- **THEN** the CLI exits with status code 0
- **AND** does not invoke advice generation

### Requirement: Backend and target command groups
MLIA SHALL expose backend and target management under the root command.

#### Scenario: Backend commands are available
- **WHEN** a user asks for `mlia backend --help`
- **THEN** the CLI lists `install`, `uninstall`, and `list`

#### Scenario: Target commands are available
- **WHEN** a user asks for `mlia target --help`
- **THEN** the CLI lists `list`

### Requirement: Deprecated script entry points
The legacy `mlia-backend` and `mlia-target` scripts SHALL remain callable and warn users to use the root subcommands.

#### Scenario: Legacy backend entry point is used
- **WHEN** a user invokes `mlia-backend`
- **THEN** the CLI writes a deprecation warning that points to `mlia backend`
- **AND** dispatches to the backend command app

#### Scenario: Legacy target entry point is used
- **WHEN** a user invokes `mlia-target`
- **THEN** the CLI writes a deprecation warning that points to `mlia target`
- **AND** dispatches to the target command app

### Requirement: Plugin discovery help
The root CLI help SHALL include plugin discovery guidance.

#### Scenario: Root help is shown
- **WHEN** a user runs `mlia --help`
- **THEN** the help output mentions `mlia target list`
- **AND** mentions `mlia backend list`
- **AND** includes the MLIA repository discovery URL

### Requirement: Backend management API
MLIA SHALL expose public API functions for backend install and uninstall operations.

#### Scenario: Backend install is requested
- **WHEN** the CLI handles `mlia backend install`
- **THEN** it calls the public backend install API with the requested names, path, EULA, noninteractive, and force settings

#### Scenario: Backend EULA acceptance is requested
- **WHEN** a user runs `mlia backend install` with `--accept-eula`
- **THEN** MLIA passes EULA acceptance to the backend install API

#### Scenario: Backend uninstall is requested
- **WHEN** the CLI handles `mlia backend uninstall`
- **THEN** it calls the public backend uninstall API with the requested names

### Requirement: Consistent list output formatting
Backend and target list commands SHALL use structured table output that matches the Typer-based CLI style.

#### Scenario: Backend list is requested
- **WHEN** a user runs `mlia backend list`
- **THEN** MLIA shows backend plugin information and backend install status in table output

#### Scenario: Target list is requested
- **WHEN** a user runs `mlia target list`
- **THEN** MLIA shows target plugin information and built-in target profiles in table output

### Requirement: CLI structure supports Typer command wiring
MLIA SHALL define CLI commands through Typer apps and command functions rather than the removed argparse option registration module.

#### Scenario: Commands are initialized
- **WHEN** the CLI entry point starts
- **THEN** command dispatch is provided by the Typer root, backend, and target apps

#### Scenario: CLI command registration is inspected
- **WHEN** maintainers inspect CLI command wiring
- **THEN** MLIA no longer relies on `CommandInfo` argparse registration for built-in CLI commands

#### Scenario: Backend option discovery is requested
- **WHEN** API callers request backend option metadata
- **THEN** MLIA discovers backend options without importing the removed CLI options module
