## ADDED Requirements

### Requirement: Lightweight generic shell completion
MLIA shell completion SHALL construct root command and option-name completions
without initializing runtime-only MLIA surfaces that are unnecessary for the
command tree.

#### Scenario: Root command completion initializes only the command tree
- **WHEN** shell completion is requested for the MLIA root command
- **THEN** MLIA SHALL suggest the matching static command names
- **AND** MLIA core SHALL NOT import the MLIA API, backend installation, target
  configuration, target registry, advice generation, or output validation merely
  to construct the command tree
- **AND** installed backend plugins MAY import dependencies they own while
  registering their options

#### Scenario: Check option-name completion includes backend plugin options
- **WHEN** shell completion is requested for option names on `mlia check`
- **THEN** MLIA SHALL suggest matching static and plugin-provided check option
  names
- **AND** SHALL obtain plugin-provided option metadata through the existing
  backend package initialization boundary
- **AND** MLIA core SHALL NOT import the MLIA API, backend installation, or target
  runtime merely to construct those option names

#### Scenario: Normal check help retains backend options
- **WHEN** `mlia check --help` is requested outside shell completion
- **THEN** MLIA SHALL discover and display the same plugin-provided backend
  options

### Requirement: Registered selectable backend completion
MLIA shell completion SHALL suggest matching backend names registered by
available backend plugins whose configurations are selectable. MLIA core SHALL
NOT query backend installation state while constructing these candidates.

#### Scenario: Available plugin registrations provide backend names
- **WHEN** completion is requested for a backend-valued CLI argument
- **THEN** MLIA SHALL use backend registrations populated through the existing
  backend package initialization boundary
- **AND** SHALL include matching plugin-provided names whose configurations are
  selectable
- **AND** SHALL exclude registrations whose configurations are not selectable

#### Scenario: Backend install value completion is state independent
- **WHEN** shell completion is requested for `mlia backend install`
- **THEN** MLIA SHALL suggest matching registered selectable backend names
- **AND** SHALL NOT filter names according to whether they are already installed

#### Scenario: Backend uninstall value completion is state independent
- **WHEN** shell completion is requested for `mlia backend uninstall`
- **THEN** MLIA SHALL suggest matching registered selectable backend names
- **AND** SHALL NOT filter names according to whether they are currently installed

#### Scenario: Check backend option completion uses the same inventory
- **WHEN** shell completion is requested for `mlia check --backend`
- **THEN** MLIA SHALL suggest matching registered selectable backend names from
  the same state-independent inventory

#### Scenario: Backend candidates are deterministic
- **WHEN** multiple selectable backend names match the current word
- **THEN** MLIA SHALL return the matching names in sorted order

### Requirement: Packaged target profile completion
MLIA shell completion SHALL suggest matching MLIA-packaged target profile names
without initializing the target runtime.

#### Scenario: Target profile names come from namespace resources
- **WHEN** completion is requested for `mlia check --target-profile`
- **THEN** MLIA SHALL scan `resources/target_profiles/*.toml` beneath active
  MLIA namespace paths
- **AND** SHALL return matching profile stems in sorted, deduplicated order
- **AND** the target-profile inventory SHALL NOT import target configuration,
  target registry, or target plugin modules

### Requirement: Shell fallback on no MLIA matches
MLIA shell completion SHALL return no MLIA candidates when no completion value
matches, allowing the invoking shell to apply its normal fallback behavior.

#### Scenario: Target profile path fallback
- **WHEN** completion is requested for `mlia check --target-profile`
- **AND** no packaged target profile name matches the current word
- **THEN** MLIA SHALL return no candidates
- **AND** the shell MAY apply filename completion for custom profile paths

#### Scenario: No sentinel candidate is returned
- **WHEN** shell completion has no matching MLIA value
- **THEN** MLIA SHALL NOT return a space, empty string, or other sentinel value
  as a completion candidate
