## ADDED Requirements

### Requirement: Backend layout validation precedes payload selection

The system SHALL validate configured backend layout expected files before
selecting any backend payload folder for installation.

#### Scenario: Valid package root with selected payload folders

- **WHEN** a backend package contains all configured package-root expected files
  and each configured payload folder exists
- **THEN** the backend package is accepted as installable

#### Scenario: Missing expected file

- **WHEN** a backend layout is missing a configured expected file
- **THEN** the backend layout is rejected before any payload folder is
  selected

### Requirement: Multiple backend payload folders can be installed from one matched layout

The system SHALL allow backend installation configuration to select multiple
backend payload folders from the same matched package layout.

#### Scenario: Installing primary and supporting payload folders

- **WHEN** a backend package is configured to install a primary payload folder
  and an additional supporting payload folder from the same package root
- **THEN** the installed backend repository entry contains both configured
  payload folders from that package

#### Scenario: Installing deeply nested supporting payload folder

- **WHEN** a configured supporting payload folder contains nested directories
  and many files
- **THEN** the installed backend repository entry contains the complete
  supporting payload folder tree

#### Scenario: Missing configured payload folder

- **WHEN** a backend layout is configured with multiple payload folders and any
  configured payload folder is missing or is not a directory
- **THEN** the backend layout is rejected as not installable

### Requirement: Compound path checking selects one complete layout

The system SHALL keep compound path checking as first-match selection between
alternative backend layouts.

#### Scenario: Earlier package layout matches before later layout

- **WHEN** a compound path checker contains a matching package checker before a
  later matching checker
- **THEN** the first package checker result is used as the complete backend
  install source set

#### Scenario: First matched checker includes multiple payload folders

- **WHEN** the first matched checker selects multiple payload folders
- **THEN** all payload folders from that matched checker are installed and later
  checker results are ignored

### Requirement: Existing single-source behavior is preserved

The system SHALL preserve existing `PackagePathChecker` behavior for backend
layouts configured with no payload folder override or one payload folder
override.

#### Scenario: No backend payload folder configured

- **WHEN** a backend layout is configured without a payload folder override and
  the layout root contains all expected files
- **THEN** the layout root is used as the backend install source

#### Scenario: One package payload folder configured

- **WHEN** a backend package is configured with one payload folder and that
  folder exists
- **THEN** the single folder remains the backend install source
