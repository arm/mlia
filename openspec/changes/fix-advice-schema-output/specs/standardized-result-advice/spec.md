## ADDED Requirements

### Requirement: Result-level advice field
MLIA standardized output schema `1.1.0` SHALL allow each result object to include an optional `advice` array.

#### Scenario: Result contains advice
- **WHEN** a standardized output payload contains a result object with `advice`
- **THEN** schema `1.1.0` accepts the payload when each advice entry satisfies the advice entry requirements

#### Scenario: Result omits advice
- **WHEN** a standardized output payload contains a result object without `advice`
- **THEN** schema `1.1.0` accepts the payload

### Requirement: Advice entry shape
Each advice entry in `results[*].advice` SHALL include `id`, `category`, `severity`, and `message`.

#### Scenario: Advice entry contains required fields
- **WHEN** an advice entry contains `id`, `category`, `severity`, and `message` with valid values
- **THEN** schema `1.1.0` accepts the advice entry

#### Scenario: Advice entry misses a required field
- **WHEN** an advice entry omits one of `id`, `category`, `severity`, or `message`
- **THEN** schema `1.1.0` rejects the advice entry

### Requirement: Advice category and severity values
Advice `category` and `severity` values in standardized JSON output SHALL use lower-case schema values.

#### Scenario: Advice uses lower-case schema values
- **WHEN** advice is serialized for standardized JSON output
- **THEN** `category` is serialized as one of `compatibility`, `performance`, or `optimization`
- **AND** `severity` is serialized as one of `info`, `warning`, or `error`

#### Scenario: Advice uses enum member names
- **WHEN** a standardized output payload contains advice with upper-case `category` or `severity` values
- **THEN** schema `1.1.0` rejects the advice entry

### Requirement: Advice affected entities
Advice entries MAY include `affected_entities`, and each affected entity SHALL use the existing operator identifier shape.

#### Scenario: Advice identifies affected operators
- **WHEN** an advice entry contains `affected_entities`
- **THEN** each affected entity contains `scope`, `name`, and `location`
- **AND** schema `1.1.0` accepts optional `id` on each affected entity

### Requirement: Advice details
Advice entries that include `details` MUST encode it as an object for advice-specific metadata.

#### Scenario: Advice includes details
- **WHEN** an advice entry contains a `details` object
- **THEN** schema `1.1.0` accepts the object without constraining its keys

### Requirement: Legacy advice field is rejected
Schema `1.1.0` SHALL reject the legacy `results[*].advices` field.

#### Scenario: Result contains legacy advice field
- **WHEN** a standardized output payload contains `results[*].advices`
- **THEN** schema `1.1.0` rejects the payload

### Requirement: Schema 1.0 remains unchanged
This change SHALL NOT modify `mlia-output-schema-1.0.0.json`.

#### Scenario: Schema 1.0 file is checked
- **WHEN** the change is reviewed
- **THEN** `mlia-output-schema-1.0.0.json` has no content changes
