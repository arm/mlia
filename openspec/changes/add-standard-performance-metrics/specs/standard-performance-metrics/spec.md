## ADDED Requirements

### Requirement: Standardized performance fields

MLIA SHALL add the supported standardized performance fields to standardized JSON output using MLIA's standardized output schema.

#### Scenario: Existing supported fields preserved

- **WHEN** MLIA emits standardized performance output
- **THEN** existing compatibility, analysis metadata, and single-inference latency output remain available through MLIA's standardized output schema.

#### Scenario: Existing latency mapping documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** existing backend-provided latency metrics are documented as a partial mapping unless a separate standard latency name and unit are defined.

#### Scenario: Existing compatibility mapping documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** existing compatibility output is documented as a richer MLIA-native structure with status, check, and entity output.

#### Scenario: Existing analyzer version mapping documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** `tool.version` and `backends[*].version` remain separate in MLIA standardized output.

#### Scenario: Existing target mapping documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** MLIA's structured `target` output remains available in MLIA standardized output.

#### Scenario: Existing evaluation type mapping documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** MLIA `results[*].mode` remains available without requiring it to match an external consumer enum.

#### Scenario: Interpretation notes emitted

- **WHEN** a result has known limitations or notes needed to interpret performance data
- **THEN** MLIA includes those notes in standardized JSON output using `results[*].warnings`.

#### Scenario: Accelerator operator percentage emitted

- **WHEN** MLIA can determine the percentage of operators executed on the accelerator from compatibility or operator-placement data
- **THEN** MLIA emits `accelerator_operator_percentage` as a result-level metric under `results[*].metrics` with a `0..100` value and unit `%`.

#### Scenario: Inference throughput emitted

- **WHEN** a backend or plugin can provide inference throughput from suitable source data
- **THEN** MLIA emits `inferences_per_second` as a result-level metric under `results[*].metrics` with unit `inferences/s`.

#### Scenario: Target utilization emitted

- **WHEN** a backend or plugin can provide target utilization from suitable source data
- **THEN** MLIA emits `target_utilization` as a result-level metric under `results[*].metrics` with a `0..100` value and unit `%`, calculated as `(compute_cycles / total_cycles) * 100 if total_cycles else 0.0`.

#### Scenario: CPU utilization emitted

- **WHEN** a backend or plugin can provide CPU utilization from suitable source data
- **THEN** MLIA emits `cpu_utilization` as a result-level metric under `results[*].metrics` with a `0..100` value and unit `%`.

#### Scenario: CPU utilization unavailable

- **WHEN** MLIA or the selected backend cannot provide CPU utilization source data
- **THEN** MLIA emits `cpu_utilization` as an availability-aware metric entry by default with `availability` set to `unavailable`, unit `%`, and a reason.

### Requirement: Standard metric mapping

MLIA SHALL define stable MLIA standardized-output names and units for string-keyed metrics in the standardized performance metric set.

#### Scenario: MLIA metric names are documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** the mapping records the MLIA standardized JSON metric name, unit, and source ownership.

#### Scenario: Existing metric satisfies requirement

- **WHEN** an existing MLIA metric already satisfies a standardized performance requirement
- **THEN** MLIA documents that existing metric name and unit in the mapping rather than adding a duplicate metric name without justification.

#### Scenario: New metric is added

- **WHEN** MLIA adds a new metric for the standardized performance metric set
- **THEN** the metric uses a generic MLIA standardized-output name and unit.

#### Scenario: Existing metric overlaps requirement

- **WHEN** an existing MLIA metric overlaps a standardized performance field
- **THEN** MLIA decides case by case whether to keep the existing name, rename it, or add an alias rather than renaming existing output blindly.

#### Scenario: Percentage metric unit convention

- **WHEN** MLIA emits a percentage metric in the standardized performance field set
- **THEN** the metric value uses the range `0..100` and unit `%`.

#### Scenario: Peak activation memory emitted

- **WHEN** peak activation memory is available for a single inference run
- **THEN** MLIA emits `peak_activation_memory` as a result-level metric under `results[*].metrics` with unit `bytes`.

#### Scenario: Peak activation memory unavailable

- **WHEN** MLIA or the selected backend cannot provide supported peak activation memory source data
- **THEN** MLIA emits `peak_activation_memory` as an availability-aware metric entry with unit `bytes` and a reason.

#### Scenario: Average memory footprint emitted

- **WHEN** average memory usage over the full measurement window is available for a single inference run
- **THEN** MLIA emits `average_memory` as a result-level metric under `results[*].metrics` with unit `bytes`.

#### Scenario: Average memory footprint unavailable

- **WHEN** MLIA or the selected backend cannot provide supported average memory source data
- **THEN** MLIA emits `average_memory` as an availability-aware metric entry with unit `bytes` and a reason.

### Requirement: Missing data behavior

MLIA SHALL define and test missing-data behavior for each added field.

#### Scenario: Source data is missing

- **WHEN** MLIA or the selected backend cannot provide the source data for an added field
- **THEN** MLIA does not fabricate a value and represents the missing-data state as an availability-aware metric entry.

#### Scenario: Added metric field is always represented

- **WHEN** MLIA emits standardized output for a result covered by this change
- **THEN** each added or standardized metric field is present in `results[*].metrics` either as a numeric metric or as an availability-aware metric entry.

#### Scenario: Core helper fills unavailable metric entries

- **WHEN** a plugin passes performance result metrics through the core helper for added standard performance fields
- **THEN** supplied numeric values are preserved and missing added standard metrics are filled as availability-aware metric entries.

#### Scenario: Core helper is called explicitly

- **WHEN** a performance result is not passed through the core helper
- **THEN** core reporting does not automatically add availability-aware metric entries.

#### Scenario: Plugin integration is follow-up work

- **WHEN** the initial core `mlia` PR is implemented
- **THEN** it defines the shared output contract and helper behavior without including plugin-owned integration tasks.

#### Scenario: Representative plugin integration uses Vela

- **WHEN** the initial implementation proves plugin integration
- **THEN** it uses the Ethos-U Vela performance path as the representative plugin call site.

#### Scenario: Vela integration has plugin OpenSpec

- **WHEN** Vela helper integration is planned
- **THEN** the plugin-owned behavior is specified in a small `mlia-ethos-u` OpenSpec change.

#### Scenario: Vela representative payload preserves result metrics

- **WHEN** the Vela representative integration emits performance output
- **THEN** result-level metrics are preserved and are not overwritten by breakdown metric construction.

#### Scenario: Metric value is available

- **WHEN** MLIA emits a supported metric value
- **THEN** the metric keeps the existing numeric shape with `name`, `value`, and `unit`, and omitted `availability` means available.

#### Scenario: Metric value has no numeric value

- **WHEN** MLIA emits a standardized performance metric without a numeric value
- **THEN** MLIA emits a metric entry with `availability` set to `unavailable`, `unit`, and reason, without a fake numeric `value`.

#### Scenario: Metric array supports both metric shapes

- **WHEN** MLIA validates `results[*].metrics`
- **THEN** the schema accepts numeric metric entries and non-value availability entries in the same metrics array.

#### Scenario: Additional non-value states are deferred

- **WHEN** implementation needs to distinguish metrics that are unsupported from metrics that are supported in principle but unavailable for a particular run
- **THEN** MLIA adds a separate availability state in a later change rather than requiring it in the initial implementation.

### Requirement: Non-value fields are explicit

MLIA SHALL explicitly represent standardized performance fields that do not have numeric values without fabricating metric values.

#### Scenario: Non-value field is in the standardized performance field set

- **WHEN** a metric field added or standardized by this change cannot be produced by MLIA or the selected backend
- **THEN** MLIA marks that field using an availability-aware metric entry rather than omitting it silently or emitting a placeholder numeric value.

#### Scenario: Ticket-covered optional design input field is unavailable

- **WHEN** `accelerator_operator_percentage` cannot be produced by MLIA or the selected backend
- **THEN** MLIA marks that field using an availability-aware metric entry even though the corresponding design-input row is not marked required.

#### Scenario: Existing mapped fields do not get metric placeholders

- **WHEN** an existing metadata, compatibility, or latency field only maps partially to the design input
- **THEN** MLIA keeps the field in its normal standardized output location and does not emit a placeholder metric entry for that mapping gap.

#### Scenario: Field is outside the standardized performance field set

- **WHEN** a field is not part of the standardized performance field set for this work
- **THEN** MLIA does not need to represent that field as part of this change.

#### Scenario: Optional memory-profile stats belong to later plugin work

- **WHEN** a field such as `memoryProfile.modelWeightMemory` belongs to Vela/Corstone statistics-completeness work
- **THEN** MLIA does not need to represent that field as part of this core standard-fields change.

#### Scenario: Non-value marker scope is documented

- **WHEN** MLIA documents the added standardized performance fields
- **THEN** the documentation states that non-value markers are limited to the standardized performance field set for this change and are not comprehensive for every possible consumer field.

#### Scenario: Availability-aware metric contract is documented

- **WHEN** MLIA documents the added standardized performance fields
- **THEN** the documentation explains that metrics may be numeric values or explicit availability entries.

### Requirement: Existing reporting remains valid

MLIA SHALL add the requested fields without regressing existing standardized JSON reporting or Python API collection workflows.

#### Scenario: Schema version identifies availability-aware metrics

- **WHEN** MLIA emits standardized output with availability-aware metric entries
- **THEN** the output uses MLIA output schema version `1.1.0`.

#### Scenario: Standardized output validates

- **WHEN** MLIA emits standardized output containing the added fields
- **THEN** the output validates against MLIA's standardized output schema.
