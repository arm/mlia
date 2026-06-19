## MODIFIED Requirements

### Requirement: Standard metric mapping

MLIA SHALL define stable MLIA standardized-output names and units for
string-keyed metrics in the standardized performance metric set.

#### Scenario: MLIA metric names are documented

- **WHEN** MLIA documents the standardized performance metric set
- **THEN** the mapping records the MLIA standardized JSON metric name, unit, and
  source ownership.

#### Scenario: Existing metric satisfies requirement

- **WHEN** an existing MLIA metric already satisfies a standardized performance
  requirement
- **THEN** MLIA documents that existing metric name and unit in the mapping
  rather than adding a duplicate metric name without justification.

#### Scenario: New metric is added

- **WHEN** MLIA adds a new metric for the standardized performance metric set
- **THEN** the metric uses a generic MLIA standardized-output name and unit.

#### Scenario: Existing metric overlaps a standard field

- **WHEN** MLIA already emits a metric that represents the same or similar data
  as a standard performance metric
- **THEN** MLIA decides explicitly whether to keep the existing metric, add the
  standard metric, or emit both, instead of renaming existing output by default.

#### Scenario: Percentage metric unit convention

- **WHEN** MLIA emits a percentage metric in the standardized performance field
  set
- **THEN** the metric value uses the range `0..100` and unit `%`.

#### Scenario: Inference latency emitted

- **WHEN** a backend or plugin can provide inference latency from suitable
  source data
- **THEN** MLIA emits `inference_time` as a result-level metric under
  `results[*].metrics` with unit `ms`.

#### Scenario: Standard metric unit mismatch rejected

- **WHEN** a plugin passes a metric with a standard performance metric name to
  the standard metric helper
- **AND** the metric unit does not match the standard metric definition
- **THEN** MLIA rejects the metric instead of preserving an ambiguous value.

#### Scenario: Model weight memory emitted

- **WHEN** model weight memory is available for a single inference run
- **THEN** MLIA emits `model_weight_memory` as a result-level metric under
  `results[*].metrics` with unit `bytes`.

#### Scenario: Model weight memory unavailable

- **WHEN** MLIA or the selected backend cannot provide supported model weight
  memory source data
- **THEN** MLIA emits `model_weight_memory` as an availability-aware metric
  entry with unit `bytes` and a reason.

#### Scenario: Peak activation memory emitted

- **WHEN** peak activation memory is available for a single inference run
- **THEN** MLIA emits `peak_activation_memory` as a result-level metric under
  `results[*].metrics` with unit `bytes`.

#### Scenario: Peak activation memory unavailable

- **WHEN** MLIA or the selected backend cannot provide supported peak activation
  memory source data
- **THEN** MLIA emits `peak_activation_memory` as an availability-aware metric
  entry with unit `bytes` and a reason.

#### Scenario: Average memory footprint emitted

- **WHEN** average memory usage over the full measurement window is available
  for a single inference run
- **THEN** MLIA emits `average_memory` as a result-level metric under
  `results[*].metrics` with unit `bytes`.

#### Scenario: Average memory footprint unavailable

- **WHEN** MLIA or the selected backend cannot provide supported average memory
  source data
- **THEN** MLIA emits `average_memory` as an availability-aware metric entry
  with unit `bytes` and a reason.

### Requirement: Missing data behavior

MLIA SHALL define and test missing-data behavior for each added field.

#### Scenario: Source data is missing

- **WHEN** MLIA or the selected backend cannot provide the source data for an
  added field
- **THEN** MLIA does not fabricate a value and represents the missing-data state
  as an availability-aware metric entry.

#### Scenario: Added metric field is always represented

- **WHEN** MLIA emits standardized output for a result covered by this change
- **THEN** each added or standardized metric field is present in
  `results[*].metrics` either as a numeric metric or as an availability-aware
  metric entry.

#### Scenario: Core helper fills unavailable metric entries

- **WHEN** a plugin passes performance result metrics through the core helper
  for added standard performance fields
- **THEN** supplied numeric values are preserved and missing added standard
  metrics are filled as availability-aware metric entries.

#### Scenario: Inference latency unavailable

- **WHEN** MLIA or the selected backend cannot provide inference latency source
  data
- **THEN** MLIA emits `inference_time` as an availability-aware metric entry
  with unit `ms` and a reason.

#### Scenario: Core helper is called explicitly

- **WHEN** a performance result is not passed through the core helper
- **THEN** core reporting does not automatically add availability-aware metric
  entries.

#### Scenario: Plugin integration is follow-up work

- **WHEN** the initial core `mlia` PR is implemented
- **THEN** it defines the shared output contract and helper behavior without
  including plugin-owned integration tasks.

#### Scenario: Metric value is available

- **WHEN** MLIA emits a supported metric value
- **THEN** the metric keeps the existing numeric shape with `name`, `value`, and
  `unit`, and omitted `availability` means available.

#### Scenario: Metric value has no numeric value

- **WHEN** MLIA emits a standardized performance metric without a numeric value
- **THEN** MLIA emits a metric entry with `availability` set to `unavailable`,
  `unit`, and reason, without a fake numeric `value`.

#### Scenario: Metric array supports both metric shapes

- **WHEN** MLIA validates `results[*].metrics`
- **THEN** the schema accepts numeric metric entries and non-value availability
  entries in the same metrics array.

#### Scenario: Additional non-value states are deferred

- **WHEN** implementation needs to distinguish metrics that are unsupported
  from metrics that are supported in principle but unavailable for a particular
  run
- **THEN** MLIA adds a separate availability state in a later change rather than
  requiring it in the initial implementation.
