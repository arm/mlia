## 1. Output Placement

- [x] 1.1 Confirm `results[*].warnings` as the schema-valid location for known limitations and interpretation notes.
- [x] 1.2 Confirm that accelerator operator percentage is a result-level metric under `results[*].metrics`.
- [x] 1.3 Confirm that inference throughput, CPU utilization, target utilization, peak activation memory, and average memory footprint are result-level metrics under `results[*].metrics`.
- [x] 1.4 Update the metric schema shape so `results[*].metrics` can contain supported numeric metrics and explicit non-value metric entries.
- [x] 1.5 Model numeric metric entries and non-value availability entries as two valid shapes in the same `results[*].metrics` array.

## 2. Field Semantics

- [x] 2.1 Define the metric-by-metric mapping for the standardized performance metric set, including existing latency and throughput-like metrics.
- [x] 2.2 Define generic MLIA standardized metric names and units for any newly added metrics.
- [x] 2.3 Record MLIA standardized JSON metric names, units, and source ownership in the mapping.
- [x] 2.4 Prefer generic MLIA names for genuinely new metric names.
- [x] 2.5 Document existing backend-provided latency metrics as a partial mapping rather than adding a new standard latency metric in this change.
- [x] 2.6 Document existing compatibility output as MLIA-native status, check, and entity output.
- [x] 2.7 Document `tool.version` and `backends[*].version` as separate MLIA standardized output fields.
- [x] 2.8 Document MLIA's structured `target` output.
- [x] 2.9 Document MLIA `results[*].mode` without requiring it to match an external enum.
- [x] 2.10 Define inference throughput as `inferences_per_second` with unit `inferences/s`, a backend/plugin-provided result-level metric derived from suitable latency data where applicable.
- [x] 2.11 Define CPU utilization as `cpu_utilization` with unit `%`, emitted as `unavailable` when no backend source can provide a real value.
- [x] 2.12 Define target utilization as `target_utilization`, calculated as `(compute_cycles / total_cycles) * 100 if total_cycles else 0.0` from suitable cycle data.
- [x] 2.13 Define peak activation memory as `peak_activation_memory`, meaning the highest activation memory usage during a single inference run.
- [x] 2.14 Define average memory as `average_memory`, meaning average memory usage over the full measurement window rather than peak, allocation total, or instantaneous memory.
- [x] 2.15 Define accelerator operator percentage as an operator-placement metric, not a measured runtime percentage.
- [x] 2.16 Define percentage metrics as `0..100` values with unit `%`.
- [x] 2.17 Define memory metrics as using unit `bytes`.
- [x] 2.18 Define missing-data behavior for each new field.
- [x] 2.19 Define the added or standardized metric field set that must always be represented as numeric metrics or explicit non-value markers, including `accelerator_operator_percentage`.
- [x] 2.20 Define non-value field behavior for those metric fields without expanding scope to existing metadata, compatibility, latency mappings, or every possible consumer field.
- [x] 2.21 Use `unavailable` as the only initial non-value availability state.
- [x] 2.22 Defer an `unsupported` state unless implementation evidence shows a separate state is needed.
- [x] 2.23 Define the required and optional fields for non-numeric metric availability entries; `unit` and `reason` are required for this contract unless implementation exposes a real exception.
- [x] 2.24 Document that supported metrics may omit `availability`, and omitted availability means available when `value` is present.

## 3. Core Implementation

- [x] 3.1 Add the chosen `results[*].warnings` output representation for known limitations and interpretation notes.
- [x] 3.2 Add or document `accelerator_operator_percentage` as the result-level metric name for percentage of operators executed on the accelerator.
- [x] 3.3 Add or document `cpu_utilization` as the result-level metric name for CPU utilization, including unavailable output when no backend source is available.
- [x] 3.4 Add or document `target_utilization` as the result-level metric name for target utilization.
- [x] 3.5 Add or document `inferences_per_second` as the result-level metric name for inference throughput.
- [x] 3.6 Add or document `average_memory` as the result-level metric name for average memory footprint.
- [x] 3.7 Add or document `peak_activation_memory` as the result-level metric name for peak activation memory.
- [x] 3.8 Add shared metric name and unit constants where the implementation introduces or standardizes string-keyed metric names.
- [x] 3.9 Add a core helper that preserves supplied numeric values and fills missing added standard metrics as availability-aware entries.
- [x] 3.10 Document that plugins should call the core helper explicitly when constructing performance results that should include the added standard metrics.
- [x] 3.11 Ensure JSON reporting and Python API collection preserve the added fields.
- [x] 3.12 Preserve existing compatibility, analysis metadata, and single-inference latency output.
- [x] 3.13 Document that non-value markers are limited to the standardized performance field set for this change.
- [x] 3.14 Bump the MLIA output schema version from `1.0.0` to `1.1.0` for availability-aware metric entries.
- [x] 3.15 Add compatibility documentation explaining that `1.1.0` metrics may be numeric values or non-value availability entries.

## 4. Validation

- [x] 4.1 Add unit tests for each new field.
- [x] 4.2 Add tests for missing-data behavior for each new field.
- [x] 4.3 Add schema validation coverage for representative standardized output.
- [x] 4.4 Check whether maintained sample JSON or representative fixtures need
  updating. No canonical committed sample JSON artifact was identified for this
  slice, so representative JSON shape is covered by tests, schema validation,
  and documentation instead of adding a large generated output file.
- [x] 4.5 Validate the Ethos-U Vela performance path as the representative plugin integration where source data is available.
- [x] 4.6 Add coverage for non-value metric representation without fabricated values.
- [x] 4.7 Check user-facing documentation or release notes describe the non-value marker scope limitation.

## 5. Plugin Follow-Up Coordination

- [x] 5.1 Record any backend-specific extraction work needed to populate the new fields.
- [x] 5.2 Create a small `mlia-ethos-u` OpenSpec change for the Vela representative integration.
- [x] 5.3 Record the Vela result-level metrics overwrite issue as plugin-owned and fix it in the Vela integration branch because it blocks a trustworthy representative payload.
- [x] 5.4 Record Vela/Corstone all-stats completeness as separate future plugin-owned work.
- [x] 5.5 Keep plugin-owned integration tasks out of the initial core `mlia` PR; specify them in the owning plugin repositories once the core contract is stable.
- [x] 5.6 Track any private plugin integration in the owning private repository rather than in this public core PR.
- [x] 5.7 Track the plugin call-site changes in the owning plugin repositories rather than in this core PR.
