## Context

Consumer input asks MLIA to expose supported model performance fields in standardized JSON output. MLIA's standardized output schema is the implementation contract, and this change updates it to schema version `1.1.0`. External consumer schemas may map from MLIA output, but they are not the implementation schema for this public core change.

This change is intentionally narrow. It covers the supported fields called out by the current work items:

- known limitations and interpretation notes
- percentage of operators executed on the accelerator
- inference throughput
- CPU utilization
- target utilization
- peak activation memory
- average memory footprint

MLIA already has schema locations for much of the broader requirement summary: analysis metadata is represented by top-level fields such as `timestamp`, `tool`, `target`, `model`, `backends`, and `context`; compatibility is represented by compatibility results, status, checks, warnings, and errors; and latency can already be emitted as a numeric result metric by backend plugins. This change should preserve those existing outputs while adding the missing supported fields. Compatibility, version, target, and evaluation-mode output may require consumer-specific mapping because MLIA deliberately keeps richer structured data than some report formats. Latency remains an existing plugin-provided metric in this change; current plugin naming and unit differences should be documented as a partial mapping rather than normalized by the core change.

The implementation should use a metric-by-metric mapping rather than assume every standardized field is new. Existing MLIA metrics that already satisfy a requirement should be documented with their existing names. Genuinely missing fields should be added with standard MLIA metric names and units. Throughput needs particular care because current plugins may already emit throughput-like metrics with backend-specific names.

The Vela/Corstone "all stats" work is related, but it is plugin-owned and should be specified separately in the owning plugin repository. Optional memory-profile fields such as model weight memory also belong to that later plugin-owned stats-completeness work, not this core standard-fields change.

## Goals / Non-Goals

**Goals:**

- Add the requested supported fields to MLIA standardized JSON output.
- Define the MLIA standardized metric mapping for the standardized performance metric set.
- Use schema-valid MLIA output locations rather than introducing a parallel consumer-shaped report.
- Put global single-inference numeric metrics under `results[*].metrics`.
- Explicitly mark requested fields without values rather than fabricating values.
- Define clear missing-data behavior for each added field.
- Add unit tests and representative JSON coverage for the new fields.
- Keep existing standardized JSON reporting and Python API collection workflows valid.

**Non-Goals:**

- Do not add measured values for power, energy, hardware-measured runtime, or batch performance fields when backend source data is absent.
- Do not make private target names, private product context, or consumer-specific requirements part of public MLIA artifacts.
- Do not implement backend-specific Vela/Corstone stats completeness in core MLIA.
- Do not fix plugin-specific bugs as part of the core change unless they block a neutral representative payload.

## Decisions

### MLIA standardized output remains authoritative

MLIA will represent the requested fields through the existing standardized output schema. Numeric global single-inference values belong in `results[*].metrics`. Notes and compatibility-adjacent information must use schema-valid non-metric locations, such as result warnings, checks, entities, or a documented extension if an existing field is not suitable.

Known limitations and interpretation notes should be represented as result warnings. A consumer that emits a consumer-shaped report can summarize or join those warnings into `analysisLimitations`.

Alternative considered: emit a separate consumer-shaped JSON document. That would duplicate MLIA's existing standardized output path and make reporting and API behavior harder to keep consistent.

### Result-level metrics are limited to global values

Accelerator operator percentage, inference throughput, CPU utilization, target utilization, peak activation memory, and average memory footprint are global result-level values and should be emitted under `results[*].metrics`.

Accelerator operator percentage is an operator-placement metric, not a measured runtime percentage. The standard MLIA metric name for this field is `accelerator_operator_percentage`. Its mapping documentation should make the placement semantics clear. The requirement context refers to compatibility-style pass, fail, and partial status, which is produced by compatibility analysis rather than performance analysis. Plugins should therefore populate this metric from compatibility/operator-placement data when that data is available, and should not infer it from a performance-only payload.

Operator-level or layer-level values are outside this core change unless they are needed as source data for one of the requested result-level metrics.

### Metric semantics are explicit

Inference throughput is derived from suitable latency data when available, but the backend or plugin owns the calculation because it owns the latency semantics. It uses the standard MLIA metric name `inferences_per_second` and unit `inferences/s`. CPU utilization uses the standard MLIA metric name `cpu_utilization`; MLIA must emit it as an availability-aware metric entry when no backend source can provide a supported value. Target utilization uses the standard MLIA metric name `target_utilization` and is calculated from suitable cycle data as `(compute_cycles / total_cycles) * 100 if total_cycles else 0.0`. Peak activation memory uses the standard MLIA metric name `peak_activation_memory` and means the highest activation memory usage during a single inference run. Average memory uses the standard MLIA metric name `average_memory` and means average memory usage over the full measurement window, not peak, allocation total, or instantaneous memory. The field tickets require these memory metrics to be added to result metrics; backends should emit numeric values where they own suitable source data and use explicit unavailable entries only where that source data is absent.

Percentage metrics in this field set use values in the range `0..100` with unit `%`. Backends and plugins should normalize source values into that convention before emitting standardized output.

Memory metrics in this field set use unit `bytes`. This follows existing MLIA metric conventions; consumers that emit consumer-shaped reports can map `bytes` to `B`.

### Standard names are generic MLIA names

Metric names and unit constants added by this work should describe the MLIA standardized output contract, not the first consuming workflow.

The identifiers selected here are MLIA standardized JSON metric names. They should be generic enough for multiple consumers, and MLIA should not emit consumer-specific field paths directly.

For existing MLIA metrics, the implementation should decide case by case whether to keep the existing name, rename it, or add an alias. Existing output should not be renamed blindly.

### Scoped MLIA mapping

This mapping is intentionally limited to the fields needed for the current standardized performance metrics requirement. It is not a full mapping to any consumer-specific schema.

| Scope | MLIA standardized output | Notes |
| --- | --- | --- |
| Analysis metadata | top-level `timestamp`, `tool`, `target`, `model`, `backends`, and `context` | Preserve existing structured metadata. |
| Compatibility | compatibility `results[*].status`, `checks`, and `entities` | Preserve the richer MLIA compatibility model, including non-boolean statuses. |
| Known limitations and interpretation notes | `results[*].warnings` | Keep notes result-scoped. |
| Single-inference latency | existing backend metric names | Preserve existing plugin output rather than defining a new standard latency metric in this change. |
| Inference throughput | metric `inferences_per_second`, unit `inferences/s` | Backend or plugin owns the latency semantics needed to calculate throughput. |
| Accelerator operator percentage | metric `accelerator_operator_percentage`, unit `%` | Operator-placement metric, not a measured runtime percentage. |
| CPU utilization | metric `cpu_utilization`, unit `%`, or an unavailable metric entry | Emit `unavailable` unless a backend can provide a real value. |
| Target utilization | metric `target_utilization`, unit `%` | Generic MLIA metric name for target utilization derived from suitable cycle data. |
| Peak activation memory | metric `peak_activation_memory`, unit `bytes`, or an unavailable metric entry | Emit numeric values only where the plugin owns suitable source data. |
| Average memory | metric `average_memory`, unit `bytes`, or an unavailable metric entry | Must mean average memory over the measurement window. |
| Scoped metric fields without values | availability-aware metric entries with `availability`, `unit`, and `reason` | MLIA must not fabricate values; markers are limited to the current standardized performance field set. |

Fields outside this mapping, including optional memory-profile and memory-traffic statistics, remain outside this core change. Existing backend-specific metrics should be preserved where already emitted, but new or standardized completeness work should be handled by a later plugin-owned statistics-completeness change if required.

### Missing data is represented in metrics

If MLIA or the selected backend cannot provide the source data for a requested field, the output must make that state clear using the chosen schema-valid representation. MLIA must not approximate or fabricate values.

Every metric field added or standardized by this change must be represented in `results[*].metrics` either as a numeric metric or as an availability-aware metric entry. This includes `accelerator_operator_percentage`, because it is covered by the MLIA work items even though it is sourced differently from ordinary performance metrics. Existing metadata, compatibility, and latency mappings should remain in their normal MLIA locations or be documented as partial mappings rather than represented as placeholder metric entries. This does not require MLIA to cover every possible consumer field.

This bounded scope should be documented as a limitation for MLIA output consumers so that users do not assume availability markers are comprehensive for every possible consumer field.

MLIA should extend the metric schema to support availability-aware metric entries. Supported metrics keep the existing numeric shape with `name`, `value`, and `unit`. Explicit `availability` is optional for supported metrics; if `value` is present and `availability` is omitted, consumers should treat the metric as available.

Added or standardized metric fields without values are represented in `results[*].metrics` with an explicit availability state, `unit`, and a short human-readable `reason`, without a fake numeric `value`. The initial implementation should use `unavailable` as the only non-value state. Add `unsupported` later only if implementation evidence shows MLIA needs to distinguish fields the selected backend cannot support from values that are supported in principle but unavailable for a particular run. Unit remains mandatory for availability entries in this contract, with canonical units defined in the mapping table; revisit only if implementation exposes a real unitless field.

The JSON schema should model this as a single metric array with two valid metric entry shapes: numeric entries with `value`, and non-value availability entries with `availability` and `reason`. This keeps consumers on one result-level metric list while making the absence of a numeric value explicit.

Alternative considered: put non-value fields in result checks or extensions. That avoids changing the metric schema, but it splits requested field availability away from the metric list consumers naturally inspect. Given the expected limited current consumer set, an availability-aware metric model is the cleaner contract as long as schema-version, compatibility notes, and tests are handled deliberately.

This change should bump the MLIA output schema version from `1.0.0` to `1.1.0`. Keeping `1.0.0` while adding non-value metric entries would silently change the contract for consumers that assume every metric entry has a numeric `value`.

### Plugin extraction stays plugin-owned

Core MLIA defines the output placement, validation expectations, standard metric names, units, and a shared helper for ensuring the added standard metrics are represented. Target and backend plugins remain responsible for extracting backend-specific source values and mapping them into the core output shape.

Plugins should call the core helper when constructing performance results. The helper should preserve supplied numeric metric values and add availability-aware entries for added or standardized metrics that are missing. The initial implementation should not run this helper automatically across all core reporting paths; plugin call sites should opt in explicitly. This keeps the common contract in core while avoiding duplicate unavailable-fill logic in each plugin.

The initial core `mlia` PR should define the shared contract and helper behavior only. Plugin-owned integration tasks are expected follow-up work, but they should be specified and reviewed in the owning plugin repositories rather than being part of the initial core PR.

The first representative plugin integration should use the Ethos-U Vela performance path. This proves the helper against a real backend with throughput, memory, and cycle data without expanding the initial implementation to every plugin. The Vela work should have a small `mlia-ethos-u` OpenSpec change that covers helper integration, backend source extraction, plugin-owned validation, and the result-level metrics overwrite fix if it blocks a trustworthy representative payload. Other plugin integrations should follow once the core shape is stable.

## Risks / Trade-offs

- The metric schema change means consumers must not assume every metric entry has a numeric `value`; the `1.1.0` schema version is the compatibility signal for that change.
- Availability states can become over-modeled; add distinct states only when implementation needs the distinction.
- Existing tests and plugin output code may need updates where they assume every metric has `name`, `value`, and `unit`.
- The design input suggested `results[*].details` for compatibility percentage, but the current MLIA schema does not define that field.
- Existing plugins already emit latency-like metrics with different names and at least one unit inconsistency risk, so latency should remain a documented partial mapping unless a separate standardization decision is made.
- Some requested values may not be available for every backend. Tests must cover missing-data behavior as well as supported values.
- Plugin-owned extraction may be needed before every requested field can be populated for every target.
- Using Vela as the first representative integration does not prove every plugin is complete; follow-up plugin work remains necessary.

## Follow-Up Work

- Create and implement a small `mlia-ethos-u` OpenSpec change for the Vela representative integration.
- Specify and implement the Vela/Corstone "all stats" work in the owning plugin repository.
- Track plugin-specific issues, such as result-level metrics being overwritten by breakdown metrics, in the owning plugin repository. If that overwrite issue blocks the Vela representative payload, fix it as part of the Vela integration rather than deferring it.
- Add plugin-specific payload tests once the core output placement is agreed.
