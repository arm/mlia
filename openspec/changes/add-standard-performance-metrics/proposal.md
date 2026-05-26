## Why

MLIA already emits standardized JSON output, but several standardized performance fields are either missing from JSON or only available in stdout. The immediate requirement is to add those supported fields to MLIA standardized output without inventing metrics that MLIA cannot produce.

## What Changes

- Add known limitations and notes on interpreting results to standardized JSON output.
- Add percentage of operators executed on the accelerator to standardized JSON output.
- Add inference throughput to result-level metrics, derived from latency when latency is available.
- Add CPU utilization to result-level metrics when backend source data is available; otherwise represent it explicitly as unavailable.
- Add target utilization to result-level metrics when the required cycle data is available.
- Add peak activation memory to result-level metrics when a source value is available, or as an unavailable metric entry otherwise.
- Add average memory footprint to result-level metrics when a source value is available, or as an unavailable metric entry otherwise.
- Preserve existing compatibility, analysis metadata, and single-inference latency output while adding the missing fields.
- Explicitly represent requested fields without backend values as unavailable, without fabricating power, energy, hardware-measured runtime, or batch metrics.
- Define missing-data behavior for each new field.
- Update tests and sample or representative JSON payloads for the new fields.

## Out of Scope

- Do not add measured values for power, energy, hardware-measured runtime data, or batch latency and throughput in this change unless a backend provides suitable source data.
- Do not implement the Vela/Corstone "all stats" work here; that belongs in the owning plugin repository.
- Do not fix plugin-specific output bugs here except where a neutral fixture is needed to prove the core output contract.
- Do not include plugin-owned integration tasks in the initial core `mlia` PR; those should be specified in the owning plugin repositories once the core contract is stable.
- Do not replace MLIA's standardized output schema with a consumer-specific report schema.

## Capabilities

### New Capabilities

- `standard-performance-metrics`: Adds the supported standardized performance fields to MLIA standardized JSON output.

### Modified Capabilities

- None.

## Impact

- Core standardized output model, schema helpers, and JSON validation paths.
- JSON reporting and Python API collection paths that return standardized output.
- Tests for each added field, missing-data behavior, schema validation, and one representative output payload.
- Plugin follow-up work may be needed where a backend owns the source value extraction.
