## Why

MLIA's standardized performance metrics define stable names and units for values
that downstream consumers can process without knowing backend-specific field
names. Inference latency is currently only partially covered by plugin-provided
metrics, and existing helpers do not reject a standard metric name emitted with a
non-standard unit.

This leaves room for drift: a plugin can emit a latency-like metric in seconds
or milliseconds under a backend-specific name, while another path derives
throughput from a value with a different implied unit. The standardized metric
set should include latency directly and should reject standard metric names when
their units do not match the shared contract.

## What Changes

- Add `inference_time` to the standard performance metric set with unit `ms`.
- Fill missing `inference_time` values as availability-aware metric entries
  when no backend source value is available.
- Validate supplied standard performance metrics against their standard units.
- Document `inference_time` in the standardized output metric list.
- Add tests for missing latency output and unit validation.

## Out of Scope

- Do not change the JSON schema shape or output schema version; this uses the
  existing metric and availability-entry shapes.
- Do not invent latency values when a backend cannot provide source data.
- Do not change plugin-owned latency extraction in the core repository beyond
  the shared standard metric contract.
- Do not remove backend-specific metrics that are still needed for compatibility.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `standard-performance-metrics`: Extends the standardized performance metric
  set with inference latency and standard-unit validation.

## Impact

- Core standardized output metric constants and helper behavior.
- Documentation for standardized metrics.
- Tests for latency availability entries and unit validation.
- Plugin repositories that emit standard performance metric names must use the
  standard units.
