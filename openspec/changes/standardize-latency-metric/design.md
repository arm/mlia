## Context

The standardized performance metric helper currently fills missing metrics with
availability-aware entries. It does not validate existing metrics, so a plugin
can pass a metric with a standard name and a non-standard unit. That makes the
metric name less useful as a contract.

Latency is the remaining required performance value that has no standard MLIA
metric name and unit. Existing plugin output already exposes latency-like data,
but the names and units are backend-specific.

## Decision

Add `inference_time` as the MLIA standard performance metric for inference
latency, using unit `ms`.

Update the standard metric helper so it validates any supplied metric whose name
matches a standard performance metric. If the unit differs from the standard
definition, the helper raises `ValueError` before adding missing entries.

The helper continues to preserve non-standard metrics and continues to add
availability-aware entries only when called explicitly.

## Consequences

Plugins that already emit `inference_time` with unit `ms` satisfy the new
contract.

Plugins that emit a standard metric name with a different unit must convert the
value before calling the helper or keep the backend-specific metric name for
legacy output.

Existing consumers of the standardized metric helper get a new unavailable
`inference_time` entry unless they supply a real value.
