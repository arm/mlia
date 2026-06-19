<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Outputs

## Overview

MLIA can present analysis results in human-readable text form or as structured
JSON. The core `mlia` repository owns the standardized result shape and the
high-level user experience around it, while the split plugin repositories
document the detailed meaning of backend-specific metrics.

## Output formats

### Text output

Text output is the default and is intended for interactive CLI use. It is the
fastest way to read a single run when you mainly want a summary.

### JSON output

Use `--json` to produce a machine-readable output for automation, CI, archived
comparisons, or more careful post-run inspection.

Typical top-level JSON fields include:

- `schema_version`
- `timestamp`
- `tool`
- `run_id`
- `context`
- `model`
- `target`
- `backends`
- `results`
- Optional result-level `results[*].advice`

Schema `1.1.0` uses `results[*].advice` for result-level advice. The earlier
`results[*].advices` spelling was emitted by code but was not part of the
validated schema contract.

A simplified example:

```json
{
  "schema_version": "...",
  "timestamp": "2026-01-01T00:00:00Z",
  "tool": {"name": "mlia", "version": "..."},
  "run_id": "...",
  "context": {"host": "...", "environment": "..."},
  "model": {"path": "model.tflite"},
  "target": {"profile": "<target-profile>"},
  "backends": [{"name": "<backend>"}],
  "results": [
    {
      "kind": "performance",
      "status": "ok",
      "producer": "<backend>",
      "advice": [
        {
          "id": "0",
          "category": "performance",
          "severity": "info",
          "message": "Review the performance metrics."
        }
      ]
    }
  ]
}
```

The exact contents of the output depend on the installed plugins, but the
high-level structure stays stable.

### Metric availability

From schema version `1.1.0`, result metrics can be represented in two ways:

- numeric metrics with `name`, `value`, and `unit`
- unavailable metric entries with `name`, `unit`, `availability`, and `reason`

For numeric metrics, omitted `availability` means that the value is available.
Unavailable metric entries do not contain a placeholder `value`.

This explicit availability marker is currently limited to the standardized
performance fields added for this work: `accelerator_operator_percentage`,
`inferences_per_second`, `cpu_utilization`, `target_utilization`,
`inference_time`, `model_weight_memory`, `peak_activation_memory`, and
`average_memory`. It is not a complete availability map for every possible
consumer field.

Example metric entries:

```json
[
  {
    "name": "inferences_per_second",
    "value": 4830.9,
    "unit": "inferences/s"
  },
  {
    "name": "cpu_utilization",
    "unit": "%",
    "availability": "unavailable",
    "reason": "CPU utilization data is not available."
  }
]
```

## How to read a result

A recommended reading order is:

1. Look at the run-level context first.
2. Identify which target profile and backend produced the result.
3. Read the top-level metrics before drilling into operator or layer detail.
4. Move to the owning plugin package when you need the exact interpretation of a
   backend-specific field.

That order helps keep the core schema and the plugin-specific semantics clearly
separated.

## Using metrics well

A practical approach is:

1. Use the core JSON or text structure to orient yourself.
2. Identify which plugin-owned backend produced the result you care about.
3. Interpret the dominant metric first, then move to deeper detail only if you
   need it.
4. Use the troubleshooting pages related to the owning plugin if the output
   appears incorrect or incomplete.

## Cross-links

- See [Backends](backends.md) for backend ownership and discovery.
- See [CLI](cli.md) for command-line examples using `--json` and backend
  selection.
- See the split packages for detailed metric glossaries, examples, and
  troubleshooting.
