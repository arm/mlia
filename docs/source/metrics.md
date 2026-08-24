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

## Entity breakdown projection

Some backends report metrics for scheduled entities while the same work is also
represented by source operators, code-stack frames, modules, segments, or other
views. MLIA can project missing breakdowns across the normalized entity DAG.
Projection is conservative and never replaces an existing target breakdown.

### Authoritative origins

Each surviving authoritative breakdown is identified by the entity ID that owns
the original metric. This ID is its **accounting origin**. Copies preserve the
origin, while arithmetic unions the distinct origins that contributed to the
result.

Graph overlap is deliberately not an accounting collision. Separate backend
entities can represent different scheduled contributions even when they both
relate to the same source operator. For example, five chain entities associated
with one resize operation remain five distinct origins and their explicit `sum`
metrics are additive.

This replaces structural-leaf accounting. Terminal descendants are not used as
accounting footprints or global equivalence keys. The graph is still validated
and used to determine represented branches, target coverage, hierarchy
competition, and contested extras.

### Declared authoritative frontiers

An `EntityKind` declaration can identify a parent-to-child relationship such as
`chain -> source_operator`. All direct authoritative parents across that
relationship form the target's **nearest authoritative frontier**.

Every parent in that frontier must be compatible with the target. Multiple
parents are separate origins and can be combined only when corresponding metrics
have compatible explicit `sum` aggregation. Equal values from distinct parents
are still distinct contributions.

A mixed parent that includes a contested unrelated branch blocks the complete
frontier, even when another parent is an exact match. This prevents a partial
value being presented as the target's complete cost. An authoritative aggregate
ancestor, such as a cascade above an authoritative chain, is shadowed by the
nearer authoritative entity and is not counted again.

A successful frontier is sealed with its complete origin set. Equivalent and
aggregate downstream code-stack or module entities use that resolved record
instead of reconsidering the individual chain records. A conflicted narrow
frontier blocks partial projection, but it does not poison every overlapping
view. Another target may re-evaluate the complete origins when it covers the
blocked scope and its derivation accounts for every required origin. Normal
contested-extra rules are then applied at that target. This permits a stack to
copy a chain covering the same source operators plus an uncontested generated
operation, without assigning figures to individual source operators.

Every actual parent-to-child entity edge must be covered by either a result-local
`EntityKind` relationship or a schema-defined well-known relationship. Core
validation normalizes one-sided parent and child references before checking the
corresponding kind pair. This prevents an undeclared graph edge from silently
avoiding authoritative-frontier semantics.

### Graph coverage and extras

A target is covered through explicit graph branches. A contributor must intersect
at least one represented target branch. A contributor branch outside the target
is an **extra**.

An extra is contested when another entity competing with the target intersects
that branch. Entities compete when they have the same kind or belong to the same
hierarchy: ancestor and descendant entities compete, as do entities sharing an
ancestor. Different-kind entities in unrelated hierarchies remain independent.

An uncontested extra may be copied with its origin. For example, a chain covering
an operator and generated work may project to an independent operator view. If a
same-kind entity, sibling, or ancestor claims the generated branch, projection is
suppressed.

### Copying and arithmetic

One complete compatible contributor is a view change and is copied unchanged. It
may use any aggregation policy, including an unsupported or absent policy.

Combining multiple origins is arithmetic. Corresponding metrics must:

- have matching names, units, qualifiers, and breakdown qualifiers;
- be numeric and available;
- use explicit `sum` aggregation; and
- have compatible sample metadata.

A record carries the set of original authoritative entity IDs represented by its
figures. The same origin can never be counted twice. Distinct origins can be
summed even when their graph coverage overlaps, because the backend entities—not
the shared graph nodes—identify the measured contributions.

### Authority, conflicts, and priority

Projection uses this priority order:

1. an existing breakdown on the target;
2. a complete declared direct-parent frontier;
3. one complete no-extra contributor;
4. a complete no-extra calculation from distinct origins;
5. one complete contributor with uncontested extras; and
6. a complete residual calculation with uncontested extras.

Sealed frontier records outrank their individual constituent origins. Otherwise,
a one-record copy with fewer origins is preferred over a transitive aggregate,
which prevents inferred parent views feeding larger aggregates back into their
own children.

At the same priority, identical canonical figures deduplicate. Conflicting figure
sets make the target ambiguous and suppress projection.

### Worked decisions

| Scenario | Decision | Rationale |
| --- | --- | --- |
| Two resize chains both map to one source operator and report 2 and 3 with `sum` | Source receives 5 | The chain IDs are distinct authoritative origins; shared source coverage does not make them duplicates. |
| A convolution origin is 10 and a resolved resize frontier contains origins worth 2 and 3 | Parent stack and module receive 15 | Downstream aggregation unions the three sealed origins and does not reconsider the raw resize chains. |
| Two generic origins cover `{a, b}` and `{b, c}` with values 5 and 7 using `sum` | Parent covering `{a, b, c}` receives 12 | The separately owned authoritative measurements are additive despite overlapping graph coverage at `b`. |
| An origin worth 2 covers `{a, generated}` and is copied to `{a}`; another origin worth 3 covers `{b, generated}` | Parent `{a, b}` receives 5 when `generated` is uncontested | Copies retain the first origin ID. The common generated branch is not itself an accounting origin and therefore does not create a false collision. |
| Exact chain `{op}` and mixed chain `{op, other}` directly parent source `{op}`, where `other` is contested | No projection | Every declared frontier parent must be compatible; the exact chain alone would be a partial value. |
| Chain `{a, b}` and stack `{a, b}` cover the same sources | Stack copies the chain figures; neither source receives figures | The stack contains and accounts for the complete chain origin, so no disaggregation is required. |
| Two chains `{a, b}` and `{c, d}` have additive figures and a parent view covers `{a, b, c, d}` | Parent receives their sum | The parent contains both complete origins and accounts for every narrower frontier obligation. |
| Cascade worth 10 contains an authoritative chain worth 4 that directly parents the source | Source receives 4 | The nearer authoritative chain shadows its aggregate ancestor. |
| One complete `max` contributor and a separate two-origin `sum` calculation cover the same target | Copy the complete `max` contributor | A one-record view change outranks arithmetic. |
