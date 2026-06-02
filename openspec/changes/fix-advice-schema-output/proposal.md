## Why

MLIA can currently emit advice inside standardized result objects, but schema `1.1.0` does not declare the emitted `results[*].advices` property. Because result objects reject additional properties, advice-bearing output can fail JSON Schema validation even though MLIA deliberately produced the advice.

This is the right point to fix the wire contract because schema `1.1.0` is where the current standardized output work is being formalized. The fix should make advice schema-valid without broadening the older `1.0.0` contract or preserving the awkward `advices` field name.

## What Changes

- **BREAKING**: Move serialized result-level advice from `results[*].advices` to `results[*].advice`.
- **BREAKING**: Serialize advice `category` and `severity` as lower-case schema values such as `"performance"` and `"info"`, while keeping Python enum members such as `AdviceCategory.PERFORMANCE`.
- Add an advice entry definition to schema `1.1.0` with required `id`, `category`, `severity`, and `message` fields.
- Allow optional `affected_entities` and `details` fields on advice entries.
- Reuse the existing operator identifier shape for `affected_entities`.
- Treat advice `id` as a required opaque string for this change; defining a long-term advice ID registry, central wording catalog, source references, or consumer action semantics is out of scope.
- Do not add a compatibility alias for `advices` in the `1.1.0` schema or parser.
- Leave `mlia-output-schema-1.0.0.json` unchanged.
- Update tests and documentation to describe `results[*].advice` as the standardized result-level advice location.

## Capabilities

### New Capabilities

- `standardized-result-advice`: Defines schema-valid result-level advice entries in MLIA standardized JSON output.

### Modified Capabilities

- None.

## Impact

- `src/mlia/resources/mlia-output-schema-1.1.0.json`
- Core advice and result serialization/parsing in `mlia.core.output_schema`.
- Advice JSON formatting in `mlia.core.advice_generation`.
- Advice merging in `mlia.core.reporting.JSONReporter`.
- JSON Schema validation tests and core reporter/schema unit tests.
- Standardized output documentation that currently refers to `extensions.advice`.
- Follow-up plugin repository changes will be needed where plugin-specific output writers or tests still expect `advices`.
