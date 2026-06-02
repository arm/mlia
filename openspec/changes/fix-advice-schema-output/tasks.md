## 1. Schema Contract

- [x] 1.1 Add `results[*].advice` to `mlia-output-schema-1.1.0.json`.
- [x] 1.2 Add a reusable advice entry schema definition with required `id`, `category`, `severity`, and `message` fields.
- [x] 1.3 Allow optional `affected_entities` using the existing operator identifier definition.
- [x] 1.4 Allow optional unconstrained `details` objects on advice entries.
- [x] 1.5 Confirm `mlia-output-schema-1.0.0.json` remains unchanged.

## 2. Core Serialization

- [x] 2.1 Update core advice serialization to emit lower-case `category` and `severity` values.
- [x] 2.2 Rename the result-level schema model field from `advices` to `advice`.
- [x] 2.3 Update result serialization to emit `advice` instead of `advices`.
- [x] 2.4 Update result parsing to read `advice` without accepting `advices` as a legacy alias.
- [x] 2.5 Update `JSONReporter` advice merging to append advice under `results[*].advice`.
- [x] 2.6 Remove or update any remaining core references to the legacy advice output field.
- [x] 2.7 Keep unrelated advice category and capability terminology, such as `supported_advice`, unchanged.

## 3. Tests

- [x] 3.1 Add core unit coverage for result serialization and parsing with `advice`.
- [x] 3.2 Add core unit coverage for lower-case advice `category` and `severity` serialization.
- [x] 3.3 Update reporter tests to expect `results[*].advice`.
- [x] 3.4 Update helper and test names that refer specifically to result advice entries.
- [x] 3.5 Add JSON Schema validation coverage for a valid `1.1.0` payload containing result-level advice.
- [x] 3.6 Add negative JSON Schema validation coverage proving `results[*].advices` is rejected.
- [x] 3.7 Add validation or diff coverage proving schema `1.0.0` was not changed.

## 4. Documentation And Follow-Up

- [x] 4.1 Update standardized output documentation from `extensions.advice` to `results[*].advice`.
- [x] 4.2 Record downstream follow-up needs for plugin-owned writers or tests that still expect `advices`.
- [x] 4.3 Run focused tests for changed core schema, reporting, and validation paths.
- [x] 4.4 Run OpenSpec validation for `fix-advice-schema-output`.
