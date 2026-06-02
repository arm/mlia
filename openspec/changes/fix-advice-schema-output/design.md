## Context

MLIA has two advice representations involved in standardized output:

- `mlia.core.advice_generation.Advice`, used by advice producers and events.
- `mlia.core.output_schema.Advice`, used by schema-shaped output.

Both already contain the fields needed for the current advice payload: `id`, `category`, `severity`, `message`, optional `affected_entities`, and optional `details`. The mismatch is in the public wire format. Current serialization emits upper-case enum values and places advice under `results[*].advices`, while schema `1.1.0` does not declare that result property.

This change updates the `1.1.0` standardized output contract. It does not patch schema `1.0.0`, and it does not define the long-term advice authoring model.

## Goals / Non-Goals

**Goals:**

- Make result-level advice validate against `mlia-output-schema-1.1.0.json`.
- Emit advice under `results[*].advice`.
- Emit lower-case schema values for advice `category` and `severity`.
- Rename the result-level schema model field from `advices` to `advice`.
- Keep Python enum members and existing advice generation patterns intact.
- Update core serialization, parsing, reporting, schema validation, and documentation.
- Provide tests that catch both the accepted new field and the rejected legacy field.

**Non-Goals:**

- Do not change `mlia-output-schema-1.0.0.json`.
- Do not accept `advices` as a legacy alias in schema `1.1.0` or in core parsing.
- Do not define stable advice ID governance, centralized advice wording, source-reference metadata, or consumer action semantics.
- Do not change how advice producers decide which advice to create.
- Do not rename unrelated advice-category or capability terminology such as `supported_advice`.
- Do not decide whether advice should be attached to every result or routed to a specific result.
- Do not fix unrelated schema mismatches in this change.

## Decisions

### Use `advice` as the result property

Standardized output will use `results[*].advice` rather than `results[*].advices`.

Rationale: `advice` is already the natural English plural form and is clearer as a public JSON property. The existing `advices` field is not schema-valid, so preserving it would turn an implementation mismatch into a public contract.

Alternative considered: accept both `advice` and `advices`. This was rejected because it would preserve the invalid spelling and make downstream consumers handle two equivalent spellings.

### Align the result schema model with the wire field

The core `Result` schema model will use a result-level `advice` field rather than retaining an internal `advices` field.

Rationale: Keeping `advices` internally would preserve the old concept in the place most likely to drive future serialization. Renaming the result-level model field keeps the code aligned with the schema contract and reduces the chance of reintroducing the invalid wire key.

This decision applies to result-level advice entries only. Existing category and capability concepts such as `supported_advice` are not part of this rename.

Alternative considered: keep the internal field name as `advices` and only rename the serialized key. This was rejected because it would reduce churn but keep an avoidable source of confusion in the schema model.

### Use lower-case enum values on the wire

Advice JSON will serialize `category` and `severity` using the enum values, for example `"performance"` and `"info"`, not the enum member names, for example `"PERFORMANCE"` and `"INFO"`.

Rationale: The rest of the standardized output schema uses lower-case wire values for similar fields such as `kind`, `status`, and `scope`. Python code can keep upper-case enum members while JSON uses schema-style values.

Alternative considered: keep upper-case values for compatibility with current emitted output. This was rejected because that output is already part of an invalid, unfinished schema path and would make advice inconsistent with the rest of the schema.

### Keep advice IDs opaque

The schema will require an `id` string but will not constrain the string format.

Rationale: The current code can provide an ID, and existing design discussion expects advice to have IDs. A full stable ID taxonomy is a larger advice-design problem and does not need to block schema validation.

Alternative considered: define stable ID prefixes or a global registry now. This was rejected as out of scope for a schema mismatch fix.

### Reuse existing operator identifiers

Advice `affected_entities` will reuse the existing `operator_identifier` definition.

Rationale: This gives advice a schema-valid way to identify affected operators or operator chains without introducing a second location model.

Alternative considered: add advice-specific affected-entity fields. This was rejected because the existing identifier shape already covers `scope`, `name`, `location`, and optional `id`.

## Risks / Trade-offs

- **Risk:** Consumers relying on the current invalid `advices` output will need to update. **Mitigation:** This change is scoped to schema `1.1.0`, updates tests and documentation, and deliberately rejects the legacy field so the contract is clear.
- **Risk:** A broad rename could accidentally touch advice category or capability APIs. **Mitigation:** Limit the rename to result-level advice entries and keep concepts such as `supported_advice` unchanged.
- **Risk:** Required `id` may be mistaken for a globally stable taxonomy. **Mitigation:** Document that `id` is required but opaque for this change, and leave governance for follow-up advice-design work.
- **Risk:** Advice is still attached by the existing reporting path rather than routed to specific results. **Mitigation:** Preserve current routing and leave result-specific association as a separate design question.
- **Risk:** Plugin-specific output writers may still emit the legacy field until their repositories are updated. **Mitigation:** Land the core contract first, then update plugin-owned paths and dependency floors in follow-up branches.
