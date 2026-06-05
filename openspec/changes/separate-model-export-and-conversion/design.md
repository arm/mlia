## Context

MLIA already had converter plugins, but the converter registry lived in a
bespoke type that overlapped with the generic `Registry` utility. There was
also no shared request type or helper for model transformations, so callers had
to combine format detection, transformer lookup, and optional argument handling
themselves.

The current implementation now introduces a shared transformation surface that
handles both direct file-to-file conversion and object export dispatch. It
still does not introduce a preparation manager, multi-step route planning, or
a dedicated transformer-options gatherer.

## Goals / Non-Goals

**Goals:**
- Provide one shared interface for direct file-to-file conversions and
  object-based exports.
- Consolidate plugin loading behind one transformer entry-point group.
- Keep optional transformer and exporter options on one request object instead
  of inventing per-caller plumbing.
- Remove the dedicated converter registry in favor of the generic `Registry`.

**Non-Goals:**
- Add a preparation manager or multi-step route planning.
- Add new CLI flags.
- Introduce a transformer-specific config object separate from
  `TransformRequest`.
- Move `enable_quantization` out of the existing backend-config path.

## Decisions

### 1. Introduce a shared transformation interface

The implementation uses a small dataclass, `TransformRequest`, plus a
`transform_model()` helper in `src/mlia/transformers/registry.py`. The request
carries the input model object or model path, output directory, optional target
format, and optional kwargs to forward to the selected transformer.

`transform_model()` returns the original path unchanged when:
- `target_format` is not provided
- the requested format matches the detected input format

When the input is a `Path`, it resolves and runs a direct converter. When the
input is a model object, it resolves and runs a matching exporter.

Rationale:
- This creates a reusable transformation surface for both conversion and
  export.
- Pass-through behavior keeps the helper safe for future higher-level
  orchestration.

### 2. Consolidate plugin loading behind one transformer entry-point group

Transformers now load through `load_transformer_plugins()` and the
`mlia.plugin.transformer` entry-point group. The registry does not distinguish
between converters and exporters; both register into the same
`Registry[Transformer]`.

Rationale:
- One plugin surface is simpler than separate converter/exporter registries.
- Selection logic stays centralized in `src/mlia/transformers/registry.py`.

### 3. Select transformers with `supports(model, target_format, transform_options)`

Transformer selection iterates over registered transformers and picks the first
one whose `supports(model, target_format, transform_options)` method returns
`True`.
`target_format` is always part of the capability check, for both path-based and
object-based inputs.

Rationale:
- This keeps routing policy local to the transformer instead of encoding it in
  registry naming rules.
- It lets exporters and converters use the same contract.

### 4. Support only direct one-step conversions in this step

`transform_model()` resolves a single transformer for the requested model and
target format. It does not attempt multi-step routing or path search.

Rationale:
- Direct transformer lookup matches the current plugin model.
- Multi-step planning belongs to a later preparation-manager change.

### 5. Pass request transform options directly; let transformers decide support

`TransformRequest.transform_options` is passed into both `supports(...)` and
the selected transformer call.

Rationale:
- Capability checks now happen through the transformer's own `supports(...)`
  logic.
- The registry stays simple and avoids duplicating per-transformer argument
  knowledge.

### 6. Replace the bespoke converter registry with the shared generic registry

The old `ConverterRegistry` class is removed. The generic `Registry` gains a
`get()` method so it can support the same lookup use case while keeping
registration behavior shared across components.

Rationale:
- The dedicated converter registry duplicated behavior already present in
  `Registry`.
- One registry implementation reduces maintenance and keeps plugin-loading code
  consistent.

### 7. Route API-side `torch.nn.Module` export through the transformation layer

`src/mlia/api.py` now builds a `TransformRequest` with `target_format="pt2"`
and `transform_options={"example_inputs": ..., "enable_quantization": ...}`
when exporting a `torch.nn.Module` input. The API still controls target
validation and backend-level quantization behavior. In particular,
`enable_quantization=False` still becomes a backend option override for
downstream compatibility.

Rationale:
- This removes bespoke API-side export plumbing.
- It makes the current `.pt2` path use the same transformation surface as
  future conversions and exports, without changing backend-option ownership in
  the same step.

## Migration Plan

1. Move converter lookup and invocation into `src/mlia/transformers/registry.py`.
2. Extend `Registry` with `get()` and remove the dedicated `ConverterRegistry`.
3. Add shared transformer plugin loading and the
   `supports(model, target_format, transform_options)` contract.
4. Route API-side module export through `TransformRequest` and
   `transform_model()`.
5. Port registry tests to the new module and expand them around format
   detection, transformer lookup, transform-options handling, and dispatch.
6. Leave multi-step routing, preparation management, and further CLI work to
   follow-up changes.
