## Why

MLIA already supported file-to-file converter plugins, but that capability was
split across a dedicated converter registry and ad hoc caller-side behavior.
That made it harder to treat conversion and export as one shared
model-transformation API.

The current implementation now exposes a shared transformation surface that
handles both direct path-based conversions and object exports, and routes the
current `torch.nn.Module` to `.pt2` API flow through that interface.

## What Changes

- Introduce `src/mlia/transformers/registry.py` as the shared transformation
  entry point for both direct converters and exporters.
- Add `TransformRequest` to carry a model object or model path, output
  directory, optional target format, and optional `transform_options`.
- Add `transform_model()` to:
  - return the original path unchanged when no target format is requested
  - return the original path unchanged when the input format already matches
    the requested target format
  - resolve a transformer through the shared `mlia.plugin.transformer`
    entry-point group
  - select the first transformer whose
    `supports(model, target_format, transform_options)` method returns `True`
  - pass `transform_options` directly to the selected transformer call
- Replace the bespoke `ConverterRegistry` class with the existing generic
  `Registry` by adding `Registry.get()`.
- Move and expand tests from `tests/test_converter_registry.py` to
  `tests/test_transformers_registry.py`.
- Update `src/mlia/api.py` so the current `torch.nn.Module` to `.pt2` export
  flow constructs a `TransformRequest` and calls `transform_model()`.
- Move both `example_inputs` and `enable_quantization` into
  `TransformRequest.transform_options`. API-side quantization disabling still
  becomes a backend option override for downstream compatibility.

## Capabilities

### New Capabilities
- `model-transformation`: Execute direct file-to-file conversions and object
  exports through `TransformRequest` and `transform_model()`.
- `transformer-dispatch`: Resolve path-based and object-based transformations
  through `supports(model, target_format, transform_options)` capability
  checks.
- `transform-request-options`: Carry optional transformer/exporter options on
  `TransformRequest` and forward them to the selected transformer.

### Modified Capabilities
- `converter-registry`: Use the shared generic `Registry` instead of a
  dedicated `ConverterRegistry` implementation.
- `api-module-export`: Route the existing API-side `torch.nn.Module` export
  path through the shared transformation surface.

## Impact

- Adds `src/mlia/transformers/registry.py` and `src/mlia/transformers/__init__.py`.
- Removes `src/mlia/plugins/converter_registry.py`.
- Extends `src/mlia/utils/registry.py` with `Registry.get()`.
- Adds shared transformer plugin loading in `src/mlia/plugins/plugins.py`.
- Updates `src/mlia/api.py` to use the shared transformation surface for
  `.pt2` export requests while keeping backend-config quantization overrides.
- Adds focused tests for format detection, transformer lookup, direct dispatch,
  request kwargs handling, and API integration.
- Defers multi-step routing, preparation management, and further CLI expansion
  to follow-up work.
