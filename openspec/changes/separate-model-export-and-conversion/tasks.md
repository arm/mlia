## 1. Shared transformation surface

- [x] 1.1 Add `TransformRequest` for path-based and object-based model
  transformations
- [x] 1.2 Add `transform_model()` with pass-through behavior when conversion is
  not needed
- [x] 1.3 Select a transformer through
  `supports(model, target_format, transform_options)`
- [x] 1.4 Add object-export dispatch through the shared transformer registry
- [x] 1.5 Pass `TransformRequest.transform_options` through `supports(...)` and
  the selected transformer call

## 2. Registry and plugin consolidation

- [x] 2.1 Remove the dedicated `ConverterRegistry` implementation
- [x] 2.2 Add `Registry.get()` and use `Registry[Transformer]` for shared
  transformer lookup
- [x] 2.3 Add shared transformer plugin loading

## 3. API integration and validation

- [x] 3.1 Route `torch.nn.Module` to `.pt2` export through
  `TransformRequest` in `src/mlia/api.py`
- [x] 3.2 Move registry coverage from `tests/test_converter_registry.py` to
  `tests/test_transformers_registry.py`
- [x] 3.3 Add tests for model-format detection, converter/exporter lookup
  failures, and transform-options-aware transformer selection
- [x] 3.4 Add tests for direct path conversion, object export dispatch, and API
  request construction
- [x] 3.5 Carry `example_inputs` and `enable_quantization` on
  `TransformRequest.transform_options` while keeping the backend-options
  compatibility override
