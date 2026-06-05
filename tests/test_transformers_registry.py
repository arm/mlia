# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for transformer registry utilities."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, create_autospec

import pytest

from mlia.core.errors import ConfigurationError
from mlia.transformers.registry import (
    TransformRequest,
    _get_model_format,
    _get_transformer,
    ensure_transformer_plugins_loaded,
    transform_model,
    transformer_registry,
)
from mlia.utils.registry import Registry


@pytest.mark.parametrize(
    ("file_suffix", "model_type"),
    [
        ("tosamlir", "tosamlir"),
        ("tosa", "tosa"),
        ("tflite", "tflite"),
        ("pt2", "pt2"),
        ("vgf", "vgf"),
    ],
)
def test_get_model_format_successful_output(file_suffix: str, model_type: str) -> None:
    model = Path(f"non/existent/path/model.{file_suffix}")

    discovered_type = _get_model_format(model)

    assert discovered_type == model_type


def test_get_model_format_raises_error() -> None:
    model = Path("non/existent/path/model")

    with pytest.raises(ValueError):
        _ = _get_model_format(model)


def test_get_model_format_lowercases_uppercase_suffix() -> None:
    model = Path("non/existent/path/model.PT2")

    discovered_type = _get_model_format(model)

    assert discovered_type == "pt2"


def test_get_transformer_returns_matching_transformer() -> None:
    supported_model = object()
    matching_transformer = MagicMock()
    matching_transformer.supports.side_effect = lambda model, target_format, kwargs: (
        model is supported_model and target_format == "crungle" and kwargs == {}
    )
    unrelated_transformer = MagicMock()
    unrelated_transformer.supports.return_value = False

    registry = Registry[Any]()
    registry.register("supported_to_crungle", matching_transformer)
    registry.register("unused_to_fake", unrelated_transformer)

    transformer = _get_transformer(registry, supported_model, "crungle", {})

    assert transformer is matching_transformer


def test_get_transformer_passes_transform_options_to_supports() -> None:
    supported_model = object()

    matching_transformer = MagicMock()
    matching_transformer.supports.side_effect = (
        lambda model, target_format, kwargs=None: (
            model is supported_model
            and target_format == "crungle"
            and kwargs == {"optional_kwarg": 77}
        )
    )

    registry = Registry[Any]()
    registry.register("supported_to_crungle", matching_transformer)

    transformer = _get_transformer(
        registry,
        supported_model,
        "crungle",
        {"optional_kwarg": 77},
    )

    assert transformer is matching_transformer

    matching_transformer.supports.assert_called_once_with(
        supported_model,
        "crungle",
        {"optional_kwarg": 77},
    )


def test_get_transformer_ignores_transformer_for_wrong_target() -> None:
    supported_model = object()
    wrong_target_transformer = MagicMock()
    wrong_target_transformer.supports.side_effect = (
        lambda model, target_format, kwargs: (
            model is supported_model and target_format == "fake" and kwargs == {}
        )
    )
    matching_transformer = MagicMock()
    matching_transformer.supports.side_effect = lambda model, target_format, kwargs: (
        model is supported_model and target_format == "crungle" and kwargs == {}
    )

    registry = Registry[Any]()
    registry.register("supported_to_fake", wrong_target_transformer)
    registry.register("supported_to_crungle", matching_transformer)

    transformer = _get_transformer(registry, supported_model, "crungle", {})

    assert transformer is matching_transformer


def test_get_transformer_raises_when_transformer_is_unavailable() -> None:
    unsupported_transformer = MagicMock()
    unsupported_transformer.supports.return_value = False

    registry = Registry[Any]()
    registry.register("unsupported_to_fake", unsupported_transformer)
    registry.register("unsupported_to_crungle", unsupported_transformer)

    with pytest.raises(ConfigurationError, match="Transformer for model"):
        _get_transformer(registry, object(), "crungle", {})


def test_get_transformer_rejects_transformer_with_unsupported_kwargs() -> None:
    supported_model = object()
    transformer = create_autospec(
        lambda model, output_dir, *, optional_kwarg: Path("ignore/this/path")
    )
    transformer.supports = lambda model, target_format, kwargs: (
        model is supported_model
        and target_format == "crungle"
        and kwargs == {"optional_kwarg": 77}
    )

    registry = Registry[Any]()
    registry.register("supported_to_crungle", transformer)

    with pytest.raises(ConfigurationError, match="Transformer for model"):
        _get_transformer(
            registry,
            supported_model,
            "crungle",
            {"unsupported_kwarg": 77},
        )


def test_ensure_transformer_plugins_loaded_loads_transformer_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    monkeypatch.setattr(
        "mlia.transformers.registry.load_transformer_plugins",
        lambda registry: calls.append(registry),
    )

    ensure_transformer_plugins_loaded()

    assert calls == [transformer_registry]


def test_transform_model_returns_original_path_when_formats_match() -> None:
    model = Path("path/to/model.crungle")

    assert (
        transform_model(
            TransformRequest(
                model=model,
                output_dir=Path("output/path"),
                target_format="crungle",
                transform_options={},
            )
        )
        == model
    )


def test_transform_model_raises_configuration_error_for_path_without_suffix() -> None:
    model = Path("path/to/model")

    with pytest.raises(ConfigurationError, match="Unsupported model format"):
        transform_model(
            TransformRequest(
                model=model,
                output_dir=Path("output/path"),
                target_format="pt2",
                transform_options={},
            )
        )


def test_transform_model_successfully_converts_from_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = MagicMock(return_value=Path("output/path/model.crungle"))
    converter.supports = MagicMock(
        side_effect=lambda model, target_format, kwargs: (
            isinstance(model, Path)
            and model.suffix == ".blorp"
            and target_format == "crungle"
            and kwargs == {"optional_kwarg": 77}
        )
    )

    registry = Registry[Any]()
    registry.register("blorp_to_crungle", converter)

    monkeypatch.setattr("mlia.transformers.registry.transformer_registry", registry)
    request = TransformRequest(
        model=Path("path/to/model.blorp"),
        output_dir=Path("output/path/"),
        target_format="crungle",
        transform_options={"optional_kwarg": 77},
    )

    result = transform_model(request)

    assert result == Path("output/path/model.crungle")
    converter.supports.assert_called_once_with(
        Path("path/to/model.blorp"),
        "crungle",
        {"optional_kwarg": 77},
    )
    converter.assert_called_once_with(
        Path("path/to/model.blorp"),
        Path("output/path/"),
        optional_kwarg=77,
    )


def test_transform_model_successfully_exports_from_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    exporter = MagicMock(return_value=Path("output/path/model.crungle"))
    exporter.supports = MagicMock(
        side_effect=lambda value, target_format, kwargs: (
            value is model
            and target_format == "crungle"
            and kwargs == {"optional_kwarg": 77}
        )
    )

    registry = Registry[Any]()
    registry.register("fake_to_crungle", exporter)

    monkeypatch.setattr("mlia.transformers.registry.transformer_registry", registry)
    request = TransformRequest(
        model=model,
        output_dir=Path("output/path/"),
        target_format="crungle",
        transform_options={"optional_kwarg": 77},
    )

    result = transform_model(request)

    assert result == Path("output/path/model.crungle")
    exporter.supports.assert_called_once_with(
        model,
        "crungle",
        {"optional_kwarg": 77},
    )
    exporter.assert_called_once_with(
        model,
        Path("output/path/"),
        optional_kwarg=77,
    )


def test_transform_model_raises_when_requested_transform_options_are_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = create_autospec(lambda model, output_dir: Path("ignore/this/path"))
    transformer.supports = lambda model, target_format, kwargs: (
        isinstance(model, Path)
        and target_format == "crungle"
        and kwargs == {"unsupported_kwarg": 77}
    )

    registry = Registry[Any]()
    registry.register("blorp_to_crungle", transformer)

    monkeypatch.setattr("mlia.transformers.registry.transformer_registry", registry)

    with pytest.raises(TypeError, match="unsupported_kwarg"):
        transform_model(
            TransformRequest(
                model=Path("path/to/model.blorp"),
                output_dir=Path("output/path/"),
                target_format="crungle",
                transform_options={"unsupported_kwarg": 77},
            )
        )
