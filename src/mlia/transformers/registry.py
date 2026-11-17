# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model transformer registry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mlia.core.errors import ConfigurationError
from mlia.plugins.plugins import load_transformer_plugins
from mlia.transformers.error import TransformerNotFoundError
from mlia.utils.registry import Registry


class Transformer(Protocol):
    """Capability-based transformer contract for paths and model objects."""

    def supports(
        self,
        model: object,
        target_format: str,
        transform_options: dict[str, Any] | None = None,
    ) -> bool:
        """Return whether this transformer can handle the requested conversion."""
        ...

    def __call__(
        self,
        model: object,
        output_dir: Path,
        **kwargs: Any,
    ) -> Path:
        """Transform the model into ``output_dir`` and return the resulting path."""
        ...


transformer_registry: Registry[Transformer] = Registry(
    loader=lambda: load_transformer_plugins(transformer_registry)
)


def ensure_transformer_plugins_loaded() -> None:
    """Load transformer plugins into the shared transformer registry."""
    transformer_registry.ensure_loaded()


@dataclass
class TransformRequest:
    """Parameters describing a requested model transformation."""

    model: object
    output_dir: Path
    target_format: str
    transform_options: dict[str, Any]


def transform_model(req: TransformRequest) -> Path:
    """Transform a model to the requested format when needed."""
    # Runtime dispatch depends on this registry; import here to avoid a cycle.
    from mlia.runtime.session import current_session
    from mlia.runtime.transformer import select_runtime_transformer

    if isinstance(req.model, Path):
        try:
            if _get_model_format(req.model) == req.target_format:
                return req.model
        except ValueError as err:
            raise ConfigurationError(str(err)) from err

    session = current_session()
    transformer: Transformer
    if session is not None:
        transformer = select_runtime_transformer(
            session.profile, req.model, req.target_format, req.transform_options
        )
    else:
        transformer = _get_transformer(
            transformer_registry,
            req.model,
            req.target_format,
            req.transform_options,
        )
    return transformer(req.model, req.output_dir, **(req.transform_options))


def _get_transformer(
    registry: Registry[Transformer],
    model: object,
    target_format: str,
    transform_options: dict[str, Any],
) -> Transformer:
    """Return the transformer for the requested model and target format."""
    valid_transformer: Transformer | None = None
    for transformer in registry.items.values():
        if transformer.supports(model, target_format, transform_options):
            valid_transformer = transformer
            break

    if valid_transformer is None:
        raise TransformerNotFoundError("Transformer for model is not available.")

    return valid_transformer


def _get_model_format(model: Path) -> str:
    suffix = Path(model).suffix.lower().lstrip(".")

    if suffix:
        return suffix

    raise ValueError(f"Unsupported model format: {model}")
