# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert model files using the converter selected by an execution profile.

For example, converting a .pt2 model to TOSA sends the input path and conversion
options to a worker process. Its output model is copied to the caller's requested
directory. The converter plugin decides which input and output formats it supports.
"""

from pathlib import Path
from typing import Any

from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import ExecutionProfile
from mlia.runtime.session import invoke
from mlia.transformers.registry import Transformer
from mlia.utils.boundaries.filesystem import (
    copy_user_output_file,
    ensure_user_output_directory,
)


def select_runtime_transformer(
    profile: ExecutionProfile,
    model: object,
    target_format: str,
    transform_options: dict[str, Any] | None = None,
) -> "RuntimeTransformer":
    """Find one selected converter supporting the requested format and options."""
    from mlia.transformers.registry import transformer_registry

    bindings = {
        operation: (name, definition)
        for name, (operation, definition) in profile.stages.items()
    }
    matches = []
    for converter in transformer_registry.items.values():
        if not isinstance(converter, RuntimeTransformer):
            continue
        if converter.operation not in bindings:
            continue
        name, definition = bindings[converter.operation]
        stage = profile.data["stages"][name]
        options = {
            **stage.get("options", {}).get("options", {}),
            **(transform_options or {}),
        }
        if not converter.supports(model, target_format, options):
            continue
        schema = definition.data["operations"][converter.operation]["result_schema"]
        if schema != "model/1":
            raise ConfigurationError(
                f"Selected runtime options are incompatible: {converter.operation} "
                f"returns {schema}; expected model/1"
            )
        matches.append(converter)
    if not matches:
        source = (
            Path(model).suffix
            if isinstance(model, (str, Path))
            else type(model).__name__
        )
        raise ConfigurationError(
            f"Execution profile {profile.data['id']} has no registered converter "
            f"from {source} to {target_format} supporting the requested options. "
            "Select a compatible converter in the profile and install its plugin."
        )
    if len(matches) != 1:
        raise ConfigurationError(
            f"Execution profile {profile.data['id']} selects multiple converters "
            f"to {target_format}; select only one for this conversion."
        )
    return matches[0]


class RuntimeTransformer:
    """Convert a model file to another representation in a separate environment."""

    def __init__(self, transformer: Transformer, operation: str) -> None:
        """Bind a provider capability to its runtime operation."""
        self.transformer = transformer
        self.operation = operation

    def supports(
        self,
        model: object,
        target_format: str,
        transform_options: dict[str, Any] | None = None,
    ) -> bool:
        """Delegate capability checks to the provider."""
        return self.transformer.supports(model, target_format, transform_options or {})

    def __call__(self, model: object, output_dir: Path, **kwargs: Any) -> Path:
        """Execute the conversion and return its single declared model artifact."""
        if not isinstance(model, (str, Path)):
            raise ConfigurationError(
                "Runtime conversion requires an exported model file"
            )
        result = invoke(
            self.operation, {"model": str(Path(model).resolve()), "options": kwargs}
        )
        models = [
            artifact
            for artifact in result["artifacts"]
            if artifact.get("kind") == "model"
        ]
        if len(models) != 1:
            raise ConfigurationError(
                f"Converter {self.operation} did not produce exactly one model artifact"
            )
        artifact = Path(models[0]["path"])
        destination = output_dir / artifact.name
        ensure_user_output_directory(output_dir)
        if destination.resolve() != artifact.resolve():
            copy_user_output_file(artifact, destination)
        return destination
