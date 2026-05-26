# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Validation utilities for standardized output schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlia.core.output_schema as schema

try:
    import jsonschema

    JSONSCHEMA_AVAILABLE = True  # pragma: no cover
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""


def load_schema() -> dict[str, Any]:
    """Load the standardized output JSON schema."""
    schema_path = (
        Path(__file__).parent.parent
        / "resources"
        / f"mlia-output-schema-{schema.SCHEMA_VERSION}.json"
    )
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, encoding="utf-8") as schema_file:
        return json.load(schema_file)  # type: ignore[no-any-return]


def load_target_schema() -> dict[str, Any]:
    """Load the standardized target JSON schema."""
    schema_path = (
        Path(__file__).parent.parent
        / "resources"
        / f"mlia-target-schema-{schema.TARGET_SCHEMA_VERSION}.json"
    )
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, encoding="utf-8") as schema_file:
        return json.load(schema_file)  # type: ignore[no-any-return]


def _build_schema_registry(output_schema: dict[str, Any]) -> Any:
    """Build a local registry for resolving schema references."""
    if not JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "jsonschema library is required for schema validation. "
            "Install it with: pip install jsonschema"
        )

    from referencing import (
        Registry,  # pylint: disable=import-outside-toplevel
        Resource,  # pylint: disable=import-outside-toplevel
    )
    from referencing.jsonschema import (  # pylint: disable=import-outside-toplevel
        DRAFT202012,
    )

    target_schema = load_target_schema()
    registry: Registry = Registry().with_resources(
        [
            (
                output_schema.get("$id", ""),
                Resource.from_contents(output_schema, DRAFT202012),
            ),
            (
                target_schema.get("$id", ""),
                Resource.from_contents(target_schema, DRAFT202012),
            ),
        ]
    )
    return registry


def validate_with_jsonschema(
    data: dict[str, Any], output_schema: dict[str, Any]
) -> None:
    """Validate data against schema using jsonschema library.

    Args:
        data: Data to validate
        output_schema: JSON schema

    Raises:
        SchemaValidationError: If validation fails
        ImportError: If jsonschema is not available
    """
    if not JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "jsonschema library is required for schema validation. "
            "Install it with: pip install jsonschema"
        )

    try:
        errors = _collect_jsonschema_errors(data, output_schema)
    except jsonschema.exceptions.ValidationError as err:
        raise SchemaValidationError(f"Schema validation failed: {err.message}") from err
    if errors:
        raise SchemaValidationError(f"Schema validation failed: {errors[0]}")


def _collect_jsonschema_errors(
    data: dict[str, Any], output_schema: dict[str, Any]
) -> list[str]:
    """Collect jsonschema validation errors."""
    registry = _build_schema_registry(output_schema)
    validator = jsonschema.Draft202012Validator(output_schema, registry=registry)
    return [err.message for err in validator.iter_errors(data)]


def collect_validation_errors(
    data: dict[str, Any], use_jsonschema: bool = True
) -> list[str]:
    """Collect validation errors without raising.

    Args:
        data: Data to validate
        use_jsonschema: Whether to use jsonschema library for validation

    Returns:
        List of validation error messages.
    """
    errors = validate_basic_structure(data)
    if not use_jsonschema:
        return errors
    if not JSONSCHEMA_AVAILABLE:
        import warnings  # pylint: disable=import-outside-toplevel

        warnings.warn(
            "jsonschema library not available. Using basic validation only. "
            "For full validation, install with: pip install jsonschema",
            UserWarning,
        )
        return errors
    validation_error = getattr(
        getattr(globals().get("jsonschema"), "exceptions", None),
        "ValidationError",
        None,
    )
    try:
        output_schema = load_schema()
        errors.extend(_collect_jsonschema_errors(data, output_schema))
    except Exception as err:
        if isinstance(validation_error, type) and isinstance(err, validation_error):
            errors.append(getattr(err, "message", str(err)))
        else:
            errors.append(f"Schema validation could not be completed: {err}")
    return errors


def validate_version_format(version: str) -> bool:
    """Validate version string format (semver: X.Y.Z).

    Args:
        version: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[0-9]+\.[0-9]+\.[0-9]+$"
    return bool(re.match(pattern, version))


def validate_timestamp_format(timestamp: str) -> bool:
    """Validate ISO 8601 timestamp format.

    Args:
        timestamp: Timestamp string to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?$"
    return bool(re.match(pattern, timestamp))


def validate_uuid_format(uuid_str: str) -> bool:
    """Validate UUID format.

    Args:
        uuid_str: UUID string to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(pattern, uuid_str.lower()))


def validate_sha256_format(hash_str: str) -> bool:
    """Validate SHA-256 hash format.

    Args:
        hash_str: Hash string to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r"^[A-Fa-f0-9]{64}$"
    return bool(re.match(pattern, hash_str))


def _validate_top_level_fields(data: dict[str, Any], errors: list[str]) -> None:
    """Validate top-level required fields."""
    required_fields = [
        "schema_version",
        "run_id",
        "timestamp",
        "tool",
        "target",
        "model",
        "context",
        "backends",
        "results",
    ]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")


def _validate_schema_version(data: dict[str, Any], errors: list[str]) -> None:
    """Validate schema_version field."""
    if "schema_version" in data:
        if not isinstance(data["schema_version"], str):
            errors.append("Field 'schema_version' must be a string")
        elif not validate_version_format(data["schema_version"]):
            errors.append(
                "Field 'schema_version' must be in semver format (e.g., '1.0.0')"
            )


def _validate_run_id(data: dict[str, Any], errors: list[str]) -> None:
    """Validate run_id field."""
    if "run_id" in data:
        if not isinstance(data["run_id"], str):
            errors.append("Field 'run_id' must be a string")
        elif not validate_uuid_format(data["run_id"]):
            errors.append("Field 'run_id' must be a valid UUID")


def _validate_timestamp(data: dict[str, Any], errors: list[str]) -> None:
    """Validate timestamp field."""
    if "timestamp" in data:
        if not isinstance(data["timestamp"], str):
            errors.append("Field 'timestamp' must be a string")
        elif not validate_timestamp_format(data["timestamp"]):
            errors.append("Field 'timestamp' must be in ISO 8601 format")


def _validate_tool(data: dict[str, Any], errors: list[str]) -> None:
    """Validate tool field."""
    if "tool" in data:
        if not isinstance(data["tool"], dict):
            errors.append("Field 'tool' must be an object")
        else:
            tool = data["tool"]
            for field in ["name", "version"]:
                if field not in tool:
                    errors.append(f"Missing required field: tool.{field}")
                elif not isinstance(tool[field], str):
                    errors.append(f"Field 'tool.{field}' must be a string")


def _validate_target(data: dict[str, Any], errors: list[str]) -> None:
    """Validate target field."""
    if "target" in data:
        if not isinstance(data["target"], dict):
            errors.append("Field 'target' must be an object")
        else:
            target = data["target"]
            for field in ["profile_name", "target_type", "components", "configuration"]:
                if field not in target:
                    errors.append(f"Missing required field: target.{field}")
            if "components" in target and not isinstance(target["components"], list):
                errors.append("Field 'target.components' must be an array")
            elif "components" in target and len(target["components"]) == 0:
                errors.append("Field 'target.components' must have at least one item")


def _validate_model(data: dict[str, Any], errors: list[str]) -> None:
    """Validate model field."""
    if "model" in data:
        if not isinstance(data["model"], dict):
            errors.append("Field 'model' must be an object")
        else:
            model = data["model"]
            for field in ["name", "format", "hash"]:
                if field not in model:
                    errors.append(f"Missing required field: model.{field}")
                elif not isinstance(model[field], str):
                    errors.append(f"Field 'model.{field}' must be a string")
            if (
                "hash" in model
                and isinstance(model["hash"], str)
                and not validate_sha256_format(model["hash"])
            ):
                errors.append("Field 'model.hash' must be a valid SHA-256 hash")


def _validate_backends(data: dict[str, Any], errors: list[str]) -> None:
    """Validate backends field."""
    if "backends" in data:
        if not isinstance(data["backends"], list):
            errors.append("Field 'backends' must be an array")
        elif len(data["backends"]) == 0:
            errors.append("Field 'backends' must have at least one item")


def _validate_results(data: dict[str, Any], errors: list[str]) -> None:
    """Validate results field."""
    if "results" in data:
        if not isinstance(data["results"], list):
            errors.append("Field 'results' must be an array")


def validate_basic_structure(data: dict[str, Any]) -> list[str]:
    """Perform basic validation without jsonschema library.

    Args:
        data: Data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    _validate_top_level_fields(data, errors)
    _validate_schema_version(data, errors)
    _validate_run_id(data, errors)
    _validate_timestamp(data, errors)
    _validate_tool(data, errors)
    _validate_target(data, errors)
    _validate_model(data, errors)
    _validate_backends(data, errors)
    _validate_results(data, errors)

    return errors


def validate_standardized_output(
    data: dict[str, Any], use_jsonschema: bool = True
) -> None:
    """Validate standardized output data.

    Args:
        data: Data to validate
        use_jsonschema: Whether to use jsonschema library for validation

    Raises:
        SchemaValidationError: If validation fails
    """
    # Always perform basic validation
    basic_errors = validate_basic_structure(data)
    if basic_errors:
        raise SchemaValidationError(
            "Basic validation failed:\n  - " + "\n  - ".join(basic_errors)
        )

    # Perform full schema validation if requested and library is available
    if use_jsonschema:
        if not JSONSCHEMA_AVAILABLE:
            import warnings  # pylint: disable=import-outside-toplevel

            warnings.warn(
                "jsonschema library not available. Using basic validation only. "
                "For full validation, install with: pip install jsonschema",
                UserWarning,
            )
        else:
            output_schema = load_schema()
            validate_with_jsonschema(data, output_schema)


def validate_output_file(filepath: Path | str, use_jsonschema: bool = True) -> None:
    """Validate a standardized output JSON file.

    Args:
        filepath: Path to JSON file
        use_jsonschema: Whether to use jsonschema library for validation

    Raises:
        SchemaValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(path, encoding="utf-8") as json_file:
        data = json.load(json_file)

    validate_standardized_output(data, use_jsonschema=use_jsonschema)
