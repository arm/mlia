# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Validation utilities for standardized output validation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.core.output_validation import load_schema
from mlia.core.output_validation import SchemaValidationError
from mlia.core.output_validation import validate_basic_structure
from mlia.core.output_validation import validate_output_file
from mlia.core.output_validation import validate_sha256_format
from mlia.core.output_validation import validate_standardized_output
from mlia.core.output_validation import validate_timestamp_format
from mlia.core.output_validation import validate_uuid_format
from mlia.core.output_validation import validate_version_format
from mlia.core.output_validation import validate_with_jsonschema


_CORRECT_DATA = {
    "schema_version": "1.0.0",
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-12-29T10:30:00Z",
    "tool": {"name": "MLIA", "version": "1.0.0"},
    "target": {
        "profile_name": "ethos-u55-256",
        "target_type": "corstone-300",
        "components": ["ethos-u55"],
        "configuration": {},
    },
    "model": {
        "name": "model.tflite",
        "format": "tflite",
        "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "context": {},
    "backends": [{"name": "vela", "version": "3.0.0"}],
    "results": [],
}


def test_load_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test load_schema function."""

    # mock the schema_path variable within load_schema
    schema_file = tmp_path / "resources" / "mlia-output-schema-1.0.0.json"
    mock_file_path = tmp_path / schema_file

    monkeypatch.setattr("mlia.core.output_validation.Path", lambda _: mock_file_path)

    monkeypatch.setattr("mlia.core.output_schema.SCHEMA_VERSION", "1.0.0")

    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        _ = load_schema()

    schema_content = {"key": "value"}
    schema_file.parent.mkdir()
    with open(schema_file, "w", encoding="utf-8") as file:
        json.dump(schema_content, file)

    schema = load_schema()
    assert schema == {"key": "value"}


def test_validate_with_jsonschema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_with_jsonschema function."""

    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", False)
    with pytest.raises(ImportError, match="jsonschema library is required"):
        validate_with_jsonschema({}, {})

    # pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name
    # Mock jsonschema.validate
    class MockJsonSchema:
        class exceptions:
            class ValidationError(Exception):
                pass

        @staticmethod
        def validate(instance: dict, schema: dict) -> None:
            """Mock validate function."""

    # pylint: enable=missing-class-docstring,too-few-public-methods,invalid-name

    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema", MockJsonSchema(), raising=False
    )

    # Test successful validation
    validate_with_jsonschema({}, {})

    # pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name
    # Test validation error is caught and re-raised as SchemaValidationError
    class MockJsonSchemaWithError:
        class exceptions:
            class ValidationError(Exception):
                def __init__(self, message: str):
                    self.message = message
                    super().__init__(message)

        @staticmethod
        def validate(instance: dict, schema: dict) -> None:
            """Mock validate function that raises ValidationError."""
            raise MockJsonSchemaWithError.exceptions.ValidationError(
                "Test validation error"
            )

    # pylint: enable=missing-class-docstring,too-few-public-methods,invalid-name

    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema",
        MockJsonSchemaWithError(),
        raising=False,
    )

    with pytest.raises(
        SchemaValidationError, match="Schema validation failed: Test validation error"
    ):
        validate_with_jsonschema({}, {})


def test_validate_version_format() -> None:
    """Test validate_version_format function."""
    assert validate_version_format("1.0.0") is True
    assert validate_version_format("0.0.1") is True
    assert validate_version_format("123.456.789") is True

    assert validate_version_format("1.0") is False
    assert validate_version_format("1.0.0.0") is False
    assert validate_version_format("v1.0.0") is False
    assert validate_version_format("1.0.0-beta") is False
    assert validate_version_format("1.a.0") is False
    assert validate_version_format("") is False
    assert validate_version_format("abc") is False


def test_validate_timestamp_format() -> None:
    """Test validate_timestamp_format function."""
    assert validate_timestamp_format("2025-12-29T10:30:00") is True
    assert validate_timestamp_format("2025-12-29T10:30:00Z") is True
    assert validate_timestamp_format("2025-12-29T10:30:00+00:00") is True
    assert validate_timestamp_format("2025-12-29T10:30:00-05:00") is True
    assert validate_timestamp_format("2025-12-29T10:30:00.123456") is True
    assert validate_timestamp_format("2025-12-29T10:30:00.123456Z") is True
    assert validate_timestamp_format("2025-12-29T10:30:00.123+01:30") is True

    assert validate_timestamp_format("2025-12-29") is False
    assert validate_timestamp_format("2025/12/29T10:30:00") is False
    assert validate_timestamp_format("2025-12-29 10:30:00") is False
    assert validate_timestamp_format("2025-12-29T10:30") is False
    assert validate_timestamp_format("") is False
    assert validate_timestamp_format("not a timestamp") is False


def test_validate_uuid_format() -> None:
    """Test validate_uuid_format function."""
    assert validate_uuid_format("550e8400-e29b-41d4-a716-446655440000") is True
    assert validate_uuid_format("550E8400-E29B-41D4-A716-446655440000") is True
    assert validate_uuid_format("00000000-0000-0000-0000-000000000000") is True
    assert validate_uuid_format("ffffffff-ffff-ffff-ffff-ffffffffffff") is True

    assert validate_uuid_format("550e8400-e29b-41d4-a716") is False
    assert validate_uuid_format("550e8400-e29b-41d4-a716-44665544000g") is False
    assert validate_uuid_format("550e8400e29b41d4a716446655440000") is False
    assert validate_uuid_format("550e8400-e29b-41d4-a716-4466554400000") is False
    assert validate_uuid_format("") is False
    assert validate_uuid_format("not-a-uuid") is False


def test_validate_sha256_format() -> None:
    """Test validate_sha256_format function."""
    assert (
        validate_sha256_format(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        is True
    )
    assert (
        validate_sha256_format(
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        )
        is True
    )
    assert (
        validate_sha256_format(
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        is True
    )
    assert (
        validate_sha256_format(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        )
        is True
    )

    assert (
        validate_sha256_format(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85"
        )
        is False
    )  # 63 characters
    assert (
        validate_sha256_format(
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8555"
        )
        is False
    )  # 65 characters
    assert (
        validate_sha256_format(
            "g3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        is False
    )  # invalid character 'g'
    assert validate_sha256_format("") is False
    assert validate_sha256_format("not a hash") is False


def test_top_level_fields() -> None:
    """Test validate_basic_structure top level fields validation."""
    assert not validate_basic_structure(_CORRECT_DATA)

    data = _CORRECT_DATA.copy()
    data.pop("schema_version")
    assert validate_basic_structure(data) == ["Missing required field: schema_version"]


def test_schema_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_basic_structure schema version validation."""
    data = _CORRECT_DATA.copy()
    data["schema_version"] = 2  # type: ignore[assignment]
    assert validate_basic_structure(data) == ["Field 'schema_version' must be a string"]

    data = _CORRECT_DATA.copy()
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_version_format",
        MagicMock(return_value=False),
    )
    assert validate_basic_structure(data) == [
        "Field 'schema_version' must be in semver format (e.g., '1.0.0')"
    ]


def test_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_basic_structure run id validation."""
    data = _CORRECT_DATA.copy()
    data["run_id"] = 2  # type: ignore[assignment]
    assert validate_basic_structure(data) == ["Field 'run_id' must be a string"]

    data = _CORRECT_DATA.copy()
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_uuid_format",
        MagicMock(return_value=False),
    )
    assert validate_basic_structure(data) == ["Field 'run_id' must be a valid UUID"]


def test_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_basic_structure timestamp validation."""
    data = _CORRECT_DATA.copy()
    data["timestamp"] = 123  # type: ignore[assignment]
    assert validate_basic_structure(data) == ["Field 'timestamp' must be a string"]

    data = _CORRECT_DATA.copy()
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_timestamp_format",
        MagicMock(return_value=False),
    )
    assert validate_basic_structure(data) == [
        "Field 'timestamp' must be in ISO 8601 format"
    ]


def test_tool() -> None:
    """Test validate_basic_structure tool validation."""
    data = _CORRECT_DATA.copy()
    data["tool"] = "not a dict"
    assert validate_basic_structure(data) == ["Field 'tool' must be an object"]

    data = _CORRECT_DATA.copy()
    data["tool"] = {"version": "1.0.0"}
    assert validate_basic_structure(data) == ["Missing required field: tool.name"]

    data = _CORRECT_DATA.copy()
    data["tool"] = {"name": "MLIA"}
    assert validate_basic_structure(data) == ["Missing required field: tool.version"]

    data = _CORRECT_DATA.copy()
    data["tool"] = {"name": 123, "version": "1.0.0"}
    assert validate_basic_structure(data) == ["Field 'tool.name' must be a string"]

    data = _CORRECT_DATA.copy()
    data["tool"] = {"name": "MLIA", "version": 123}
    assert validate_basic_structure(data) == ["Field 'tool.version' must be a string"]


def test_target() -> None:
    """Test validate_basic_structure target validation."""
    data = _CORRECT_DATA.copy()
    data["target"] = "not a dict"
    assert validate_basic_structure(data) == ["Field 'target' must be an object"]

    data = _CORRECT_DATA.copy()
    data["target"] = {}
    errors = validate_basic_structure(data)
    assert "Missing required field: target.profile_name" in errors
    assert "Missing required field: target.target_type" in errors
    assert "Missing required field: target.components" in errors
    assert "Missing required field: target.configuration" in errors
    assert len(errors) == 4

    data = _CORRECT_DATA.copy()
    data["target"] = {
        "profile_name": "test",
        "target_type": "test",
        "components": "not a list",
        "configuration": {},
    }
    assert validate_basic_structure(data) == [
        "Field 'target.components' must be an array"
    ]

    data = _CORRECT_DATA.copy()
    data["target"] = {
        "profile_name": "test",
        "target_type": "test",
        "components": [],
        "configuration": {},
    }
    assert validate_basic_structure(data) == [
        "Field 'target.components' must have at least one item"
    ]


def test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_basic_structure model validation."""
    data = _CORRECT_DATA.copy()
    data["model"] = "not a dict"
    assert validate_basic_structure(data) == ["Field 'model' must be an object"]

    data = _CORRECT_DATA.copy()
    data["model"] = {}
    errors = validate_basic_structure(data)
    assert "Missing required field: model.name" in errors
    assert "Missing required field: model.format" in errors
    assert "Missing required field: model.hash" in errors
    assert len(errors) == 3

    data = _CORRECT_DATA.copy()
    data["model"] = {
        "name": 123,
        "format": "tflite",
        "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert validate_basic_structure(data) == ["Field 'model.name' must be a string"]

    data = _CORRECT_DATA.copy()
    data["model"] = {
        "name": "model.tflite",
        "format": 123,
        "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert validate_basic_structure(data) == ["Field 'model.format' must be a string"]

    data = _CORRECT_DATA.copy()
    data["model"] = {"name": "model.tflite", "format": "tflite", "hash": 123}
    assert validate_basic_structure(data) == ["Field 'model.hash' must be a string"]

    data = _CORRECT_DATA.copy()
    data["model"] = {"name": "model.tflite", "format": "tflite", "hash": "invalid"}
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_sha256_format",
        MagicMock(return_value=False),
    )
    assert validate_basic_structure(data) == [
        "Field 'model.hash' must be a valid SHA-256 hash"
    ]


def test_backends() -> None:
    """Test validate_basic_structure backends validation."""
    data = _CORRECT_DATA.copy()
    data["backends"] = "not a list"
    assert validate_basic_structure(data) == ["Field 'backends' must be an array"]

    data = _CORRECT_DATA.copy()
    data["backends"] = []
    assert validate_basic_structure(data) == [
        "Field 'backends' must have at least one item"
    ]


def test_results() -> None:
    """Test validate_basic_structure results validation."""
    data = _CORRECT_DATA.copy()
    data["results"] = "not a list"
    assert validate_basic_structure(data) == ["Field 'results' must be an array"]

    data = _CORRECT_DATA.copy()
    data["results"] = []
    assert not validate_basic_structure(data)


def test_multiple_errors() -> None:
    """Test validate_basic_structure with multiple validation errors."""
    data = {
        "schema_version": 123,  # Should be string
        "run_id": 456,  # Should be string
        "timestamp": 789,  # Should be string
        "tool": "not a dict",  # Should be dict
        "target": "not a dict",  # Should be dict
        "model": "not a dict",  # Should be dict
        "context": {},
        "backends": [],  # Should have at least one item
        "results": "not a list",  # Should be list
    }
    errors = validate_basic_structure(data)

    assert "Field 'schema_version' must be a string" in errors
    assert "Field 'run_id' must be a string" in errors
    assert "Field 'timestamp' must be a string" in errors
    assert "Field 'tool' must be an object" in errors
    assert "Field 'target' must be an object" in errors
    assert "Field 'model' must be an object" in errors
    assert "Field 'backends' must have at least one item" in errors
    assert "Field 'results' must be an array" in errors

    assert len(errors) == 8


def test_validate_standardized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_standardized_output."""
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        MagicMock(return_value=["error"]),
    )
    with pytest.raises(SchemaValidationError, match="Basic validation failed"):
        validate_standardized_output({}, False)

    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        MagicMock(return_value=[]),
    )

    warn_mock = MagicMock()
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", False)
    monkeypatch.setattr("warnings.warn", warn_mock)

    validate_standardized_output({}, False)
    warn_mock.assert_not_called()

    validate_standardized_output({}, True)
    warn_mock.assert_called_once()

    validate_with_jsonschema_mock = MagicMock()
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation.load_schema",
        MagicMock(return_value={"field": "schema"}),
    )
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_with_jsonschema",
        validate_with_jsonschema_mock,
    )
    validate_standardized_output({"key": "value"}, True)

    validate_with_jsonschema_mock.assert_called_once_with(
        {"key": "value"}, {"field": "schema"}
    )


@pytest.mark.parametrize("use_jsonschema", [True, False])
def test_validate_output_file(
    use_jsonschema: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test validate_output_file."""
    filepath = tmp_path / "file.json"
    with pytest.raises(FileNotFoundError, match="File not found"):
        validate_output_file(filepath)

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump({"key": "val"}, file)

    validate_standardized_output_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_standardized_output",
        validate_standardized_output_mock,
    )
    validate_output_file(filepath, use_jsonschema)

    validate_standardized_output_mock.assert_called_once_with(
        {"key": "val"}, use_jsonschema=use_jsonschema
    )
