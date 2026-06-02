# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Validation utilities for standardized output validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import mlia.core.output_schema as schema
from mlia.core.output_validation import (
    SchemaValidationError,
    _build_schema_registry,
    collect_validation_errors,
    load_schema,
    load_target_schema,
    validate_basic_structure,
    validate_output_file,
    validate_sha256_format,
    validate_standardized_output,
    validate_timestamp_format,
    validate_uuid_format,
    validate_version_format,
    validate_with_jsonschema,
)

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


def _valid_standardized_output(metric: dict[str, Any]) -> dict[str, Any]:
    """Build a valid standardized output payload containing one metric."""
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": "550e8400-e29b-41d4-a716-446655440000",
        "timestamp": "2025-12-29T10:30:00Z",
        "tool": {"name": "MLIA", "version": "1.0.0"},
        "target": {
            "profile_name": "ethos-u55-256",
            "target_type": "corstone-300",
            "components": [{"type": "npu", "family": "ethos-u"}],
            "configuration": {},
        },
        "model": {
            "name": "model.tflite",
            "format": "tflite",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        "context": {},
        "backends": [
            {"id": "vela", "name": "vela", "version": "3.0.0", "configuration": {}}
        ],
        "results": [
            {
                "kind": "performance",
                "status": "ok",
                "producer": "vela",
                "metrics": [metric],
            }
        ],
    }


def _valid_standardized_output_with_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build a valid standardized output payload containing one result."""
    output = _valid_standardized_output(
        {"name": "inference_time", "value": 1.0, "unit": "ms"}
    )
    output["results"] = [result]
    return output


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


def test_load_target_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test load_target_schema function."""
    schema_file = tmp_path / "resources" / "mlia-target-schema-1.0.0.json"
    mock_file_path = tmp_path / schema_file

    monkeypatch.setattr("mlia.core.output_validation.Path", lambda _: mock_file_path)
    monkeypatch.setattr("mlia.core.output_schema.SCHEMA_VERSION", "1.0.0")

    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        _ = load_target_schema()

    schema_content = {"target": "schema"}
    schema_file.parent.mkdir()
    with open(schema_file, "w", encoding="utf-8") as file:
        json.dump(schema_content, file)

    assert load_target_schema() == schema_content


def test_validate_with_jsonschema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_with_jsonschema function."""

    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", False)
    with pytest.raises(ImportError, match="jsonschema library is required"):
        validate_with_jsonschema({}, {})

    # pylint: disable=missing-class-docstring,too-few-public-methods,invalid-name
    class MockJsonSchema:
        class exceptions:
            class ValidationError(Exception):
                def __init__(self, message: str):
                    self.message = message
                    super().__init__(message)

    # pylint: enable=missing-class-docstring,too-few-public-methods,invalid-name

    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema", MockJsonSchema(), raising=False
    )

    monkeypatch.setattr(
        "mlia.core.output_validation._collect_jsonschema_errors",
        lambda _data, _schema: [],
    )
    validate_with_jsonschema({}, {})

    def raise_validation_error(_data: dict, _schema: dict) -> list[str]:
        raise MockJsonSchema.exceptions.ValidationError("Test validation error")

    monkeypatch.setattr(
        "mlia.core.output_validation._collect_jsonschema_errors",
        raise_validation_error,
    )
    with pytest.raises(
        SchemaValidationError, match="Schema validation failed: Test validation error"
    ):
        validate_with_jsonschema({}, {})


def test_build_schema_registry_requires_jsonschema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema registry building should require jsonschema support."""
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", False)

    with pytest.raises(ImportError, match="jsonschema library is required"):
        _build_schema_registry({"$id": "output"})


def test_build_schema_registry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schema registry helper should wire output and target schemas together."""

    class FakeResource:
        @staticmethod
        def from_contents(contents: dict[str, Any], _draft: object) -> tuple[str, dict]:
            return ("resource", contents)

    class FakeRegistryBuilder:
        def __init__(self) -> None:
            self.resources: list[tuple[str, tuple[str, dict]]] | None = None

        def with_resources(
            self, resources: list[tuple[str, tuple[str, dict]]]
        ) -> dict[str, object]:
            self.resources = resources
            return {"resources": resources}

    fake_registry_builder = FakeRegistryBuilder()
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation.load_target_schema",
        lambda: {"$id": "target-schema"},
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "referencing",
        MagicMock(Registry=lambda: fake_registry_builder, Resource=FakeResource),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "referencing.jsonschema",
        MagicMock(DRAFT202012="draft"),
    )

    registry = _build_schema_registry({"$id": "output-schema"})

    assert registry["resources"][0][0] == "output-schema"
    assert registry["resources"][1][0] == "target-schema"


def test_loaded_schema_requires_current_schema_version() -> None:
    """The loaded JSON Schema should require the current schema version."""
    loaded_schema = load_schema()

    assert loaded_schema["properties"]["schema_version"] == {
        "type": "string",
        "const": schema.SCHEMA_VERSION,
    }


def test_jsonschema_accepts_unavailable_metric_entry() -> None:
    """Schema validation should accept unavailable metric entries."""
    validate_standardized_output(
        _valid_standardized_output(
            {
                "name": schema.METRIC_NAME_CPU_UTILIZATION,
                "unit": schema.UNIT_PERCENT,
                "availability": "unavailable",
                "reason": "CPU utilization data is not available.",
            }
        )
    )


def test_jsonschema_rejects_mismatched_schema_version() -> None:
    """Schema 1.1.0 should reject payloads that claim another schema version."""
    output = _valid_standardized_output(
        {"name": "inference_time", "value": 1.0, "unit": "ms"}
    )
    output["schema_version"] = "1.0.0"

    with pytest.raises(SchemaValidationError, match="Schema validation failed"):
        validate_standardized_output(output)


def test_jsonschema_rejects_unavailable_metric_with_fake_value() -> None:
    """Unavailable metric entries should not contain fabricated values."""
    with pytest.raises(SchemaValidationError, match="Schema validation failed"):
        validate_standardized_output(
            _valid_standardized_output(
                {
                    "name": schema.METRIC_NAME_CPU_UTILIZATION,
                    "value": 0.0,
                    "unit": schema.UNIT_PERCENT,
                    "availability": "unavailable",
                    "reason": "CPU utilization data is not available.",
                }
            )
        )


def test_jsonschema_accepts_result_with_breakdown_and_entity() -> None:
    """Schema validation should accept breakdowns and entities."""
    validate_standardized_output(
        _valid_standardized_output_with_result(
            {
                "kind": "performance",
                "status": "ok",
                "producer": "backend",
                "metrics": [{"name": "inference_time", "value": 1.0, "unit": "ms"}],
                "breakdowns": [
                    {
                        "scope": "operator",
                        "name": "CONV_2D",
                        "location": "model/conv",
                        "metrics": [
                            {"name": "npu_cycles", "value": 1000, "unit": "cycles"}
                        ],
                    }
                ],
                "entities": [
                    {
                        "scope": "operator",
                        "name": "CONV_2D",
                        "location": "model/conv",
                        "placement": "NPU",
                    }
                ],
            }
        )
    )


def test_jsonschema_accepts_result_with_advice() -> None:
    """Schema validation should accept result-level advice."""
    validate_standardized_output(
        _valid_standardized_output_with_result(
            {
                "kind": "performance",
                "status": "ok",
                "producer": "backend",
                "advice": [
                    {
                        "id": "0",
                        "category": "performance",
                        "severity": "info",
                        "message": "Review the performance metrics.",
                        "affected_entities": [
                            {
                                "scope": "operator",
                                "name": "CONV_2D",
                                "location": "model/conv",
                            }
                        ],
                        "details": {"reason": "example"},
                    }
                ],
            }
        )
    )


def test_jsonschema_rejects_legacy_advices_field() -> None:
    """Schema validation should reject the legacy advices property."""
    with pytest.raises(SchemaValidationError, match="Schema validation failed"):
        validate_standardized_output(
            _valid_standardized_output_with_result(
                {
                    "kind": "performance",
                    "status": "ok",
                    "producer": "backend",
                    "advices": [
                        {
                            "id": "0",
                            "category": "performance",
                            "severity": "info",
                            "message": "Review the performance metrics.",
                        }
                    ],
                }
            )
        )


def test_jsonschema_rejects_uppercase_advice_values() -> None:
    """Schema validation should reject enum member names in advice JSON."""
    with pytest.raises(SchemaValidationError, match="Schema validation failed"):
        validate_standardized_output(
            _valid_standardized_output_with_result(
                {
                    "kind": "performance",
                    "status": "ok",
                    "producer": "backend",
                    "advice": [
                        {
                            "id": "0",
                            "category": "PERFORMANCE",
                            "severity": "INFO",
                            "message": "Review the performance metrics.",
                        }
                    ],
                }
            )
        )


def test_schema_1_0_does_not_define_advice() -> None:
    """Schema 1.0.0 should remain unchanged for advice fields."""
    schema_path = (
        Path(__file__).parents[1]
        / "src"
        / "mlia"
        / "resources"
        / "mlia-output-schema-1.0.0.json"
    )
    schema_1_0 = json.loads(schema_path.read_text(encoding="utf-8"))
    result_properties = schema_1_0["properties"]["results"]["items"]["properties"]

    assert "advice" not in result_properties
    assert "advices" not in result_properties


def test_validate_with_jsonschema_raises_on_collected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collected jsonschema errors should be re-raised as SchemaValidationError."""
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation._collect_jsonschema_errors",
        lambda _data, _schema: ["first error"],
    )

    with pytest.raises(SchemaValidationError, match="first error"):
        validate_with_jsonschema({}, {})


def test_collect_jsonschema_errors_uses_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collected jsonschema errors should come from the configured validator."""

    class FakeError:
        def __init__(self, message: str):
            self.message = message

    class FakeValidator:
        def __init__(self, _schema: dict, registry: object) -> None:
            self.registry = registry

        def iter_errors(self, _data: dict) -> list[FakeError]:
            return [FakeError("err1"), FakeError("err2")]

    monkeypatch.setattr(
        "mlia.core.output_validation._build_schema_registry",
        lambda _schema: "registry",
    )
    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema",
        MagicMock(Draft202012Validator=FakeValidator),
        raising=False,
    )

    from mlia.core.output_validation import _collect_jsonschema_errors

    assert _collect_jsonschema_errors({}, {"$id": "schema"}) == ["err1", "err2"]


def test_collect_validation_errors_use_jsonschema_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation collection can stop after basic structure checks."""
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        lambda _data: ["basic error"],
    )

    assert collect_validation_errors({}, use_jsonschema=False) == ["basic error"]


def test_collect_validation_errors_warns_without_jsonschema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing jsonschema should warn and return basic validation errors."""
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        lambda _data: ["basic error"],
    )
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", False)

    with pytest.warns(UserWarning, match="jsonschema library not available"):
        assert collect_validation_errors({}) == ["basic error"]


def test_collect_validation_errors_handles_validator_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation collection should append jsonschema errors without raising."""

    class MockJsonSchema:
        class exceptions:
            class ValidationError(Exception):
                def __init__(self, message: str):
                    self.message = message
                    super().__init__(message)

    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        lambda _data: [],
    )
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema", MockJsonSchema(), raising=False
    )
    monkeypatch.setattr("mlia.core.output_validation.load_schema", lambda: {"$id": "x"})

    def raise_validation_error(_data: dict, _schema: dict) -> list[str]:
        raise MockJsonSchema.exceptions.ValidationError("jsonschema exploded")

    monkeypatch.setattr(
        "mlia.core.output_validation._collect_jsonschema_errors",
        raise_validation_error,
    )

    assert collect_validation_errors({}) == ["jsonschema exploded"]


def test_collect_validation_errors_handles_schema_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-loading failures should be reported as validation errors."""
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        lambda _data: [],
    )
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr(
        "mlia.core.output_validation.load_schema",
        lambda: (_ for _ in ()).throw(FileNotFoundError("schema file missing")),
    )
    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema", MagicMock(), raising=False
    )
    assert collect_validation_errors({}) == [
        "Schema validation could not be completed: schema file missing"
    ]


def test_collect_validation_errors_handles_validator_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validator setup failures should be reported without raising."""
    monkeypatch.setattr(
        "mlia.core.output_validation.validate_basic_structure",
        lambda _data: [],
    )
    monkeypatch.setattr("mlia.core.output_validation.JSONSCHEMA_AVAILABLE", True)
    monkeypatch.setattr("mlia.core.output_validation.load_schema", lambda: {"$id": "x"})
    monkeypatch.setattr(
        "mlia.core.output_validation._collect_jsonschema_errors",
        lambda _data, _schema: (_ for _ in ()).throw(
            RuntimeError("validator setup failed")
        ),
    )
    monkeypatch.setattr(
        "mlia.core.output_validation.jsonschema", MagicMock(), raising=False
    )
    assert collect_validation_errors({}) == [
        "Schema validation could not be completed: validator setup failed"
    ]


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
