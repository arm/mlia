# SPDX-FileCopyrightText: Copyright 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standardized output schema classes."""
import tempfile
from pathlib import Path

import pytest

from mlia.core.output_schema import Backend
from mlia.core.output_schema import Component
from mlia.core.output_schema import ComponentType
from mlia.core.output_schema import Context
from mlia.core.output_schema import Metric
from mlia.core.output_schema import Model
from mlia.core.output_schema import Result
from mlia.core.output_schema import ResultKind
from mlia.core.output_schema import ResultStatus
from mlia.core.output_schema import SCHEMA_VERSION
from mlia.core.output_schema import StandardizedOutput
from mlia.core.output_schema import Target
from mlia.core.output_schema import TARGET_SCHEMA_VERSION
from mlia.core.output_schema import Tool
from mlia.core.output_validation import load_schema
from mlia.core.output_validation import SchemaValidationError
from mlia.core.output_validation import validate_basic_structure
from mlia.core.output_validation import validate_sha256_format
from mlia.core.output_validation import validate_standardized_output
from mlia.core.output_validation import validate_timestamp_format
from mlia.core.output_validation import validate_uuid_format
from mlia.core.output_validation import validate_version_format


class TestTool:
    """Test Tool class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tool = Tool(name="mlia", version="1.0.0")
        assert tool.to_dict() == {"name": "mlia", "version": "1.0.0"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"name": "mlia", "version": "1.0.0"}
        tool = Tool.from_dict(data)
        assert tool.name == "mlia"
        assert tool.version == "1.0.0"


class TestBackend:
    """Test Backend class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        backend = Backend(
            id="vela",
            name="Vela Compiler",
            version="3.10.0",
            configuration={"option": "value"},
        )
        result = backend.to_dict()
        assert result["id"] == "vela"
        assert result["name"] == "Vela Compiler"
        assert result["version"] == "3.10.0"
        assert result["configuration"] == {"option": "value"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "vela",
            "name": "Vela Compiler",
            "version": "3.10.0",
            "configuration": {"option": "value"},
        }
        backend = Backend.from_dict(data)
        assert backend.id == "vela"
        assert backend.name == "Vela Compiler"


class TestComponent:
    """Test Component class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        component = Component(
            type=ComponentType.NPU,
            family="ethos-u",
            model="u55",
            variant="256",
        )
        result = component.to_dict()
        assert result["type"] == "npu"
        assert result["family"] == "ethos-u"
        assert result["model"] == "u55"
        assert result["variant"] == "256"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "type": "npu",
            "family": "ethos-u",
            "model": "u55",
            "variant": "256",
        }
        component = Component.from_dict(data)
        assert component.type == ComponentType.NPU
        assert component.family == "ethos-u"


class TestTarget:
    """Test Target class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        target = Target(
            profile_name="ethos-u55-256",
            target_type="ethos-u55",
            components=[
                Component(
                    type=ComponentType.NPU, family="ethos-u", model="u55", variant="256"
                )
            ],
            configuration={"param": "value"},
        )
        result = target.to_dict()
        assert result["profile_name"] == "ethos-u55-256"
        assert len(result["components"]) == 1

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "profile_name": "ethos-u55-256",
            "target_type": "ethos-u55",
            "components": [
                {"type": "npu", "family": "ethos-u", "model": "u55", "variant": "256"}
            ],
            "configuration": {},
        }
        target = Target.from_dict(data)
        assert target.profile_name == "ethos-u55-256"


class TestModel:
    """Test Model class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        model = Model(
            name="mobilenet.tflite",
            format="tflite",
            hash="a" * 64,
            size_bytes=1024,
        )
        result = model.to_dict()
        assert result["name"] == "mobilenet.tflite"
        assert result["format"] == "tflite"
        assert result["hash"] == "a" * 64
        assert result["size_bytes"] == 1024

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"name": "mobilenet.tflite", "format": "tflite", "hash": "a" * 64}
        model = Model.from_dict(data)
        assert model.name == "mobilenet.tflite"


class TestMetric:
    """Test Metric class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metric = Metric(name="inference_time", value=10.5, unit="ms")
        assert metric.to_dict() == {
            "name": "inference_time",
            "value": 10.5,
            "unit": "ms",
        }

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"name": "inference_time", "value": 10.5, "unit": "ms"}
        metric = Metric.from_dict(data)
        assert metric.name == "inference_time"
        assert metric.value == 10.5


class TestResult:
    """Test Result class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = Result(
            kind=ResultKind.PERFORMANCE,
            status=ResultStatus.OK,
            producer="vela",
            metrics=[Metric(name="cycles", value=1000, unit="cycles")],
        )
        result_dict = result.to_dict()
        assert result_dict["kind"] == "performance"
        assert result_dict["status"] == "ok"
        assert result_dict["producer"] == "vela"
        assert len(result_dict["metrics"]) == 1

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "kind": "performance",
            "status": "ok",
            "producer": "vela",
            "metrics": [{"name": "cycles", "value": 1000, "unit": "cycles"}],
        }
        result = Result.from_dict(data)
        assert result.kind == ResultKind.PERFORMANCE
        assert result.status == ResultStatus.OK


class TestStandardizedOutput:
    """Test StandardizedOutput class."""

    def test_create_timestamp(self) -> None:
        """Test timestamp creation."""
        timestamp = StandardizedOutput.create_timestamp()
        assert validate_timestamp_format(timestamp)

    def test_create_run_id(self) -> None:
        """Test run_id creation."""
        run_id = StandardizedOutput.create_run_id()
        assert validate_uuid_format(run_id)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        output = StandardizedOutput(
            schema_version=SCHEMA_VERSION,
            run_id=StandardizedOutput.create_run_id(),
            timestamp=StandardizedOutput.create_timestamp(),
            tool=Tool(name="mlia", version="1.0.0"),
            target=Target(
                profile_name="ethos-u55-256",
                target_type="ethos-u55",
                components=[
                    Component(
                        type=ComponentType.NPU,
                        family="ethos-u",
                        model="u55",
                        variant="256",
                    )
                ],
                configuration={},
            ),
            model=Model(name="model.tflite", format="tflite", hash="a" * 64),
            context=Context(),
            backends=[
                Backend(id="vela", name="Vela", version="3.10.0", configuration={})
            ],
            results=[
                Result(
                    kind=ResultKind.PERFORMANCE,
                    status=ResultStatus.OK,
                    producer="vela",
                )
            ],
        )
        result_dict = output.to_dict()
        assert result_dict["schema_version"] == SCHEMA_VERSION
        assert "run_id" in result_dict
        assert "timestamp" in result_dict

    def test_serialization_roundtrip(self) -> None:
        """Test serialization and deserialization."""
        output = StandardizedOutput(
            schema_version=SCHEMA_VERSION,
            run_id=StandardizedOutput.create_run_id(),
            timestamp=StandardizedOutput.create_timestamp(),
            tool=Tool(name="mlia", version="1.0.0"),
            target=Target(
                profile_name="ethos-u55-256",
                target_type="ethos-u55",
                components=[
                    Component(
                        type=ComponentType.NPU,
                        family="ethos-u",
                        model="u55",
                        variant="256",
                    )
                ],
                configuration={},
            ),
            model=Model(name="model.tflite", format="tflite", hash="a" * 64),
            context=Context(),
            backends=[
                Backend(id="vela", name="Vela", version="3.10.0", configuration={})
            ],
            results=[],
        )
        json_str = output.to_json()
        loaded = StandardizedOutput.from_json(json_str)
        assert loaded.schema_version == output.schema_version
        assert loaded.run_id == output.run_id

    def test_save_and_load(self) -> None:
        """Test saving and loading from file."""
        output = StandardizedOutput(
            schema_version=SCHEMA_VERSION,
            run_id=StandardizedOutput.create_run_id(),
            timestamp=StandardizedOutput.create_timestamp(),
            tool=Tool(name="mlia", version="1.0.0"),
            target=Target(
                profile_name="ethos-u55-256",
                target_type="ethos-u55",
                components=[
                    Component(
                        type=ComponentType.NPU,
                        family="ethos-u",
                        model="u55",
                        variant="256",
                    )
                ],
                configuration={},
            ),
            model=Model(name="model.tflite", format="tflite", hash="a" * 64),
            context=Context(),
            backends=[
                Backend(id="vela", name="Vela", version="3.10.0", configuration={})
            ],
            results=[],
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            filepath = Path(temp_file.name)

        try:
            output.save(filepath)
            loaded = StandardizedOutput.load(filepath)
            assert loaded.schema_version == output.schema_version
            assert loaded.tool.name == output.tool.name
        finally:
            filepath.unlink()


class TestValidation:
    """Test validation functions."""

    def test_validate_version_format(self) -> None:
        """Test version format validation."""
        assert validate_version_format("1.0.0")
        assert validate_version_format("10.20.30")
        assert not validate_version_format("1.0")
        assert not validate_version_format("v1.0.0")

    def test_validate_uuid_format(self) -> None:
        """Test UUID format validation."""
        assert validate_uuid_format("550e8400-e29b-41d4-a716-446655440000")
        assert not validate_uuid_format("invalid-uuid")
        assert not validate_uuid_format("550e8400e29b41d4a716446655440000")

    def test_validate_sha256_format(self) -> None:
        """Test SHA-256 format validation."""
        assert validate_sha256_format("a" * 64)
        assert validate_sha256_format("A" * 64)
        assert not validate_sha256_format("a" * 63)
        assert not validate_sha256_format("g" * 64)

    def test_validate_basic_structure(self) -> None:
        """Test basic structure validation."""
        valid_data = {
            "schema_version": "1.0.0",
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "timestamp": "2025-01-01T00:00:00Z",
            "tool": {"name": "mlia", "version": "1.0.0"},
            "target": {
                "profile_name": "test",
                "target_type": "ethos-u55",
                "components": [{"type": "npu", "family": "ethos-u"}],
                "configuration": {},
            },
            "model": {"name": "test.tflite", "format": "tflite", "hash": "a" * 64},
            "context": {},
            "backends": [
                {"id": "test", "name": "Test", "version": "1.0.0", "configuration": {}}
            ],
            "results": [],
        }
        errors = validate_basic_structure(valid_data)
        assert len(errors) == 0

    def test_validate_invalid_output(self) -> None:
        """Test validation of invalid output."""
        data = {"schema_version": "invalid"}
        with pytest.raises(SchemaValidationError):
            validate_standardized_output(data, use_jsonschema=False)

    def test_load_schema(self) -> None:
        """Test loading the JSON schema file."""
        schema = load_schema()
        assert schema is not None
        assert "$schema" in schema
        assert "$id" in schema
        assert (
            schema["$id"]
            == f"https://schemas.arm.com/mlia/output-schema-{SCHEMA_VERSION}.json"
        )
        assert "properties" in schema
        assert "target" in schema["properties"]
        # Verify target references the child schema
        assert "$ref" in schema["properties"]["target"]
        assert (
            schema["properties"]["target"]["$ref"]
            == f"https://schemas.arm.com/mlia/target-{TARGET_SCHEMA_VERSION}.json"
        )
