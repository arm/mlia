# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standardized output schema classes."""
import tempfile
from pathlib import Path

import pytest

import mlia.core.output_schema as schema
from mlia.core import output_validation


class TestTool:
    """Test Tool class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        tool = schema.Tool(name="mlia", version="1.0.0")
        assert tool.to_dict() == {"name": "mlia", "version": "1.0.0"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"name": "mlia", "version": "1.0.0"}
        tool = schema.Tool.from_dict(data)
        assert tool.name == "mlia"
        assert tool.version == "1.0.0"


class TestBackend:
    """Test Backend class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        backend = schema.Backend(
            id="vela",
            name="Vela Compiler",
            version="3.10.0",
            configuration={"option": "value"},
            impl={"backend": "option"},
        )
        result = backend.to_dict()
        assert result["id"] == "vela"
        assert result["name"] == "Vela Compiler"
        assert result["version"] == "3.10.0"
        assert result["configuration"] == {"option": "value"}
        assert result["impl"] == {"backend": "option"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "vela",
            "name": "Vela Compiler",
            "version": "3.10.0",
            "configuration": {"option": "value"},
        }
        backend = schema.Backend.from_dict(data)
        assert backend.id == "vela"
        assert backend.name == "Vela Compiler"


class TestComponent:
    """Test Component class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        component = schema.Component(
            type=schema.ComponentType.NPU,
            family="ethos-u",
            model="u55",
            variant="256",
            name="ethos-u55-256",
            components=[
                schema.Component(
                    type=schema.ComponentType.SPECIFICATION, family="some-family"
                ),
            ],
        )
        result = component.to_dict()
        assert result["type"] == "npu"
        assert result["family"] == "ethos-u"
        assert result["model"] == "u55"
        assert result["variant"] == "256"
        assert result["name"] == "ethos-u55-256"
        assert result["components"] == [
            {"type": "specification", "family": "some-family"}
        ]

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "type": "npu",
            "family": "ethos-u",
            "model": "u55",
            "variant": "256",
        }
        component = schema.Component.from_dict(data)
        assert component.type == schema.ComponentType.NPU
        assert component.family == "ethos-u"


class TestTarget:
    """Test Target class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        target = schema.Target(
            profile_name="ethos-u55-256",
            target_type="ethos-u55",
            components=[
                schema.Component(
                    type=schema.ComponentType.NPU,
                    family="ethos-u",
                    model="u55",
                    variant="256",
                )
            ],
            configuration={"param": "value"},
            host_platform="linux",
        )
        result = target.to_dict()
        assert result["profile_name"] == "ethos-u55-256"
        assert len(result["components"]) == 1
        assert result["host_platform"] == "linux"

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
        target = schema.Target.from_dict(data)
        assert target.profile_name == "ethos-u55-256"


class TestModel:
    """Test Model class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        model = schema.Model(
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
        model = schema.Model.from_dict(data)
        assert model.name == "mobilenet.tflite"


class TestContext:
    """Test Context class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        context = schema.Context(
            runtime_configuration={"python_version": "3.10.0", "os": "linux"},
            git={"commit": "abc123", "branch": "main"},
            notes="Test run with new model",
            cli_arguments=["--target", "ethos-u55", "--optimize"],
        )
        result = context.to_dict()
        assert result["runtime_configuration"] == {
            "python_version": "3.10.0",
            "os": "linux",
        }
        assert result["git"] == {"commit": "abc123", "branch": "main"}
        assert result["notes"] == "Test run with new model"
        assert result["cli_arguments"] == ["--target", "ethos-u55", "--optimize"]

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "runtime_configuration": {"python_version": "3.10.0", "os": "linux"},
            "git": {"commit": "abc123", "branch": "main"},
            "notes": "Test run with new model",
            "cli_arguments": ["--target", "ethos-u55", "--optimize"],
        }
        context = schema.Context.from_dict(data)
        assert context.runtime_configuration == {
            "python_version": "3.10.0",
            "os": "linux",
        }
        assert context.git == {"commit": "abc123", "branch": "main"}
        assert context.notes == "Test run with new model"
        assert context.cli_arguments == ["--target", "ethos-u55", "--optimize"]


class TestMetric:
    """Test Metric class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metric = schema.Metric(
            name="inference_time",
            value=10.5,
            unit="ms",
            aggregation="sum",
            samples=5,
            qualifiers={"key": "value"},
        )

        assert metric.to_dict() == {
            "name": "inference_time",
            "value": 10.5,
            "unit": "ms",
            "aggregation": "sum",
            "samples": 5,
            "qualifiers": {"key": "value"},
        }

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"name": "inference_time", "value": 10.5, "unit": "ms"}
        metric = schema.Metric.from_dict(data)
        assert metric.name == "inference_time"
        assert metric.value == 10.5


class TestOperatorIdentifier:
    """Test OperatorIdentifier class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        operator_id = schema.OperatorIdentifier(
            scope=schema.OperatorScope.OPERATOR,
            name="CONV_2D",
            location="layer_0",
            id="op_001",
        )
        result = operator_id.to_dict()
        assert result["scope"] == "operator"
        assert result["name"] == "CONV_2D"
        assert result["location"] == "layer_0"
        assert result["id"] == "op_001"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "scope": "operator",
            "name": "CONV_2D",
            "location": "layer_0",
            "id": "op_001",
        }
        operator_id = schema.OperatorIdentifier.from_dict(data)
        assert operator_id.scope == schema.OperatorScope.OPERATOR
        assert operator_id.name == "CONV_2D"
        assert operator_id.location == "layer_0"
        assert operator_id.id == "op_001"


class TestBreakdown:
    """Test Breakdown class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        breakdown = schema.Breakdown(
            scope=schema.OperatorScope.OPERATOR,
            name="CONV_2D",
            location="layer_0",
            metrics=[
                schema.Metric(name="cycles", value=1000, unit="cycles"),
                schema.Metric(name="energy", value=50.5, unit="mJ"),
            ],
            id="op_001",
            qualifiers={"device": "npu", "precision": "int8"},
        )
        result = breakdown.to_dict()
        assert result["scope"] == "operator"
        assert result["name"] == "CONV_2D"
        assert result["location"] == "layer_0"
        assert len(result["metrics"]) == 2
        assert result["id"] == "op_001"
        assert result["qualifiers"] == {"device": "npu", "precision": "int8"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "scope": "operator",
            "name": "CONV_2D",
            "location": "layer_0",
            "metrics": [
                {"name": "cycles", "value": 1000, "unit": "cycles"},
                {"name": "energy", "value": 50.5, "unit": "mJ"},
            ],
            "id": "op_001",
            "qualifiers": {"device": "npu", "precision": "int8"},
        }
        breakdown = schema.Breakdown.from_dict(data)
        assert breakdown.scope == schema.OperatorScope.OPERATOR
        assert breakdown.name == "CONV_2D"
        assert breakdown.location == "layer_0"
        assert len(breakdown.metrics) == 2
        assert breakdown.id == "op_001"
        assert breakdown.qualifiers == {"device": "npu", "precision": "int8"}


class TestCheck:
    """Test Check class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        check = schema.Check(
            id="compatibility_check",
            status=schema.CheckStatus.PASS,
            details={"message": "All operators supported", "count": 42},
        )
        result = check.to_dict()
        assert result["id"] == "compatibility_check"
        assert result["status"] == "pass"
        assert result["details"] == {"message": "All operators supported", "count": 42}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "compatibility_check",
            "status": "pass",
            "details": {"message": "All operators supported", "count": 42},
        }
        check = schema.Check.from_dict(data)
        assert check.id == "compatibility_check"
        assert check.status == schema.CheckStatus.PASS
        assert check.details == {"message": "All operators supported", "count": 42}


class TestEntity:
    """Test Entity class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        entity = schema.Entity(
            scope=schema.OperatorScope.OPERATOR,
            name="CONV_2D",
            location="layer_0",
            placement="npu",
            id="entity_001",
            attributes={"dtype": "int8", "kernel_size": [3, 3]},
        )
        result = entity.to_dict()
        assert result["scope"] == "operator"
        assert result["name"] == "CONV_2D"
        assert result["location"] == "layer_0"
        assert result["placement"] == "npu"
        assert result["id"] == "entity_001"
        assert result["attributes"] == {"dtype": "int8", "kernel_size": [3, 3]}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "scope": "operator",
            "name": "CONV_2D",
            "location": "layer_0",
            "placement": "npu",
            "id": "entity_001",
            "attributes": {"dtype": "int8", "kernel_size": [3, 3]},
        }
        entity = schema.Entity.from_dict(data)
        assert entity.scope == schema.OperatorScope.OPERATOR
        assert entity.name == "CONV_2D"
        assert entity.location == "layer_0"
        assert entity.placement == "npu"
        assert entity.id == "entity_001"
        assert entity.attributes == {"dtype": "int8", "kernel_size": [3, 3]}


class TestResult:
    """Test Result class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer="vela",
            metrics=[schema.Metric(name="cycles", value=1000, unit="cycles")],
            warnings=["warning"],
            errors=["error"],
        )
        result_dict = result.to_dict()
        assert result_dict["kind"] == "performance"
        assert result_dict["status"] == "ok"
        assert result_dict["producer"] == "vela"
        assert result_dict["warnings"] == ["warning"]
        assert result_dict["errors"] == ["error"]
        assert len(result_dict["metrics"]) == 1

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "kind": "performance",
            "status": "ok",
            "producer": "vela",
            "metrics": [{"name": "cycles", "value": 1000, "unit": "cycles"}],
        }
        result = schema.Result.from_dict(data)
        assert result.kind == schema.ResultKind.PERFORMANCE
        assert result.status == schema.ResultStatus.OK


class TestStandardizedOutput:
    """Test StandardizedOutput class."""

    def test_create_timestamp(self) -> None:
        """Test timestamp creation."""
        timestamp = schema.StandardizedOutput.create_timestamp()
        assert output_validation.validate_timestamp_format(timestamp)

    def test_create_run_id(self) -> None:
        """Test run_id creation."""
        run_id = schema.StandardizedOutput.create_run_id()
        assert output_validation.validate_uuid_format(run_id)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        output = schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=schema.StandardizedOutput.create_run_id(),
            timestamp=schema.StandardizedOutput.create_timestamp(),
            tool=schema.Tool(name="mlia", version="1.0.0"),
            target=schema.Target(
                profile_name="ethos-u55-256",
                target_type="ethos-u55",
                components=[
                    schema.Component(
                        type=schema.ComponentType.NPU,
                        family="ethos-u",
                        model="u55",
                        variant="256",
                    )
                ],
                configuration={},
            ),
            model=schema.Model(name="model.tflite", format="tflite", hash="a" * 64),
            context=schema.Context(),
            backends=[
                schema.Backend(
                    id="vela", name="Vela", version="3.10.0", configuration={}
                )
            ],
            results=[
                schema.Result(
                    kind=schema.ResultKind.PERFORMANCE,
                    status=schema.ResultStatus.OK,
                    producer="vela",
                )
            ],
            extensions={"ext0": "val0"},
        )
        result_dict = output.to_dict()
        assert result_dict["schema_version"] == schema.SCHEMA_VERSION
        assert "run_id" in result_dict
        assert "timestamp" in result_dict
        assert "extensions" in result_dict

    def test_serialization_roundtrip(self) -> None:
        """Test serialization and deserialization."""
        output = schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=schema.StandardizedOutput.create_run_id(),
            timestamp=schema.StandardizedOutput.create_timestamp(),
            tool=schema.Tool(name="mlia", version="1.0.0"),
            target=schema.Target(
                profile_name="ethos-u55-256",
                target_type="ethos-u55",
                components=[
                    schema.Component(
                        type=schema.ComponentType.NPU,
                        family="ethos-u",
                        model="u55",
                        variant="256",
                    )
                ],
                configuration={},
            ),
            model=schema.Model(name="model.tflite", format="tflite", hash="a" * 64),
            context=schema.Context(),
            backends=[
                schema.Backend(
                    id="vela", name="Vela", version="3.10.0", configuration={}
                )
            ],
            results=[],
        )
        json_str = output.to_json()
        loaded = schema.StandardizedOutput.from_json(json_str)
        assert loaded.schema_version == output.schema_version
        assert loaded.run_id == output.run_id

    def test_save_and_load(self) -> None:
        """Test saving and loading from file."""
        output = schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=schema.StandardizedOutput.create_run_id(),
            timestamp=schema.StandardizedOutput.create_timestamp(),
            tool=schema.Tool(name="mlia", version="1.0.0"),
            target=schema.Target(
                profile_name="ethos-u55-256",
                target_type="ethos-u55",
                components=[
                    schema.Component(
                        type=schema.ComponentType.NPU,
                        family="ethos-u",
                        model="u55",
                        variant="256",
                    )
                ],
                configuration={},
            ),
            model=schema.Model(name="model.tflite", format="tflite", hash="a" * 64),
            context=schema.Context(),
            backends=[
                schema.Backend(
                    id="vela", name="Vela", version="3.10.0", configuration={}
                )
            ],
            results=[],
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            filepath = Path(temp_file.name)

        try:
            output.save(filepath)
            loaded = schema.StandardizedOutput.load(filepath)
            assert loaded.schema_version == output.schema_version
            assert loaded.tool.name == output.tool.name
        finally:
            filepath.unlink()


class TestValidation:
    """Test validation functions."""

    def test_validate_version_format(self) -> None:
        """Test version format validation."""
        assert output_validation.validate_version_format("1.0.0")
        assert output_validation.validate_version_format("10.20.30")
        assert not output_validation.validate_version_format("1.0")
        assert not output_validation.validate_version_format("v1.0.0")

    def test_validate_uuid_format(self) -> None:
        """Test UUID format validation."""
        assert output_validation.validate_uuid_format(
            "550e8400-e29b-41d4-a716-446655440000"
        )
        assert not output_validation.validate_uuid_format("invalid-uuid")
        assert not output_validation.validate_uuid_format(
            "550e8400e29b41d4a716446655440000"
        )

    def test_validate_sha256_format(self) -> None:
        """Test SHA-256 format validation."""
        assert output_validation.validate_sha256_format("a" * 64)
        assert output_validation.validate_sha256_format("A" * 64)
        assert not output_validation.validate_sha256_format("a" * 63)
        assert not output_validation.validate_sha256_format("g" * 64)

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
        errors = output_validation.validate_basic_structure(valid_data)
        assert len(errors) == 0

    def test_validate_invalid_output(self) -> None:
        """Test validation of invalid output."""
        data = {"schema_version": "invalid"}
        with pytest.raises(output_validation.SchemaValidationError):
            output_validation.validate_standardized_output(data, use_jsonschema=False)

    def test_load_schema(self) -> None:
        """Test loading the JSON schema file."""
        output_schema = output_validation.load_schema()
        assert output_schema is not None
        assert "$schema" in output_schema
        assert "$id" in output_schema
        expected_id = (
            f"https://schemas.arm.com/mlia/output-schema-{schema.SCHEMA_VERSION}.json"
        )
        assert output_schema["$id"] == expected_id
        assert "properties" in output_schema
        assert "target" in output_schema["properties"]
        # Verify target references the child schema
        assert "$ref" in output_schema["properties"]["target"]
        expected_ref = (
            f"https://schemas.arm.com/mlia/target-{schema.TARGET_SCHEMA_VERSION}.json"
        )
        assert output_schema["properties"]["target"]["$ref"] == expected_ref
