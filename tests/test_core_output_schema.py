# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for standardized output schema classes."""

import tempfile
from pathlib import Path
from typing import Any, cast

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
        )
        result = context.to_dict()
        assert result["runtime_configuration"] == {
            "python_version": "3.10.0",
            "os": "linux",
        }
        assert result["git"] == {"commit": "abc123", "branch": "main"}
        assert result["notes"] == "Test run with new model"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "runtime_configuration": {"python_version": "3.10.0", "os": "linux"},
            "git": {"commit": "abc123", "branch": "main"},
            "notes": "Test run with new model",
        }
        context = schema.Context.from_dict(data)
        assert context.runtime_configuration == {
            "python_version": "3.10.0",
            "os": "linux",
        }
        assert context.git == {"commit": "abc123", "branch": "main"}
        assert context.notes == "Test run with new model"


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

    def test_well_known_aggregation_type_to_dict(self) -> None:
        """Well-known aggregation types should serialize as schema strings."""
        assert {aggregation.value for aggregation in schema.AggregationType} == {
            "sum",
            "max",
            "min",
            "mean",
        }
        metric = schema.Metric(
            name="cycles",
            value=10,
            unit="cycles",
            aggregation=schema.AggregationType.MAX,
        )

        assert metric.to_dict()["aggregation"] == "max"

    def test_from_dict_preserves_unknown_aggregation_string(self) -> None:
        """Unknown aggregation policies should remain forward compatible."""
        metric = schema.Metric.from_dict(
            {
                "name": "cycles",
                "value": 10,
                "unit": "cycles",
                "aggregation": "future-policy",
            }
        )

        assert metric.aggregation == "future-policy"
        assert metric.to_dict()["aggregation"] == "future-policy"

    def test_unavailable_to_dict(self) -> None:
        """Test conversion of unavailable metric to dictionary."""
        metric = schema.Metric(
            name="metric_name",
            value=None,
            unit="metric_unit",
            availability=schema.MetricAvailability.UNAVAILABLE,
            reason="Metric is unavailable.",
        )

        assert metric.to_dict() == {
            "name": "metric_name",
            "unit": "metric_unit",
            "availability": "unavailable",
            "reason": "Metric is unavailable.",
        }

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"name": "inference_time", "value": 10.5, "unit": "ms"}
        metric = schema.Metric.from_dict(data)
        assert metric.name == "inference_time"
        assert metric.value == 10.5

    def test_unavailable_from_dict(self) -> None:
        """Test creation of unavailable metric from dictionary."""
        metric = schema.Metric.from_dict(
            {
                "name": schema.METRIC_NAME_CPU_UTILIZATION,
                "unit": schema.UNIT_PERCENT,
                "availability": "unavailable",
                "reason": "CPU utilization data is not available.",
            }
        )

        assert metric.name == schema.METRIC_NAME_CPU_UTILIZATION
        assert metric.value is None
        assert metric.availability == schema.MetricAvailability.UNAVAILABLE
        assert metric.reason == "CPU utilization data is not available."

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_init_rejects_non_finite_values(self, value: float) -> None:
        """Metric values must be representable as strict JSON numbers."""
        with pytest.raises(ValueError, match="must be finite"):
            schema.Metric(name="metric_name", value=value, unit="metric_unit")

    @pytest.mark.parametrize(
        ("metric_kwargs", "match"),
        [
            (
                {
                    "value": 10.5,
                    "availability": schema.MetricAvailability.UNAVAILABLE,
                    "reason": "Metric is unavailable.",
                },
                "cannot combine",
            ),
            ({"value": None}, "must include a value or availability"),
            (
                {"value": None, "availability": schema.MetricAvailability.UNAVAILABLE},
                "must include a reason",
            ),
            (
                {
                    "value": None,
                    "availability": schema.MetricAvailability.UNAVAILABLE,
                    "reason": "Metric is unavailable.",
                    "aggregation": "sum",
                },
                "cannot include aggregation or samples",
            ),
        ],
    )
    def test_metric_init_rejects_invalid_shapes(
        self, metric_kwargs: dict[str, Any], match: str
    ) -> None:
        """Metric construction should reject shapes not allowed by the schema."""
        with pytest.raises(ValueError, match=match):
            schema.Metric(name="metric_name", unit="metric_unit", **metric_kwargs)


class TestEnsureStandardPerformanceMetrics:
    """Test standard performance metric helpers."""

    def test_ensure_standard_performance_metrics_preserves_supplied_metric(
        self,
    ) -> None:
        """Existing metrics should be preserved."""
        throughput = schema.Metric(
            name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
            value=42.0,
            unit=schema.UNIT_INFERENCES_PER_SECOND,
        )

        metrics = schema.ensure_standard_performance_metrics([throughput])

        assert metrics[0] == throughput
        assert metrics[0].to_dict() == {
            "name": schema.METRIC_NAME_INFERENCES_PER_SECOND,
            "value": 42.0,
            "unit": schema.UNIT_INFERENCES_PER_SECOND,
        }

    def test_ensure_standard_performance_metrics_fills_missing_metrics(
        self,
    ) -> None:
        """Missing standard metrics should be represented as unavailable."""
        throughput = schema.Metric(
            name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
            value=42.0,
            unit=schema.UNIT_INFERENCES_PER_SECOND,
        )

        metrics = schema.ensure_standard_performance_metrics([throughput])
        by_name = {metric.name: metric for metric in metrics}

        assert set(by_name) == {
            definition.name for definition in schema.STANDARD_PERFORMANCE_METRICS
        }
        assert by_name[schema.METRIC_NAME_INFERENCES_PER_SECOND].value == 42.0
        model_weight_memory = by_name[schema.METRIC_NAME_MODEL_WEIGHT_MEMORY]
        assert model_weight_memory.value is None
        assert model_weight_memory.availability == schema.MetricAvailability.UNAVAILABLE
        assert model_weight_memory.unit == schema.UNIT_BYTES
        assert (
            model_weight_memory.reason == "Model weight memory data is not available."
        )
        unavailable_metric = by_name[schema.METRIC_NAME_CPU_UTILIZATION]
        assert unavailable_metric.value is None
        assert unavailable_metric.availability == schema.MetricAvailability.UNAVAILABLE
        assert unavailable_metric.unit == schema.UNIT_PERCENT
        assert unavailable_metric.reason
        inference_time = by_name[schema.METRIC_NAME_INFERENCE_TIME]
        assert inference_time.value is None
        assert inference_time.availability == schema.MetricAvailability.UNAVAILABLE
        assert inference_time.unit == schema.UNIT_MILLISECONDS
        assert inference_time.reason == "Inference latency data is not available."

    def test_ensure_standard_performance_metrics_does_not_mutate_input(
        self,
    ) -> None:
        """Helper should return a new list without mutating caller state."""
        metrics = [
            schema.Metric(
                name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
                value=42.0,
                unit=schema.UNIT_INFERENCES_PER_SECOND,
            )
        ]

        filled_metrics = schema.ensure_standard_performance_metrics(metrics)

        assert filled_metrics is not metrics
        assert len(metrics) == 1
        assert len(filled_metrics) == len(schema.STANDARD_PERFORMANCE_METRICS)

    def test_ensure_standard_performance_metrics_rejects_mismatched_unit(
        self,
    ) -> None:
        """Standard metrics should use the unit defined by the shared contract."""
        metric = schema.Metric(
            name=schema.METRIC_NAME_INFERENCE_TIME,
            value=12.5,
            unit="seconds",
        )

        with pytest.raises(
            ValueError,
            match=(
                "Standard performance metric 'inference_time' must use unit "
                "'ms', got 'seconds'."
            ),
        ):
            schema.ensure_standard_performance_metrics([metric])


class TestBreakdown:
    """Test Breakdown class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        breakdown = schema.Breakdown(
            entity_id="entity_001",
            metrics=[
                schema.Metric(name="cycles", value=1000, unit="cycles"),
                schema.Metric(name="energy", value=50.5, unit="mJ"),
            ],
            id="breakdown_001",
            qualifiers={"device": "npu", "precision": "int8"},
        )
        result = breakdown.to_dict()
        assert result["entity_id"] == "entity_001"
        assert len(result["metrics"]) == 2
        assert result["id"] == "breakdown_001"
        assert result["qualifiers"] == {"device": "npu", "precision": "int8"}
        assert "scope" not in result
        assert "name" not in result
        assert "locations" not in result
        assert "subgraph_kind" not in result

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "entity_id": "entity_001",
            "metrics": [
                {"name": "cycles", "value": 1000, "unit": "cycles"},
                {"name": "energy", "value": 50.5, "unit": "mJ"},
            ],
            "id": "breakdown_001",
            "qualifiers": {"device": "npu", "precision": "int8"},
        }
        breakdown = schema.Breakdown.from_dict(data)
        assert breakdown.entity_id == "entity_001"
        assert len(breakdown.metrics) == 2
        assert breakdown.id == "breakdown_001"
        assert breakdown.qualifiers == {"device": "npu", "precision": "int8"}


class TestCheck:
    """Test Check class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        check = schema.Check(
            id="compatibility_check",
            status=schema.CheckStatus.PASS,
            entity_id="entity_001",
            details={"message": "All operators supported", "count": 42},
        )
        result = check.to_dict()
        assert result["id"] == "compatibility_check"
        assert result["status"] == "pass"
        assert result["entity_id"] == "entity_001"
        assert result["details"] == {"message": "All operators supported", "count": 42}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "compatibility_check",
            "status": "pass",
            "entity_id": "entity_001",
            "details": {"message": "All operators supported", "count": 42},
        }
        check = schema.Check.from_dict(data)
        assert check.id == "compatibility_check"
        assert check.status == schema.CheckStatus.PASS
        assert check.entity_id == "entity_001"
        assert check.details == {"message": "All operators supported", "count": 42}


class TestEntityKind:
    """Test EntityKind class."""

    def test_well_known_entity_kinds(self) -> None:
        """Well-known entity kinds should include schema-defined kind ids."""
        assert schema.ENTITY_KIND_CODE_STACK == "code_stack"
        assert schema.ENTITY_KIND_CODE_LINE == "code_line"
        assert schema.WELL_KNOWN_ENTITY_KINDS == frozenset(
            {
                schema.ENTITY_KIND_SOURCE_OPERATOR,
                schema.ENTITY_KIND_MODEL,
                schema.ENTITY_KIND_CODE_STACK,
                schema.ENTITY_KIND_CODE_LINE,
            }
        )

    def test_well_known_entity_kind_definitions(self) -> None:
        """Well-known hierarchy relationships should be centrally defined."""
        assert schema.WELL_KNOWN_ENTITY_KIND_DEFINITIONS == {
            schema.ENTITY_KIND_CODE_LINE: schema.WellKnownEntityKind(
                id=schema.ENTITY_KIND_CODE_LINE,
                child_kinds=(schema.ENTITY_KIND_CODE_STACK,),
            ),
            schema.ENTITY_KIND_CODE_STACK: schema.WellKnownEntityKind(
                id=schema.ENTITY_KIND_CODE_STACK,
                parent_kinds=(
                    schema.ENTITY_KIND_CODE_LINE,
                    schema.ENTITY_KIND_CODE_STACK,
                ),
                child_kinds=(
                    schema.ENTITY_KIND_CODE_STACK,
                    schema.ENTITY_KIND_SOURCE_OPERATOR,
                ),
            ),
        }

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        entity_kind = schema.EntityKind(
            id="nn_module",
            parent_kinds=["nn_module"],
            child_kinds=["nn_module", "source_operator"],
        )

        assert entity_kind.to_dict() == {
            "id": "nn_module",
            "parent_kinds": ["nn_module"],
            "child_kinds": ["nn_module", "source_operator"],
        }

    def test_to_dict_omits_empty_relationships(self) -> None:
        """Empty parent and child kind lists should be omitted."""
        assert schema.EntityKind(id="segment").to_dict() == {"id": "segment"}

    def test_to_dict_copies_relationship_lists(self) -> None:
        """Dictionary callers should not mutate the entity kind instance."""
        entity_kind = schema.EntityKind(id="segment", child_kinds=["cascade"])
        result = entity_kind.to_dict()

        result["child_kinds"].append("chain")

        assert entity_kind.child_kinds == ["cascade"]

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        entity_kind = schema.EntityKind.from_dict(
            {
                "id": "chain",
                "parent_kinds": ["cascade"],
                "child_kinds": ["source_operator"],
            }
        )

        assert entity_kind.id == "chain"
        assert entity_kind.parent_kinds == ["cascade"]
        assert entity_kind.child_kinds == ["source_operator"]


class TestOnnxSourceOperatorId:
    """Test canonical ONNX source-operator identity construction."""

    def test_uses_zero_based_top_level_node_index(self) -> None:
        """ONNX identity is independent of any presentation name."""
        assert schema.onnx_source_operator_id(0) == "source_operator/0"
        assert schema.onnx_source_operator_id(17) == "source_operator/17"

    @pytest.mark.parametrize("node_index", [-1, True, 1.5, "1"])
    def test_rejects_invalid_node_index(self, node_index: object) -> None:
        """Only non-negative integer top-level node positions are valid."""
        with pytest.raises(ValueError, match="non-negative integer"):
            schema.onnx_source_operator_id(node_index)  # type: ignore[arg-type]


class TestTosaSourceOperatorId:
    """Test canonical direct-TOSA source-operator identity construction."""

    def test_uses_parser_numeric_operator_id(self) -> None:
        """TOSA identity uses the parser's exact numeric operation identity."""
        assert schema.tosa_source_operator_id(0) == "source_operator/0"
        assert schema.tosa_source_operator_id(97) == "source_operator/97"

    @pytest.mark.parametrize("operator_id", [-1, True, 1.5, "1", None])
    def test_rejects_invalid_operator_id(self, operator_id: object) -> None:
        """Only exact non-negative integer operation identities are valid."""
        with pytest.raises(ValueError, match="non-negative integer"):
            schema.tosa_source_operator_id(operator_id)  # type: ignore[arg-type]


class TestPt2SourceOperatorId:
    """Test canonical PT2 source-operator identity construction."""

    def test_uses_top_level_fx_node_name(self) -> None:
        """PT2 identity uses the exact operation node name from the export."""
        assert schema.pt2_source_operator_id("conv2d") == "source_operator/conv2d"
        assert schema.pt2_source_operator_id("conv2d_1") == "source_operator/conv2d_1"

    @pytest.mark.parametrize("node_name", ["", "   ", True, 1, None])
    def test_rejects_invalid_node_name(self, node_name: object) -> None:
        """Only non-empty FX node-name strings are valid."""
        with pytest.raises(ValueError, match="non-empty string"):
            schema.pt2_source_operator_id(node_name)  # type: ignore[arg-type]


class TestVgfSourceOperatorId:
    """Test canonical VGF source-operator identity construction."""

    def test_uses_segment_and_spirv_result_id(self) -> None:
        """VGF identity uses structural numeric coordinates only."""
        assert schema.vgf_source_operator_id(0, 437) == (
            "source_operator/segment_0/spirv-437"
        )
        assert schema.vgf_source_operator_id(7, 0) == (
            "source_operator/segment_7/spirv-0"
        )

    @pytest.mark.parametrize(
        ("segment_index", "result_id"),
        [(-1, 0), (True, 0), ("0", 0), (0, -1), (0, True), (0, "1")],
    )
    def test_rejects_invalid_coordinates(
        self, segment_index: object, result_id: object
    ) -> None:
        """VGF coordinates must be non-negative integers."""
        with pytest.raises(ValueError, match="non-negative integer"):
            schema.vgf_source_operator_id(
                cast(int, segment_index), cast(int, result_id)
            )


class TestEntity:
    """Test Entity class."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        entity = schema.Entity(
            id="chain_001",
            kind="chain",
            name="chain_0",
            placement="npu",
            parent_ids=["model", "module_conv"],
            child_ids=["layer_0", "layer_1"],
            attributes={"dtype": "int8", "kernel_size": [3, 3]},
            stack_trace="forward > conv2d",
        )
        result = entity.to_dict()
        assert result["id"] == "chain_001"
        assert result["kind"] == "chain"
        assert result["name"] == "chain_0"
        assert "locations" not in result
        assert result["placement"] == "npu"
        assert result["parent_ids"] == ["model", "module_conv"]
        assert result["child_ids"] == ["layer_0", "layer_1"]
        assert result["attributes"] == {"dtype": "int8", "kernel_size": [3, 3]}
        assert result["stack_trace"] == "forward > conv2d"

    def test_to_dict_omits_optional_empty_values(self) -> None:
        """Test optional entity fields are omitted when empty."""
        entity = schema.Entity(
            id="entity_001",
            kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
            name="CONV_2D",
        )

        assert entity.to_dict() == {
            "id": "entity_001",
            "kind": "source_operator",
            "name": "CONV_2D",
        }

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "chain_001",
            "kind": "chain",
            "name": "chain_0",
            "placement": "npu",
            "parent_ids": ["model", "module_conv"],
            "child_ids": ["layer_0", "layer_1"],
            "attributes": {"dtype": "int8", "kernel_size": [3, 3]},
            "stack_trace": "forward > conv2d",
        }
        entity = schema.Entity.from_dict(data)
        assert entity.id == "chain_001"
        assert entity.kind == "chain"
        assert entity.name == "chain_0"
        assert not hasattr(entity, "locations")
        assert entity.placement == "npu"
        assert entity.parent_ids == ["model", "module_conv"]
        assert entity.child_ids == ["layer_0", "layer_1"]
        assert entity.attributes == {"dtype": "int8", "kernel_size": [3, 3]}
        assert entity.stack_trace == "forward > conv2d"

    @pytest.mark.parametrize(
        "legacy_field", ["locations", "parent_id", "scope", "subgraph_kind"]
    )
    def test_from_dict_rejects_legacy_entity_fields(self, legacy_field: str) -> None:
        """Legacy entity identity fields are not accepted by schema model parsing."""
        with pytest.raises(ValueError, match=legacy_field):
            schema.Entity.from_dict(
                {
                    "id": "chain_001",
                    "kind": "chain",
                    "name": "chain_0",
                    legacy_field: "legacy_value",
                }
            )


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

    def test_to_dict_with_entity_kinds(self) -> None:
        """Test conversion to dictionary with entity kind metadata."""
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer="backend",
            entity_kinds=[
                schema.EntityKind(id="segment", child_kinds=["cascade"]),
                schema.EntityKind(
                    id="nn_module",
                    parent_kinds=["nn_module"],
                    child_kinds=["nn_module", "source_operator"],
                ),
            ],
        )

        assert result.to_dict()["entity_kinds"] == [
            {"id": "segment", "child_kinds": ["cascade"]},
            {
                "id": "nn_module",
                "parent_kinds": ["nn_module"],
                "child_kinds": ["nn_module", "source_operator"],
            },
        ]

    def test_from_dict_with_entity_kinds(self) -> None:
        """Test creation from dictionary with entity kind metadata."""
        result = schema.Result.from_dict(
            {
                "kind": "performance",
                "status": "ok",
                "producer": "backend",
                "entity_kinds": [
                    {"id": "segment", "child_kinds": ["cascade"]},
                    {
                        "id": "chain",
                        "parent_kinds": ["cascade"],
                        "child_kinds": ["source_operator"],
                    },
                ],
            }
        )

        assert result.entity_kinds == [
            schema.EntityKind(id="segment", child_kinds=["cascade"]),
            schema.EntityKind(
                id="chain",
                parent_kinds=["cascade"],
                child_kinds=["source_operator"],
            ),
        ]

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

    def test_to_dict_with_advice(self) -> None:
        """Test conversion to dictionary with advice."""
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer="backend",
            advice=[
                schema.Advice(
                    id="0",
                    category=schema.AdviceCategory.PERFORMANCE,
                    severity=schema.AdviceSeverity.INFO,
                    message="Review the performance metrics.",
                    affected_entity_ids=["entity_001"],
                    details={"reason": "example"},
                )
            ],
        )

        result_dict = result.to_dict()

        assert "advice" in result_dict
        assert "advices" not in result_dict
        assert result_dict["advice"] == [
            {
                "id": "0",
                "category": "performance",
                "severity": "info",
                "message": "Review the performance metrics.",
                "affected_entity_ids": ["entity_001"],
                "details": {"reason": "example"},
            }
        ]

    def test_from_dict_with_advice(self) -> None:
        """Test creation from dictionary with advice."""
        result = schema.Result.from_dict(
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
                        "affected_entity_ids": ["entity_001"],
                    }
                ],
            }
        )

        assert len(result.advice) == 1
        assert result.advice[0].id == "0"
        assert result.advice[0].category == schema.AdviceCategory.PERFORMANCE
        assert result.advice[0].severity == schema.AdviceSeverity.INFO
        assert result.advice[0].affected_entity_ids == ["entity_001"]

    def test_from_dict_does_not_parse_legacy_advices_alias(self) -> None:
        """Test creation from dictionary ignores the legacy advices field."""
        result = schema.Result.from_dict(
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

        assert result.advice == []


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
