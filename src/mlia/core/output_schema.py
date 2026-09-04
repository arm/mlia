# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Standardized output schema classes for MLIA."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

# Schema version for standardized output
SCHEMA_VERSION = "1.2.0"

# Target schema version
TARGET_SCHEMA_VERSION = "1.0.0"


class ComponentType(str, Enum):
    """Component type enumeration."""

    CPU = "cpu"
    NPU = "npu"
    GPU = "gpu"
    DSP = "dsp"
    SPECIFICATION = "specification"
    SOC = "soc"


class ResultKind(str, Enum):
    """Result kind enumeration."""

    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"


class ResultStatus(str, Enum):
    """Result status enumeration."""

    OK = "ok"
    PARTIAL = "partial"
    INCOMPATIBLE = "incompatible"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModeType(str, Enum):
    """Mode type enumeration."""

    MEASURED = "measured"
    SIMULATED = "simulated"
    PREDICTED = "predicted"


class PlacementType(str, Enum):
    """Placement type enumeration."""

    NPU = "NPU"
    NX = "NX"
    CPU = "CPU"
    GPU = "GPU"
    DSP = "DSP"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class WellKnownEntityKind:
    """Schema-defined semantic relationships for a well-known entity kind."""

    id: str  # pylint: disable=invalid-name
    parent_kinds: tuple[str, ...] = ()
    child_kinds: tuple[str, ...] = ()


ENTITY_KIND_SOURCE_OPERATOR = "source_operator"
ENTITY_KIND_MODEL = "model"
ENTITY_KIND_CODE_STACK = "code_stack"
ENTITY_KIND_CODE_LINE = "code_line"


def tflite_source_operator_id(operator_index: int, subgraph_index: int = 0) -> str:
    """Return the canonical identity for a TFLite source operation.

    ``operator_index`` is the zero-based position in the containing TFLite
    subgraph's operator list. The main graph at index zero omits its subgraph
    coordinate; operations in nested subgraphs include both structural indexes.
    """
    if type(operator_index) is not int or operator_index < 0:
        raise ValueError("TFLite operator index must be a non-negative integer.")
    if type(subgraph_index) is not int or subgraph_index < 0:
        raise ValueError("TFLite subgraph index must be a non-negative integer.")
    if subgraph_index == 0:
        return f"{ENTITY_KIND_SOURCE_OPERATOR}/operator/{operator_index}"
    return (
        f"{ENTITY_KIND_SOURCE_OPERATOR}/subgraph/{subgraph_index}"
        f"/operator/{operator_index}"
    )


def onnx_source_operator_id(node_index: int) -> str:
    """Return the canonical identity for a top-level ONNX graph node.

    ``node_index`` is the zero-based position in ``ModelProto.graph.node``. ONNX
    node names are presentation metadata and never contribute to identity.
    Operations inside ONNX subgraphs and functions are not currently supported;
    they require a future graph-qualified identity format.
    """
    if type(node_index) is not int or node_index < 0:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("ONNX source node index must be a non-negative integer.")
    return f"{ENTITY_KIND_SOURCE_OPERATOR}/{node_index}"


def tosa_source_operator_id(operator_id: int) -> str:
    """Return the canonical identity for one directly checked TOSA operation.

    ``operator_id`` is the parser-provided numeric operation identity. For a TOSA
    flatbuffer it is assigned by deterministic region/block/operator traversal;
    for TOSA-MLIR it is the numeric operation/result ID parsed from the SSA result.
    Location and debug metadata never contribute to identity. The identity is
    scoped to the containing result and is not guaranteed to remain stable across
    separate flatbuffer serializations or TOSA-MLIR conversions.
    """
    if type(operator_id) is not int or operator_id < 0:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("TOSA source operator ID must be a non-negative integer.")
    return f"{ENTITY_KIND_SOURCE_OPERATOR}/{operator_id}"


def pt2_source_operator_id(node_name: str) -> str:
    """Return the canonical identity for a top-level PT2 operation node.

    ``node_name`` is the exact unique ``torch.fx.Node.name`` in the original
    top-level ``torch.export.ExportedProgram`` graph. The identity is scoped to
    the containing result and does not need to remain stable across separate
    exports. Placeholder, ``get_attr``, and output nodes are not source
    operations. Operations inside nested graphs are not currently supported;
    they require a future graph-qualified identity format.
    """
    if not isinstance(node_name, str) or not node_name.strip():
        raise ValueError("PT2 source node name must be a non-empty string.")
    return f"{ENTITY_KIND_SOURCE_OPERATOR}/{node_name}"


def vgf_source_operator_id(segment_index: int, spirv_result_id: int) -> str:
    """Return the canonical identity for one VGF SPIR-V operation.

    ``segment_index`` is the zero-based VGF model-sequence segment index and
    ``spirv_result_id`` is the numeric SPIR-V instruction result ID within that
    segment. Debug names and API labels are presentation/provenance metadata and
    never contribute to this identity.
    """
    if type(segment_index) is not int or segment_index < 0:
        raise ValueError("VGF segment index must be a non-negative integer.")
    if type(spirv_result_id) is not int or spirv_result_id < 0:
        raise ValueError("SPIR-V result ID must be a non-negative integer.")
    return (
        f"{ENTITY_KIND_SOURCE_OPERATOR}/segment_{segment_index}/spirv-{spirv_result_id}"
    )


# Entity kind ids with schema-defined semantics do not require matching
# result-level entity_kinds declarations. Backends may produce source_operator,
# model, and code_stack entities; code_line is derived centrally by core output
# postprocessing. Every other entity kind id used by an entity is backend-defined
# and must be declared in result.entity_kinds, even if it has no relationships.
#
# A code_stack entity represents one source/debug stack-frame prefix rather than
# a globally unique function/frame. Its identity includes all ancestor frames
# plus the current frame, so the same frame reached through different callers is
# represented by distinct code_stack entities. Parent/child links reconstruct
# the call tree, and leaf code_stack entities may parent source_operator
# entities associated with the stack.
#
# A code_line entity represents one exact source/debug file and 1-based line
# number within a result, independent of call-stack ancestry. Core derives these
# entities from retained code_stack entities during output postprocessing.
#
# Well-known code_stack and code_line attributes:
# - file: source/debug file path using forward slash as the separator. The path
#   may be absolute or relative.
# - line: 1-based index into the lines of file.
WELL_KNOWN_ENTITY_KIND_DEFINITIONS: Mapping[str, WellKnownEntityKind] = (
    MappingProxyType(
        {
            ENTITY_KIND_CODE_LINE: WellKnownEntityKind(
                id=ENTITY_KIND_CODE_LINE,
                child_kinds=(ENTITY_KIND_CODE_STACK,),
            ),
            ENTITY_KIND_CODE_STACK: WellKnownEntityKind(
                id=ENTITY_KIND_CODE_STACK,
                parent_kinds=(ENTITY_KIND_CODE_LINE, ENTITY_KIND_CODE_STACK),
                child_kinds=(ENTITY_KIND_CODE_STACK, ENTITY_KIND_SOURCE_OPERATOR),
            ),
        }
    )
)

WELL_KNOWN_ENTITY_KINDS = frozenset(
    {
        ENTITY_KIND_SOURCE_OPERATOR,
        ENTITY_KIND_MODEL,
        *WELL_KNOWN_ENTITY_KIND_DEFINITIONS,
    }
)


class CheckStatus(str, Enum):
    """Check status enumeration."""

    PASS = "pass"  # nosec B105
    FAIL = "fail"
    PARTIAL = "partial"


class AdviceCategory(str, Enum):
    """Advice category enumeration."""

    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    OPTIMIZATION = "optimization"


class AdviceSeverity(str, Enum):
    """Advice severity enumeration."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MetricAvailability(str, Enum):
    """Metric availability enumeration."""

    UNAVAILABLE = "unavailable"


class AggregationType(str, Enum):
    """Well-known metric aggregation types.

    Metric producers may still use other string values for forward-compatible
    aggregation policies that are not yet defined by this schema version.
    """

    SUM = "sum"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"


@dataclass(frozen=True)
class StandardPerformanceMetric:
    """Definition for a standardized performance metric."""

    name: str
    unit: str
    unavailable_reason: str


METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE = "accelerator_operator_percentage"
METRIC_NAME_INFERENCES_PER_SECOND = "inferences_per_second"
METRIC_NAME_CPU_UTILIZATION = "cpu_utilization"
METRIC_NAME_TARGET_UTILIZATION = "target_utilization"
METRIC_NAME_INFERENCE_TIME = "inference_time"
METRIC_NAME_MODEL_WEIGHT_MEMORY = "model_weight_memory"
METRIC_NAME_PEAK_ACTIVATION_MEMORY = "peak_activation_memory"
METRIC_NAME_AVERAGE_MEMORY = "average_memory"

UNIT_PERCENT = "%"
UNIT_INFERENCES_PER_SECOND = "inferences/s"
UNIT_MILLISECONDS = "ms"
UNIT_BYTES = "bytes"

STANDARD_PERFORMANCE_METRICS = (
    StandardPerformanceMetric(
        name=METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        unit=UNIT_PERCENT,
        unavailable_reason="Accelerator operator placement data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_INFERENCES_PER_SECOND,
        unit=UNIT_INFERENCES_PER_SECOND,
        unavailable_reason="Inference throughput data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_CPU_UTILIZATION,
        unit=UNIT_PERCENT,
        unavailable_reason="CPU utilization data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_TARGET_UTILIZATION,
        unit=UNIT_PERCENT,
        unavailable_reason="Target utilization data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_INFERENCE_TIME,
        unit=UNIT_MILLISECONDS,
        unavailable_reason="Inference latency data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_MODEL_WEIGHT_MEMORY,
        unit=UNIT_BYTES,
        unavailable_reason="Model weight memory data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_PEAK_ACTIVATION_MEMORY,
        unit=UNIT_BYTES,
        unavailable_reason="Peak activation memory data is not available.",
    ),
    StandardPerformanceMetric(
        name=METRIC_NAME_AVERAGE_MEMORY,
        unit=UNIT_BYTES,
        unavailable_reason="Average memory data is not available.",
    ),
)


@dataclass(frozen=True)
class Tool:
    """Tool information."""

    name: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tool:
        """Create from dictionary."""
        return cls(name=data["name"], version=data["version"])


@dataclass(frozen=True)
class Backend:
    """Backend information."""

    id: str  # pylint: disable=invalid-name
    name: str
    version: str
    configuration: dict[str, Any]
    impl: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "configuration": self.configuration,
        }
        if self.impl is not None:
            result["impl"] = self.impl
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Backend:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            configuration=data["configuration"],
            impl=data.get("impl"),
        )


@dataclass(frozen=True)
class Component:
    """Component information."""

    type: ComponentType
    family: str
    model: str | None = None
    variant: str | None = None
    name: str | None = None
    components: list[Component] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"type": self.type.value, "family": self.family}
        if self.model is not None:
            result["model"] = self.model
        if self.variant is not None:
            result["variant"] = self.variant
        if self.name is not None:
            result["name"] = self.name
        if self.components:
            result["components"] = [comp.to_dict() for comp in self.components]  # type: ignore[assignment]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Component:
        """Create from dictionary."""
        return cls(
            type=ComponentType(data["type"]),
            family=data["family"],
            model=data.get("model"),
            variant=data.get("variant"),
            name=data.get("name"),
            components=[cls.from_dict(c) for c in data.get("components", [])],
        )


@dataclass(frozen=True)
class Target:
    """Target information."""

    profile_name: str
    target_type: str
    components: list[Component]
    configuration: dict[str, Any]
    description: str | None = None
    host_platform: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "profile_name": self.profile_name,
            "target_type": self.target_type,
            "components": [c.to_dict() for c in self.components],
            "configuration": self.configuration,
        }
        if self.description is not None:
            result["description"] = self.description
        if self.host_platform is not None:
            result["host_platform"] = self.host_platform
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Target:
        """Create from dictionary."""
        return cls(
            profile_name=data["profile_name"],
            target_type=data["target_type"],
            components=[Component.from_dict(c) for c in data["components"]],
            configuration=data["configuration"],
            description=data.get("description"),
            host_platform=data.get("host_platform"),
        )


@dataclass(frozen=True)
class Model:
    """Model information."""

    name: str
    format: str
    hash: str
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name, "format": self.format, "hash": self.hash}
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes  # type: ignore[assignment]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Model:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            format=data["format"],
            hash=data["hash"],
            size_bytes=data.get("size_bytes"),
        )


@dataclass(frozen=True)
class Context:
    """Context information."""

    runtime_configuration: dict[str, Any] | None = None
    git: dict[str, Any] | None = None
    notes: str | None = None
    cli_arguments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {}
        if self.runtime_configuration is not None:
            result["runtime_configuration"] = self.runtime_configuration
        if self.git is not None:
            result["git"] = self.git
        if self.notes is not None:
            result["notes"] = self.notes
        if self.cli_arguments:
            result["cli_arguments"] = self.cli_arguments
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Create from dictionary."""
        return cls(
            runtime_configuration=data.get("runtime_configuration"),
            git=data.get("git"),
            notes=data.get("notes"),
            cli_arguments=data.get("cli_arguments", []),
        )


@dataclass(frozen=True)
class Metric:
    """Metric information."""

    name: str
    value: float | int | None
    unit: str
    aggregation: AggregationType | str | None = None
    samples: int | None = None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    availability: MetricAvailability | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate that the metric matches one of the supported schema shapes."""
        if self.value is not None:
            # JSON numbers cannot represent NaN or infinities. Reject them at the
            # schema boundary so producers use an unavailable metric instead of
            # creating output that permissive Python encoders write as invalid JSON.
            if not math.isfinite(self.value):
                raise ValueError("Metric value must be finite.")
            if self.availability is not None or self.reason is not None:
                raise ValueError(
                    "Metric cannot combine a numeric value with availability fields."
                )
            return

        if self.availability is None:
            raise ValueError("Metric must include a value or availability.")

        if not self.reason:
            raise ValueError("Unavailable metrics must include a reason.")

        if self.aggregation is not None or self.samples is not None:
            raise ValueError(
                "Unavailable metrics cannot include aggregation or samples."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"name": self.name, "unit": self.unit}
        if self.value is not None:
            result["value"] = self.value
        if self.aggregation is not None:
            result["aggregation"] = (
                self.aggregation.value
                if isinstance(self.aggregation, AggregationType)
                else self.aggregation
            )
        if self.samples is not None:
            result["samples"] = self.samples
        if self.qualifiers:
            result["qualifiers"] = self.qualifiers
        if self.availability is not None:
            result["availability"] = self.availability.value
        if self.reason is not None:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        """Create from dictionary."""
        aggregation = data.get("aggregation")
        if aggregation is not None:
            try:
                aggregation = AggregationType(aggregation)
            except ValueError:
                # Unknown strings remain valid for forward compatibility.
                pass
        return cls(
            name=data["name"],
            value=data.get("value"),
            unit=data["unit"],
            aggregation=aggregation,
            samples=data.get("samples"),
            qualifiers=data.get("qualifiers", {}),
            availability=(
                MetricAvailability(data["availability"])
                if "availability" in data
                else None
            ),
            reason=data.get("reason"),
        )


def ensure_standard_performance_metrics(metrics: list[Metric]) -> list[Metric]:
    """Validate and add unavailable entries for standard performance metrics.

    Plugin performance collectors should add the standard metrics they can
    report, then call this helper to include unavailable entries for the
    remaining standard metrics.
    """
    definitions_by_name = {
        definition.name: definition for definition in STANDARD_PERFORMANCE_METRICS
    }
    for metric in metrics:
        if definition := definitions_by_name.get(metric.name):
            if metric.unit != definition.unit:
                raise ValueError(
                    f"Standard performance metric '{metric.name}' must use unit "
                    f"'{definition.unit}', got '{metric.unit}'."
                )

    existing_names = {metric.name for metric in metrics}
    missing_metrics = [
        Metric(
            name=definition.name,
            value=None,
            unit=definition.unit,
            availability=MetricAvailability.UNAVAILABLE,
            reason=definition.unavailable_reason,
        )
        for definition in STANDARD_PERFORMANCE_METRICS
        if definition.name not in existing_names
    ]
    return [*metrics, *missing_metrics]


@dataclass(frozen=True)
class Breakdown:
    """Breakdown metrics for an entity."""

    entity_id: str
    metrics: list[Metric]
    id: str | None = None  # pylint: disable=invalid-name
    qualifiers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "entity_id": self.entity_id,
            "metrics": [m.to_dict() for m in self.metrics],
        }
        if self.id is not None:
            result["id"] = self.id
        if self.qualifiers:
            result["qualifiers"] = self.qualifiers
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Breakdown:
        """Create from dictionary."""
        return cls(
            entity_id=data["entity_id"],
            metrics=[Metric.from_dict(m) for m in data["metrics"]],
            id=data.get("id"),
            qualifiers=data.get("qualifiers", {}),
        )


@dataclass(frozen=True)
class Check:
    """Check information."""

    id: str  # pylint: disable=invalid-name
    status: CheckStatus
    entity_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"id": self.id, "status": self.status.value}
        if self.entity_id is not None:
            result["entity_id"] = self.entity_id
        if self.details:
            result["details"] = self.details  # type: ignore[assignment]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Check:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            status=CheckStatus(data["status"]),
            entity_id=data.get("entity_id"),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class EntityKind:
    """Semantic relationship metadata for an entity kind."""

    id: str  # pylint: disable=invalid-name
    parent_kinds: list[str] = field(default_factory=list)
    child_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"id": self.id}
        if self.parent_kinds:
            result["parent_kinds"] = list(self.parent_kinds)
        if self.child_kinds:
            result["child_kinds"] = list(self.child_kinds)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityKind:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            parent_kinds=list(data.get("parent_kinds", [])),
            child_kinds=list(data.get("child_kinds", [])),
        )


@dataclass(frozen=True)
class Entity:
    """Entity information.

    Entity ids are unique within their containing result. A source_operator id
    is the complete canonical source-operation identifier and is opaque to
    generic consumers. For TFLite, the canonical id is
    ``source_operator/operator/<operator_index>`` for the main graph or
    ``source_operator/subgraph/<subgraph_index>/operator/<operator_index>`` for
    a nested subgraph. For ONNX, the canonical id is
    ``source_operator/<node_index>``, where ``node_index`` is the zero-based
    position in the top-level ``ModelProto.graph.node`` list. ONNX node names
    are presentation-only and never identity. For a PyTorch ExportedProgram in
    PT2 format, the canonical id is ``source_operator/<fx_node_name>``, using the
    exact unique ``torch.fx.Node.name`` of an operation in the original top-level
    exported graph. For VGF, the canonical id is
    ``source_operator/segment_<segment_index>/spirv-<result_id>``, using the
    zero-based model-sequence segment index and numeric SPIR-V instruction result
    ID. Debug names and API labels never contribute to identity. ONNX subgraphs,
    ONNX functions, and PT2 nested graphs are not currently supported and require
    a future graph-qualified identity.
    """

    id: str  # pylint: disable=invalid-name
    kind: str
    name: str
    placement: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
        }
        if self.placement is not None:
            result["placement"] = self.placement
        if self.parent_ids:
            result["parent_ids"] = self.parent_ids
        if self.child_ids:
            result["child_ids"] = self.child_ids
        if self.attributes:
            result["attributes"] = self.attributes
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create from dictionary."""
        if "locations" in data:
            raise ValueError("Entity uses unsupported field 'locations'.")
        if "parent_id" in data:
            raise ValueError("Entity uses unsupported legacy field 'parent_id'.")
        if "scope" in data:
            raise ValueError("Entity uses unsupported legacy field 'scope'.")
        if "subgraph_kind" in data:
            raise ValueError("Entity uses unsupported legacy field 'subgraph_kind'.")

        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            placement=data.get("placement"),
            parent_ids=list(data.get("parent_ids", [])),
            child_ids=list(data.get("child_ids", [])),
            attributes=data.get("attributes", {}),
            stack_trace=data.get("stack_trace", ""),
        )


@dataclass(frozen=True)
class Advice:
    """Advice information."""

    id: str  # pylint: disable=invalid-name
    category: AdviceCategory
    severity: AdviceSeverity
    message: str
    affected_entity_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.affected_entity_ids:
            result["affected_entity_ids"] = self.affected_entity_ids
        if self.details:
            result["details"] = self.details
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Advice:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=AdviceCategory(data["category"]),
            severity=AdviceSeverity(data["severity"]),
            message=data["message"],
            affected_entity_ids=list(data.get("affected_entity_ids", [])),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class Result:  # pylint: disable=too-many-instance-attributes
    """Result information."""

    kind: ResultKind
    status: ResultStatus
    producer: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    breakdowns: list[Breakdown] = field(default_factory=list)
    mode: ModeType | None = None
    checks: list[Check] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    entity_kinds: list[EntityKind] = field(default_factory=list)
    advice: list[Advice] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "status": self.status.value,
            "producer": self.producer,
        }
        if self.warnings:
            result["warnings"] = self.warnings
        if self.errors:
            result["errors"] = self.errors
        if self.metrics:
            result["metrics"] = [m.to_dict() for m in self.metrics]
        if self.breakdowns:
            result["breakdowns"] = [b.to_dict() for b in self.breakdowns]
        if self.mode is not None:
            result["mode"] = self.mode.value
        if self.checks:
            result["checks"] = [c.to_dict() for c in self.checks]
        if self.entities:
            result["entities"] = [e.to_dict() for e in self.entities]
        if self.entity_kinds:
            result["entity_kinds"] = [k.to_dict() for k in self.entity_kinds]
        if self.advice:
            result["advice"] = [a.to_dict() for a in self.advice]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Result:
        """Create from dictionary."""
        return cls(
            kind=ResultKind(data["kind"]),
            status=ResultStatus(data["status"]),
            producer=data["producer"],
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            metrics=[Metric.from_dict(m) for m in data.get("metrics", [])],
            breakdowns=[Breakdown.from_dict(b) for b in data.get("breakdowns", [])],
            mode=ModeType(data["mode"]) if "mode" in data else None,
            checks=[Check.from_dict(c) for c in data.get("checks", [])],
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            entity_kinds=[
                EntityKind.from_dict(k) for k in data.get("entity_kinds", [])
            ],
            advice=[Advice.from_dict(a) for a in data.get("advice", [])],
        )


@dataclass(frozen=True)
class StandardizedOutput:
    """Main standardized output structure for MLIA."""

    schema_version: str
    run_id: str
    timestamp: str
    tool: Tool
    target: Target
    model: Model
    context: Context
    backends: list[Backend]
    results: list[Result]
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "tool": self.tool.to_dict(),
            "target": self.target.to_dict(),
            "model": self.model.to_dict(),
            "context": self.context.to_dict(),
            "backends": [b.to_dict() for b in self.backends],
            "results": [r.to_dict() for r in self.results],
        }
        if self.extensions:
            result["extensions"] = self.extensions
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardizedOutput:
        """Create from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            tool=Tool.from_dict(data["tool"]),
            target=Target.from_dict(data["target"]),
            model=Model.from_dict(data["model"]),
            context=Context.from_dict(data["context"]),
            backends=[Backend.from_dict(b) for b in data["backends"]],
            results=[Result.from_dict(r) for r in data["results"]],
            extensions=data.get("extensions", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> StandardizedOutput:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Path | str) -> None:
        """Save to JSON file."""
        path = Path(filepath)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Path | str) -> StandardizedOutput:
        """Load from JSON file."""
        path = Path(filepath)
        return cls.from_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def create_timestamp() -> str:
        """Create ISO 8601 timestamp for current time."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def create_run_id() -> str:
        """Create a new UUID for run_id."""
        return str(uuid4())
