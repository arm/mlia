# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Standardized output schema classes for MLIA."""
from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

# Schema version for standardized output
SCHEMA_VERSION = "1.0.0"

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


class OperatorScope(str, Enum):
    """Operator scope enumeration."""

    OPERATOR = "operator"
    OPERATOR_CHAIN = "operator_chain"


class CheckStatus(str, Enum):
    """Check status enumeration."""

    PASS = "pass"  # nosec B105
    FAIL = "fail"
    PARTIAL = "partial"


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
            result["components"] = [
                comp.to_dict() for comp in self.components
            ]  # type: ignore[assignment]
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
    value: float
    unit: str
    aggregation: str | None = None
    samples: int | None = None
    qualifiers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name, "value": self.value, "unit": self.unit}
        if self.aggregation is not None:
            result["aggregation"] = self.aggregation
        if self.samples is not None:
            result["samples"] = self.samples
        if self.qualifiers:
            result["qualifiers"] = self.qualifiers
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            aggregation=data.get("aggregation"),
            samples=data.get("samples"),
            qualifiers=data.get("qualifiers", {}),
        )


@dataclass(frozen=True)
class OperatorIdentifier:
    """Operator identifier."""

    scope: OperatorScope
    name: str
    location: str
    id: str | None = None  # pylint: disable=invalid-name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "scope": self.scope.value,
            "name": self.name,
            "location": self.location,
        }
        if self.id is not None:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorIdentifier:
        """Create from dictionary."""
        return cls(
            scope=OperatorScope(data["scope"]),
            name=data["name"],
            location=data["location"],
            id=data.get("id"),
        )


@dataclass(frozen=True)
class Breakdown:
    """Breakdown information."""

    scope: OperatorScope
    name: str
    location: str
    metrics: list[Metric]
    id: str | None = None  # pylint: disable=invalid-name
    qualifiers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "scope": self.scope.value,
            "name": self.name,
            "location": self.location,
            "metrics": [m.to_dict() for m in self.metrics],
        }
        if self.id is not None:
            result["id"] = self.id
        if self.qualifiers:
            result["qualifiers"] = self.qualifiers  # type: ignore[assignment]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Breakdown:
        """Create from dictionary."""
        return cls(
            scope=OperatorScope(data["scope"]),
            name=data["name"],
            location=data["location"],
            metrics=[Metric.from_dict(m) for m in data["metrics"]],
            id=data.get("id"),
            qualifiers=data.get("qualifiers", {}),
        )


@dataclass(frozen=True)
class Check:
    """Check information."""

    id: str  # pylint: disable=invalid-name
    status: CheckStatus
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"id": self.id, "status": self.status.value}
        if self.details:
            result["details"] = self.details  # type: ignore[assignment]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Check:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            status=CheckStatus(data["status"]),
            details=data.get("details", {}),
        )


@dataclass(frozen=True)
class Entity:
    """Entity information."""

    scope: OperatorScope
    name: str
    location: str
    placement: str
    id: str | None = None  # pylint: disable=invalid-name
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "scope": self.scope.value,
            "name": self.name,
            "location": self.location,
            "placement": self.placement,
        }
        if self.id is not None:
            result["id"] = self.id
        if self.attributes:
            result["attributes"] = self.attributes  # type: ignore[assignment]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Create from dictionary."""
        return cls(
            scope=OperatorScope(data["scope"]),
            name=data["name"],
            location=data["location"],
            placement=data["placement"],
            id=data.get("id"),
            attributes=data.get("attributes", {}),
        )


@dataclass(frozen=True)
class Result:
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
