# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Find runtime packages and resolve execution profiles to operation bindings.

Catalogs contain runtime manifests and profiles supplied by installed plugins
or local bundles. Reading them validates declarations without loading workers
or installing their dependencies.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename

from mlia.core.errors import ConfigurationError
from mlia.utils.filesystem import get_mlia_resource_dirs

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass(frozen=True)
class RuntimeDefinition:
    """A validated package selection and the files needed to install it."""

    path: Path
    data: dict[str, Any]
    identity: str
    source_template: bool = False

    @property
    def name(self) -> str:
        """Return the qualified public identifier."""
        return str(self.data["id"])

    @classmethod
    def load(cls, path: Path) -> RuntimeDefinition:
        """Validate a manifest without installing or importing worker code."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            source_template = str(data["id"]).endswith("@{version}")
            data["id"] = _runtime_id(data["id"], data.get("provider_package"))
            _validate_runtime(data)
            _validate_package(data)
            _validate_wheels(data, path.parent)
            validate_operations(data.get("operations"))
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ConfigurationError(
                f"Invalid runtime definition {path}: {exc}"
            ) from exc
        return cls(
            path.resolve(),
            data,
            hashlib.sha256(canonical.encode()).hexdigest(),
            source_template,
        )

    def verify_identity(self) -> None:
        """Reject mutation of a previously validated manifest."""
        canonical = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != self.identity:
            raise ConfigurationError("Runtime definition changed after validation")

    def verify_sources(self) -> None:
        """Reject missing or modified provisioning inputs."""
        self.verify_identity()
        for wheel in self.data.get("wheels", []):
            path = contained(self.path.parent, wheel["source"])
            if not path.is_file() or digest(path) != wheel["sha256"]:
                raise ConfigurationError(
                    f"Missing wheel or hash mismatch for runtime {self.name}: {path}"
                )
        if self.data.get("license"):
            self.license_text()

    def license_text(self) -> str:
        """Read the full EULA from its verified wheel before installation."""
        self.verify_identity()
        license_source = self.data["license"]
        wheel = next(
            w for w in self.data["wheels"] if w["source"] == license_source["wheel"]
        )
        path = contained(self.path.parent, wheel["source"])
        try:
            if digest(path) != wheel["sha256"]:
                raise ValueError("EULA wheel hash mismatch")
            with zipfile.ZipFile(path) as stream:
                text = stream.read(license_source["path"]).decode("utf-8")
            if not text.strip():
                raise ValueError("Empty EULA")
            return text
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise ConfigurationError(
                f"Cannot read runtime EULA for {self.name}: {exc}"
            ) from exc


@dataclass(frozen=True)
class ExecutionProfile:
    """Resolved operation bindings and serializable profile metadata."""

    path: Path
    data: dict[str, Any]
    stages: dict[str, tuple[str, RuntimeDefinition]]
    profile_sha256: str

    def describe(self) -> dict[str, Any]:
        """Describe the exact choices without provisioning them."""
        return {
            "profile": self.data["id"],
            "profile_sha256": self.profile_sha256,
            "stages": [
                {
                    "stage": name,
                    "operation": op,
                    "runtime": definition.name,
                    "identity": definition.identity,
                    "checks": "runtime preflight required",
                }
                for name, (op, definition) in self.stages.items()
            ],
        }


@dataclass
class _CatalogSnapshot:
    definitions: dict[str, RuntimeDefinition] | None = None
    profiles: list[tuple[Path, dict[str, Any]]] | None = None
    profile_contents: dict[Path, bytes] = field(default_factory=dict)


_SNAPSHOT: ContextVar[_CatalogSnapshot | None] = ContextVar(
    "mlia_runtime_catalog_snapshot", default=None
)


def digest(path: Path) -> str:
    """Hash file contents without reading large tools into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained(root: Path, name: str) -> Path:
    """Resolve a relative artifact name inside its assigned directory."""
    try:
        path = (root / name).resolve()
    except RuntimeError as exc:
        raise ConfigurationError(f"Invalid artifact path: {name}") from exc
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ConfigurationError(f"Artifact path outside assigned directory: {name}")
    return path


def _runtime_id(value: str, provider: str | None) -> str:
    """Resolve a current bundle against its owning installed wheel's version."""
    if not isinstance(value, str):
        raise ValueError("Runtime id must be a string")
    if not value.endswith("@{version}"):
        return value
    if not isinstance(provider, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", provider
    ):
        raise ValueError("A version-derived runtime requires provider_package")
    try:
        version = importlib.metadata.version(provider)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConfigurationError(
            f"The provider package '{provider}' is not installed. "
            f"Install '{provider}' in the MLIA environment to resolve {value}."
        ) from exc
    return value.removesuffix("{version}") + version


def _validate_runtime(data: dict[str, Any]) -> None:
    """Check the runtime identity and host requirements."""
    if type(data["schema_version"]) is not int or data["schema_version"] != 2:
        raise ValueError("Unsupported runtime schema")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+@[A-Za-z0-9._+!-]+", data["id"]) or any(
        part in {"", ".", ".."} for part in data["id"].split("/")
    ):
        raise ValueError("Runtime id must be a qualified name@release")
    if not isinstance(data["python"], str) or not data["python"]:
        raise ValueError("A Python interpreter is required")
    if data["platform"] != "linux":
        raise ValueError("Only Linux runtime definitions are supported")
    if "architecture" in data and (
        not isinstance(data["architecture"], str) or not data["architecture"]
    ):
        raise ValueError("Runtime architecture must be a nonempty string")


def _validate_package(data: dict[str, Any]) -> None:
    """Require an exact package version and HTTPS dependency indexes."""
    requirement = Requirement(data["package"])
    pins = list(requirement.specifier)
    if (
        requirement.url
        or requirement.marker
        or len(pins) != 1
        or pins[0].operator != "=="
        or "*" in pins[0].version
    ):
        raise ValueError("Runtime package must select one exact version")
    indexes = data.get("indexes", [])
    if not isinstance(indexes, list) or any(
        not isinstance(index, str)
        or urlsplit(index).scheme != "https"
        or not urlsplit(index).netloc
        for index in indexes
    ):
        raise ValueError("Runtime indexes must be HTTPS URLs")


def _validate_wheels(data: dict[str, Any], directory: Path) -> None:
    """Check supplied wheel paths, hashes and the EULA's location within a wheel."""
    wheels = data.get("wheels", [])
    if not isinstance(wheels, list):
        raise ValueError("Runtime wheels must be a list")
    sources = set()
    for wheel in wheels:
        name = wheel["source"]
        contained(directory, name)
        parse_wheel_filename(Path(name).name)
        if name in sources:
            raise ValueError(f"Duplicate runtime wheel: {name}")
        sources.add(name)
        if not re.fullmatch(r"[a-f0-9]{64}", wheel["sha256"]):
            raise ValueError("Invalid wheel hash")
    license_source = data.get("license")
    if license_source is None:
        return
    if license_source["wheel"] not in sources:
        raise ValueError("Runtime license must name a supplied wheel")
    if not isinstance(license_source["path"], str) or not license_source["path"]:
        raise ValueError("Runtime license must name the full EULA in its wheel")
    contained(directory, license_source["path"])


def validate_operations(operations: Any) -> None:
    """Validate entrypoint declarations without importing worker code."""
    if not isinstance(operations, dict) or not operations:
        raise ValueError("At least one operation is required")
    for name, operation in operations.items():
        if not isinstance(name, str) or not name or not isinstance(operation, dict):
            raise ValueError("Invalid operation declaration")
        if not isinstance(operation.get("entrypoint"), str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", operation["entrypoint"]
        ):
            raise ValueError("Invalid operation entrypoint")
        if (
            not isinstance(operation.get("result_schema"), str)
            or not operation["result_schema"]
        ):
            raise ValueError("An operation result schema is required")


def _profile_content(path: Path) -> bytes:
    """Keep profile discovery and binding on the same file contents for one check."""
    snapshot = _SNAPSHOT.get()
    if snapshot is None:
        return path.read_bytes()
    path = path.resolve()
    if path not in snapshot.profile_contents:
        snapshot.profile_contents[path] = path.read_bytes()
    return snapshot.profile_contents[path]


def _read_profile(path: Path, content: bytes | None = None) -> dict[str, Any]:
    """Read and validate profile structure before resolving any runtime bindings."""
    try:
        data = tomllib.loads(
            (_profile_content(path) if content is None else content).decode("utf-8")
        )
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or not isinstance(data["id"], str)
        ):
            raise ValueError("Unsupported execution profile schema")
        selected_backends(data)
        if "default" in data and not isinstance(data["default"], bool):
            raise ValueError("Execution profile default must be a boolean")
        if "inputs" in data and (
            not isinstance(data["inputs"], list)
            or any(not isinstance(item, str) or not item for item in data["inputs"])
        ):
            raise ValueError(
                "Execution profile inputs must be a list of nonempty strings"
            )
        if not isinstance(data["stages"], dict) or not data["stages"]:
            raise ValueError("Execution profile requires a nonempty stages table")
        for name, stage in data["stages"].items():
            _validate_stage(name, stage)
        return data
    except (OSError, ValueError, KeyError, TypeError, ConfigurationError) as exc:
        raise ConfigurationError(f"Invalid execution profile {path}: {exc}") from exc


def _validate_stage(name: str, stage: Any) -> None:
    """Check one profile binding before attempting catalog resolution."""
    if not isinstance(stage, dict):
        raise ValueError(f"Stage {name} must be a table")
    for key in ("operation", "runtime"):
        if not isinstance(stage.get(key), str) or not stage[key]:
            raise ValueError(f"Stage {name} requires {key}")
    if "inputs" in stage and (
        not isinstance(stage["inputs"], list)
        or any(not isinstance(item, str) or not item for item in stage["inputs"])
    ):
        raise ValueError(f"Stage {name} inputs must be a list of nonempty strings")
    options = stage.get("options", {})
    if not isinstance(options, dict):
        raise ValueError(f"Stage {name} options must be a table")
    if not isinstance(options.get("options", {}), dict):
        raise ValueError(f"Stage {name} nested options must be a table")
    json.dumps(options, allow_nan=False)
    provider = stage.get("provider_package")
    if provider is not None and (
        not isinstance(provider, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", provider)
    ):
        raise ValueError(f"Stage {name} provider_package must be a package name")


def selected_backends(data: dict[str, Any]) -> list[str]:
    """Return a profile's backend selection, including combined backend paths."""
    value = data.get("backends")
    if value is None:
        backend = data.get("backend")
        if backend is None:
            return []
        if not isinstance(backend, str) or not backend:
            raise ConfigurationError("Execution profile backend must be a name")
        return [backend]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
        or "backend" in data
    ):
        raise ConfigurationError(
            "Execution profile requires unique backends or one backend"
        )
    return value


def local_catalog_directory() -> Path:
    """Return the durable catalog of explicitly installed local runtime bundles."""
    return (
        Path(os.environ.get("MLIA_RUNTIME_HOME", "~/.mlia/runtimes"))
        .expanduser()
        .resolve()
        / ".catalog"
    )


def catalog_directories() -> list[Path]:
    """Find packaged and explicitly supplied catalogs in deterministic order."""
    directories = [
        Path(p).expanduser()
        for p in os.environ.get("MLIA_RUNTIME_CATALOG", "").split(os.pathsep)
        if p
    ]
    directories.extend(p / "runtimes" for p in get_mlia_resource_dirs())
    directories.append(local_catalog_directory())
    return list(dict.fromkeys(p.resolve() for p in directories if p.is_dir()))


@contextmanager
def catalog_snapshot() -> Iterator[None]:
    """Reuse discovery within one operation; later calls see current resources."""
    if _SNAPSHOT.get() is not None:
        yield
        return
    token = _SNAPSHOT.set(_CatalogSnapshot())
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


def catalog_profiles() -> list[tuple[Path, dict[str, Any]]]:
    """Read catalog profiles consistently, reusing the current discovery snapshot."""
    snapshot = _SNAPSHOT.get()
    if snapshot is not None and snapshot.profiles is not None:
        return snapshot.profiles
    profiles: list[tuple[Path, dict[str, Any]]] = []
    for directory in catalog_directories():
        for path in sorted(directory.rglob("*.profile.toml")):
            data = _read_profile(path)
            # Editable installs may expose both source and built copies.
            if any(data == existing for _, existing in profiles):
                continue
            profiles.append((path, data))
    if snapshot is not None:
        snapshot.profiles = profiles
    return profiles


def load_catalog_variants() -> dict[tuple[str, str | None], RuntimeDefinition]:
    """Validate every architecture variant before choosing a host candidate."""
    variants: dict[tuple[str, str | None], RuntimeDefinition] = {}
    for directory in catalog_directories():
        for path in sorted(directory.rglob("*.runtime.json")):
            definition = RuntimeDefinition.load(path)
            key = (definition.name, definition.data.get("architecture"))
            previous = variants.get(key)
            # Editable installs expose both source templates and wheel resources.
            # The built definition binds the current worker sources and archives.
            if (
                previous is not None
                and previous.source_template != definition.source_template
                and previous.data.get("provider_package")
                == definition.data.get("provider_package")
            ):
                if definition.source_template:
                    continue
            elif previous is not None and previous.identity != definition.identity:
                raise ConfigurationError(
                    f"Conflicting runtime definition: {definition.name}"
                )
            variants[key] = definition
    return variants


def load_catalog() -> dict[str, RuntimeDefinition]:
    """Select host-specific or universal runtimes after validating all variants."""
    snapshot = _SNAPSHOT.get()
    if snapshot is not None and snapshot.definitions is not None:
        return snapshot.definitions
    definitions: dict[str, RuntimeDefinition] = {}
    priority = {None: 1, platform.machine(): 2}
    variants = sorted(
        load_catalog_variants().values(),
        key=lambda item: (
            priority.get(item.data.get("architecture"), 0),
            str(item.data.get("architecture")),
        ),
    )
    for definition in variants:
        definitions[definition.name] = definition
    if snapshot is not None:
        snapshot.definitions = definitions
    return definitions


def load_profile(
    path: str | Path,
    catalog: dict[str, RuntimeDefinition] | None = None,
    *,
    model: str | Path | None = None,
) -> ExecutionProfile:
    """Resolve profile bindings, selecting stages applicable to the model input."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        matches = [file for file, data in catalog_profiles() if data["id"] == str(path)]
        if len(matches) != 1:
            raise ConfigurationError(
                f"Execution profile is unknown or ambiguous: {path}"
            )
        candidate = matches[0]
    definitions = load_catalog() if catalog is None else catalog
    try:
        content = _profile_content(candidate)
        data = _read_profile(candidate, content)
        stages = {}
        operations = set()
        suffix = Path(model).suffix.lower().lstrip(".") if model is not None else None
        for name, stage in data["stages"].items():
            if stage.get("inputs") and suffix not in stage["inputs"]:
                continue
            operation, definition = _resolve_stage(name, stage, definitions, data["id"])
            stage["runtime"] = definition.name
            if operation in operations:
                raise ValueError(f"Ambiguous repeated operation: {operation}")
            operations.add(operation)
            stages[name] = (operation, definition)
        if not stages:
            raise ValueError("No execution stages match the supplied input")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigurationError(
            f"Invalid execution profile {candidate}: {exc}"
        ) from exc
    return ExecutionProfile(
        candidate.resolve(), data, stages, hashlib.sha256(content).hexdigest()
    )


def _resolve_stage(
    name: str,
    stage: dict[str, Any],
    definitions: dict[str, RuntimeDefinition],
    profile_id: str,
) -> tuple[str, RuntimeDefinition]:
    """Bind an operation to its selected runtime, with actionable provider errors."""
    provider = stage.get("provider_package")
    runtime_id = _runtime_id(stage["runtime"], provider)
    if runtime_id not in definitions:
        raise ConfigurationError(
            f"Unknown runtime {runtime_id} required by execution profile "
            f"'{profile_id}' (stage '{name}'). {_provider_guidance(provider)}"
        )
    definition = definitions[runtime_id]
    operation = stage["operation"]
    if operation not in definition.data["operations"]:
        raise ValueError(f"Runtime {runtime_id} does not provide {operation}")
    return operation, definition


def _provider_guidance(provider: str | None) -> str:
    """Explain how to obtain a missing runtime without guessing its provider."""
    if provider is None:
        return (
            "Install the provider package or set MLIA_RUNTIME_CATALOG "
            "to a supplied catalog containing this runtime."
        )
    try:
        installed_version = importlib.metadata.version(provider)
    except importlib.metadata.PackageNotFoundError:
        return (
            f"The provider package '{provider}' is not installed. "
            f"Install '{provider}' in the MLIA environment and retry."
        )
    return (
        f"Installed provider '{provider}' ({installed_version}) "
        "does not supply this runtime. Install a compatible release "
        "of the provider or select an available execution profile."
    )


def default_backend_runtime(backend: str, operation: str) -> RuntimeDefinition:
    """Resolve a backend alias only when its published defaults agree."""
    definitions = load_catalog()
    selected: dict[str, RuntimeDefinition] = {}
    for path, data in catalog_profiles():
        try:
            if not data.get("default") or backend not in selected_backends(data):
                continue
            for name, stage in data["stages"].items():
                if stage["operation"] != operation:
                    continue
                _, definition = _resolve_stage(name, stage, definitions, data["id"])
                selected[definition.identity] = definition
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(
                f"Invalid execution profile {path}: {exc}"
            ) from exc
    if not selected:
        raise ConfigurationError(
            f"No default runtime catalog is available for {backend}. "
            "Install a provider wheel containing its runtime archives and profiles, "
            "or set MLIA_RUNTIME_CATALOG to a supplied catalog directory."
        )
    if len(selected) > 1:
        raise ConfigurationError(
            f"Multiple default runtimes for {backend}; "
            "install a qualified runtime ID explicitly"
        )
    return next(iter(selected.values()))
