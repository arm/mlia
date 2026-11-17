# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Package backend code for installation in its own Python environment.

When building a plugin, read its runtime/*/pyproject.toml projects and build their
worker wheels. Include those wheels, vendor files, licences and runtime manifests
inside the plugin package. Backend dependencies remain in the worker's metadata;
they are installed when the backend is first used, outside the frontend environment.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.builders.wheel import WheelBuilder
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from mlia.runtime.catalog import RuntimeDefinition, digest, validate_operations

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _runtime_imports(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    """Walk executable imports, excluding conventional typing-only branches."""
    imports: list[ast.Import | ast.ImportFrom] = []
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.If) and (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            or isinstance(node.test, ast.Attribute)
            and node.test.attr == "TYPE_CHECKING"
        ):
            pending.extend(node.orelse)
        else:
            pending.extend(ast.iter_child_nodes(node))
    return imports


def module_sources(modules: list[str], roots: list[Path]) -> dict[str, Path]:
    """Resolve runtime MLIA helper imports without importing frontend code.

    Unrequested parent packages remain namespaces, avoiding frontend registration
    side effects. Explicit package imports retain their initializer.
    """
    roots = list(dict.fromkeys(root.resolve() for root in roots))
    found: dict[str, Path] = {}
    visited = set()
    pending = list(modules)
    while pending:
        name = pending.pop()
        if name in visited or name == "mlia":
            continue
        visited.add(name)
        relative = Path(*name.split("."))
        matches = list(
            dict.fromkeys(
                candidate.resolve()
                for root in roots
                for candidate in (
                    root / relative.with_suffix(".py"),
                    root / relative / "__init__.py",
                )
                if candidate.is_file()
            )
        )
        if not matches:
            continue  # Imported attributes and namespace packages have no file.
        if len(matches) != 1:
            raise ValueError(f"Ambiguous source module: {name}")
        source = matches[0]
        found[name] = source
        package = name if source.name == "__init__.py" else name.rpartition(".")[0]
        for node in _runtime_imports(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            else:
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    if node.level > len(parts):
                        raise ValueError(f"Relative import escapes package: {source}")
                    base = ".".join(
                        parts[: len(parts) - node.level + 1] + ([base] if base else [])
                    )
                candidates = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            pending.extend(
                candidate for candidate in candidates if candidate.startswith("mlia.")
            )
    for name in modules:
        if name not in found:
            raise ValueError(f"Required worker source module is missing: {name}")
    return found


class RuntimePackageBuildHook(BuildHookInterface):
    """Tell Hatch which backend files belong in a worker wheel.

    Include selected shared Python modules without copying the frontend package,
    ask the plugin's prepare.py for vendor binaries and licences, and tag native
    wheels for their target architecture. Workers with self-contained sources and
    standard packaging do not need this hook.
    """

    def initialize(self, version: str, build_data: dict) -> None:
        """Prepare the wheel for installation by a standard package installer."""
        project = Path(self.root)
        root = project.parent.parent
        includes = build_data.setdefault("force_include", {})
        roots = [root / "src", Path(__file__).resolve().parents[2]]
        for name, source in module_sources(
            self.config.get("modules", []), roots
        ).items():
            suffix = "/__init__.py" if source.name == "__init__.py" else ".py"
            includes[str(source)] = name.replace(".", "/") + suffix
        configuration = tomllib.loads((project / "pyproject.toml").read_text())
        architectures = _architectures(configuration["tool"]["mlia"]["runtime"])
        architecture = os.environ.get("MLIA_RUNTIME_BUILD_ARCH") or (
            platform.machine() if architectures else None
        )
        if architecture is not None and architecture not in architectures:
            raise ValueError(f"Unsupported runtime build architecture: {architecture}")
        if architecture:
            python = configuration["tool"]["mlia"]["runtime"].get("python", "3.12")
            match = re.fullmatch(r"(3)\.(\d+)(?:\.\d+)?", python)
            if match is None:
                raise ValueError(
                    "Runtime package builds require a numeric Python version"
                )
            build_data["tag"] = f"py{match[1]}{match[2]}-none-linux_{architecture}"
            build_data["pure_python"] = False
        preparer = root / "runtime/prepare.py"
        if preparer.is_file():
            spec = importlib.util.spec_from_file_location(
                "mlia_package_inputs", preparer
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.prepare(root, architecture=architecture, package=project.name)
            includes.update(module.package_files(root, project.name, architecture))


def runtime_resource_files(
    root: Path,
    provider: str,
    release: str,
    target_name: str,
    *,
    architecture: str | None = None,
) -> dict[str, str]:
    """Return the worker wheels, manifests and profiles to include in a plugin.

    A checkout builds the workers first. An extracted source distribution already
    contains their wheels and can be repackaged without vendor downloads or git.
    """
    Version(release)
    resources = root / "src/mlia/resources/runtimes"
    if (root / "PKG-INFO").is_file():
        return _included_resources(resources, target_name, architecture)
    output = root / "build/runtime-resources"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    _build_runtime_projects(root, output, provider, release, architecture)
    _copy_existing_resources(resources, output)
    return _included_resources(output, target_name, architecture)


def _build_runtime_projects(
    root: Path, output: Path, provider: str, release: str, architecture: str | None
) -> None:
    """Build each declared worker for the requested architectures."""
    for project in sorted((root / "runtime").glob("*/pyproject.toml")):
        config = tomllib.loads(project.read_text())
        runtime = config["tool"]["mlia"]["runtime"]
        validate_operations(runtime["operations"])
        name = runtime["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", name)
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise ValueError(f"Invalid runtime name: {name}")
        architectures: list[str | None] = list(_architectures(runtime))
        for arch in architectures or [None]:
            if architecture and arch not in (None, architecture):
                continue
            destination = output / name / (arch or "any") / release
            destination.mkdir(parents=True)
            _build_runtime(project.parent, config, destination, provider, release, arch)


def _architectures(runtime: dict[str, Any]) -> list[str]:
    """Return declared architectures; an omitted field denotes a universal wheel."""
    if "architectures" not in runtime:
        return []
    values = runtime["architectures"]
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Invalid runtime architectures: expected a nonempty list of unique names"
        )
    return values


@contextmanager
def _worker_build_environment(release: str, architecture: str | None) -> Iterator[None]:
    """Use the plugin's version and target architecture for this worker build only."""
    overrides = {
        "UV_DYNAMIC_VERSIONING_BYPASS": release,
        "MLIA_RUNTIME_BUILD_ARCH": architecture,
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_runtime(
    project: Path,
    config: dict[str, Any],
    destination: Path,
    provider: str,
    release: str,
    architecture: str | None,
) -> None:
    """Build one worker wheel and write the manifest used to install it."""
    with _worker_build_environment(release, architecture):
        builder = WheelBuilder(str(project))
        wheel = Path(next(builder.build(directory=str(destination))))
    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheel.name)
    if wheel_name != canonicalize_name(
        config["project"]["name"]
    ) or wheel_version != Version(release):
        raise ValueError(
            f"Runtime wheel must match provider release {release}: {wheel.name}"
        )
    runtime = config["tool"]["mlia"]["runtime"]
    wheels = [wheel, *_dependency_wheels(project, runtime, destination, architecture)]
    data = {
        "schema_version": 2,
        "id": f"{runtime['id']}@{release}",
        "provider_package": provider,
        "python": runtime.get("python", "3.12"),
        "platform": "linux",
        "package": f"{config['project']['name']}=={release}",
        "wheels": [{"source": item.name, "sha256": digest(item)} for item in wheels],
        "operations": runtime["operations"],
    }
    if architecture:
        data["architecture"] = architecture
    if runtime.get("indexes"):
        data["indexes"] = runtime["indexes"]
    if runtime.get("license"):
        data["license"] = {"wheel": wheel.name, "path": runtime["license"]}
    manifest = destination / "definition.runtime.json"
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    RuntimeDefinition.load(manifest).verify_sources()


def _dependency_wheels(
    project: Path, runtime: dict[str, Any], destination: Path, architecture: str | None
) -> list[Path]:
    """Copy supplied dependency wheels beside the worker for offline installation."""
    wheels = []
    for pattern in runtime.get("wheels", []):
        matches = sorted(project.glob(pattern.format(architecture=architecture)))
        if not matches:
            raise ValueError(f"Missing supplied dependency wheel: {pattern}")
        for source in matches:
            target = destination / source.name
            shutil.copyfile(source, target)
            wheels.append(target)
    return wheels


def _copy_existing_resources(resources: Path, output: Path) -> None:
    """Keep historical backend versions and copy the plugin's execution profiles."""
    for manifest in resources.rglob("*.runtime.json"):
        shutil.copytree(
            manifest.parent, output / manifest.parent.relative_to(resources)
        )
    for profile in resources.rglob("*.profile.toml"):
        target = output / profile.relative_to(resources)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile, target)


def _included_resources(
    output: Path, target_name: str, architecture: str | None
) -> dict[str, str]:
    """Map verified backend files to their paths inside the plugin wheel or sdist."""
    manifests = list(output.rglob("*.runtime.json"))
    if not manifests:
        raise ValueError(f"No runtime package definitions found in {output}")
    prefix = Path("mlia/resources/runtimes")
    if target_name == "sdist":
        prefix = Path("src") / prefix
    includes = {}
    for manifest in manifests:
        definition = RuntimeDefinition.load(manifest)
        if (
            architecture
            and definition.data.get("architecture", architecture) != architecture
        ):
            continue
        definition.verify_sources()
        wheels = [
            manifest.parent / wheel["source"]
            for wheel in definition.data.get("wheels", [])
        ]
        for item in [manifest, *wheels]:
            includes[str(item)] = str(prefix / item.relative_to(output))
    for profile in output.rglob("*.profile.toml"):
        includes[str(profile)] = str(prefix / profile.relative_to(output))
    return includes
