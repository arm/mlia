# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Install runtime packages explicitly supplied through the CLI or Python API.

Handles ``mlia backend install --path`` and ``install_backends(..., path=...)``:
find the supplied manifest and wheels, validate the requested runtime selection,
provision its environment, and retain the package files in the user's catalog.
Later checks and reinstalls can reuse those files after the originals are removed.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import (
    RuntimeDefinition,
    contained,
    load_catalog_variants,
    local_catalog_directory,
)
from mlia.runtime.session import provision_runtimes
from mlia.utils.logging import log_boundary_action


def install_local_runtime(
    path: Path,
    *,
    runtime_id: str | None = None,
    operation: str | None = None,
    accept_eula: bool | None = False,
    force: bool = False,
) -> None:
    """Validate a local backend's selection, install it and retain its source files."""
    definition = load_local_runtime(path)
    if runtime_id is not None and runtime_id != definition.name:
        raise ConfigurationError(
            f"Local runtime {definition.name} does not match {runtime_id}"
        )
    if operation is not None and operation not in definition.data["operations"]:
        raise ConfigurationError(
            f"Local runtime {definition.name} does not provide {operation}"
        )
    _check_catalog_conflicts(definition)
    provision_runtimes([definition], accept_eula=accept_eula, force=force)
    try:
        register_local_runtime(definition)
    except (ConfigurationError, OSError) as exc:
        raise ConfigurationError(
            f"Runtime {definition.name} is installed, but catalog registration failed: "
            f"{exc}. Fix the catalog problem and repeat installation from the same "
            "local package without --force to reuse the installed environment."
        ) from exc
    logging.getLogger(__name__).info(
        "Installed runtime %s; execution profiles select its runtime ID.",
        definition.name,
    )


def _check_catalog_conflicts(definition: RuntimeDefinition) -> None:
    """Reject definitions that would share an installed environment on this host."""
    architecture = definition.data.get("architecture")
    architectures = (
        {None, platform.machine()}
        if architecture in {None, platform.machine()}
        else {architecture}
    )
    for (name, arch), existing in load_catalog_variants().items():
        if (
            name == definition.name
            and arch in architectures
            and existing.identity != definition.identity
        ):
            raise ConfigurationError(
                f"Conflicting runtime definition: {definition.name}; "
                "use a distinct runtime ID"
            )


def load_local_runtime(path: Path) -> RuntimeDefinition:
    """Read a bundle directory, manifest or wheel with adjacent metadata."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"Local runtime path does not exist: {path}")
    if path.is_file() and path.suffix == ".json":
        return RuntimeDefinition.load(path)
    directory = path if path.is_dir() else path.parent
    candidates = sorted(directory.glob("*.runtime.json"))
    if path.is_file():
        candidates = [
            candidate
            for candidate in candidates
            if any(
                contained(directory, wheel["source"]) == path
                for wheel in RuntimeDefinition.load(candidate).data.get("wheels", [])
            )
        ]
    if len(candidates) != 1:
        raise ConfigurationError(
            f"Local runtime at {path} needs one runtime manifest and its supplied "
            "wheels. Pass a published bundle directory, its "
            "definition.runtime.json, or a wheel with adjacent metadata. "
            "A bare backend binary or vendor archive does not define a runtime."
        )
    return RuntimeDefinition.load(candidates[0])


def register_local_runtime(definition: RuntimeDefinition) -> None:
    """Retain verified provisioning inputs for future selection and reinstallation."""
    import fcntl

    definition.verify_sources()
    log_boundary_action(
        logging.getLogger(__name__), f"Registering local runtime {definition.name}"
    )
    catalog = local_catalog_directory()
    if catalog.is_symlink():
        raise ConfigurationError(f"Runtime catalog is a symlink: {catalog}")
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with (catalog.parent / ".catalog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _check_catalog_conflicts(definition)
        destination = catalog / definition.identity
        if destination.exists():
            stored = RuntimeDefinition.load(destination / "definition.runtime.json")
            if stored.identity != definition.identity:
                raise ConfigurationError(
                    f"Conflicting runtime identity at {destination}"
                )
            stored.verify_sources()
            return
        definition.verify_sources()
        # Stage outside the catalog so concurrent readers never see partial inputs.
        with TemporaryDirectory(prefix=".catalog-", dir=catalog.parent) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            names = {wheel["source"] for wheel in definition.data.get("wheels", [])}
            for name in names:
                source = contained(definition.path.parent, name)
                target = contained(bundle, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            manifest = bundle / "definition.runtime.json"
            manifest.write_text(json.dumps(definition.data, indent=2), encoding="utf-8")
            RuntimeDefinition.load(manifest).verify_sources()
            catalog.mkdir(exist_ok=True)
            bundle.rename(destination)
