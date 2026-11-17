# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Keep a check's backend selection consistent across conversion and analysis.

RuntimeSession installs the selected backends, runs their operations and records
the resulting files in execution-plan.json. Entering a session makes it available
to converters and target analyzers; leaving it restores the previous session.
Profile selection and licence handling are shared by CLI and API callers.
"""

from __future__ import annotations

import json
import os
import pydoc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.metadata import distributions
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlia.core.common import AdviceCategory
from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import (
    ExecutionProfile,
    RuntimeDefinition,
    catalog_profiles,
    load_profile,
    selected_backends,
)
from mlia.runtime.manager import RuntimeManager

if TYPE_CHECKING:
    from mlia.target.config import TargetProfile


@dataclass
class RuntimeSession:
    """Own the selected backends and record the jobs run during one analysis."""

    profile: ExecutionProfile
    manager: RuntimeManager
    output_dir: Path
    calls: list[dict[str, Any]] = field(default_factory=list)
    frontend_packages: dict[str, str] = field(
        default_factory=lambda: {
            dist.metadata["Name"]: dist.version
            for dist in distributions()
            if dist.metadata["Name"].lower().startswith("mlia")
        }
    )

    @contextmanager
    def activate(
        self, *, accept_eula: bool | None, prompt: Callable[[str], str] | None = None
    ) -> Iterator[RuntimeSession]:
        """Install selected backends and make this session available to callers."""
        provision_runtimes(
            [definition for _, definition in self.profile.stages.values()],
            manager=self.manager,
            accept_eula=accept_eula,
            prompt=prompt,
        )
        token = _CURRENT.set(self)
        try:
            self.persist()
            yield self
        finally:
            _CURRENT.reset(token)

    def invoke(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run an operation using its explicit binding and record completion."""
        for stage, (selected, definition) in self.profile.stages.items():
            if selected != operation:
                continue
            stage_options = self.profile.data["stages"][stage].get("options", {})
            effective = {**stage_options, **payload}
            if "options" in stage_options or "options" in payload:
                effective["options"] = {
                    **stage_options.get("options", {}),
                    **payload.get("options", {}),
                }
            result = self.manager.execute(
                definition, operation, effective, self.output_dir / "runtime-jobs"
            )
            self.calls.append(
                {
                    "operation": operation,
                    "runtime": definition.name,
                    "identity": definition.identity,
                    "job_id": result["job_id"],
                    "artifacts": result["artifacts"],
                }
            )
            self.persist()
            return result
        raise ConfigurationError(
            f"Execution profile {self.profile.data['id']} does not "
            f"define required operation {operation}"
        )

    def persist(self) -> None:
        """Preserve exact selections and actual worker invocations."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "execution-plan.json"
        temporary = path.with_suffix(".tmp")
        plan = self.profile.describe()
        for stage in plan["stages"]:
            stage["checks"] = "runtime ready"
        temporary.write_text(
            json.dumps(
                {
                    **plan,
                    "frontend_packages": self.frontend_packages,
                    "calls": self.calls,
                },
                indent=2,
            )
        )
        temporary.replace(path)


_CURRENT: ContextVar[RuntimeSession | None] = ContextVar(
    "mlia_runtime_session", default=None
)


def current_session() -> RuntimeSession | None:
    """Return this execution's bindings without global mutable selection."""
    return _CURRENT.get()


def invoke(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Invoke an operation only through its selected isolated runtime."""
    session = current_session()
    if session is None:
        raise ConfigurationError(
            f"Operation {operation} requires an execution profile. "
            "Use mlia check or run_advisor to select and provision its runtime."
        )
    return session.invoke(operation, payload)


def select_profile(
    selection: str | Path | None, backends: list[str], model: str | Path | None
) -> ExecutionProfile | None:
    """Find an explicit profile or one unambiguous applicable default."""
    selection = selection or os.environ.get("MLIA_EXECUTION_PROFILE")
    if selection is None:
        selection = _default_profile_path(backends, model)
    if selection is None:
        return None
    profile = load_profile(selection, model=model)
    profile_backend = selected_backends(profile.data)
    if profile_backend and set(backends) != set(profile_backend):
        raise ConfigurationError(
            f"Execution profile requires backend {profile_backend}, "
            f"but selected {backends}"
        )
    inputs = profile.data.get("inputs")
    if (
        inputs
        and model is not None
        and Path(model).suffix.lower().lstrip(".") not in inputs
    ):
        raise ConfigurationError(
            f"Execution profile does not accept model "
            f"representation {Path(model).suffix}"
        )
    return profile


def _default_profile_path(backends: list[str], model: str | Path | None) -> Path | None:
    """Find one matching default, allowing no profile only for unmanaged backends."""
    suffix = Path(model).suffix.lower().lstrip(".") if model is not None else ""
    candidates = [
        path
        for path, data in catalog_profiles()
        if data.get("default")
        and set(selected_backends(data)) == set(backends)
        and (not data.get("inputs") or suffix in data["inputs"])
    ]
    if len(candidates) > 1:
        raise ConfigurationError(
            "Multiple default execution profiles match; "
            "select --execution-profile explicitly"
        )
    if candidates:
        return candidates[0]
    from mlia.backend.registry import registry

    for backend in backends:
        config = registry.items.get(backend)
        if config is not None and config.runtime_operation:
            raise ConfigurationError(
                f"No default runtime catalog profile matches {backend} "
                f"and .{suffix}. Install a provider wheel containing the backend "
                "and its profiles, or select a supplied catalog using "
                "MLIA_RUNTIME_CATALOG."
            )
    return None


@contextmanager
def execution_session(
    profile: ExecutionProfile | None,
    output_dir: Path,
    *,
    accept_eula: bool | None,
    prompt: Callable[[str], str] | None = None,
) -> Iterator[RuntimeSession | None]:
    """Provision selected runtimes before execution and scope their bindings."""
    if profile is None:
        yield None
        return
    session = RuntimeSession(profile, RuntimeManager(), output_dir)
    with session.activate(accept_eula=accept_eula, prompt=prompt):
        yield session


def validate_target_profile(
    profile: ExecutionProfile | None,
    target_config: TargetProfile,
    model: str | Path | None,
    categories: set[AdviceCategory],
) -> None:
    """Delegate representation and result-contract checks to the consuming target."""
    if profile is None:
        return
    from mlia.target.registry import registry

    validator = registry.items[target_config.target].runtime_validator
    if validator is None:
        raise ConfigurationError(
            f"Target {target_config.target} does not support "
            "isolated execution profiles"
        )
    validator(profile, model, categories, target_config)


def profile_backends(
    selection: str | Path | None,
    backends: list[str] | None,
    *,
    target_profile: str | Path | None = None,
    model: object = None,
) -> list[str] | None:
    """Use an explicitly selected execution profile before applying target defaults."""
    selection = selection or os.environ.get("MLIA_EXECUTION_PROFILE")
    if not backends and selection:
        selected = selected_backends(
            load_profile(
                selection, model=model if isinstance(model, (str, Path)) else None
            ).data
        )
        if selected:
            return selected
    if (
        not backends
        and not selection
        and target_profile
        and isinstance(model, (str, Path))
    ):
        from mlia.target.registry import profile

        try:
            return profile(target_profile).default_runtime_backends(Path(model))
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
    return backends


def provision_runtimes(
    definitions: list[RuntimeDefinition],
    *,
    manager: RuntimeManager | None = None,
    accept_eula: bool | None = False,
    prompt: Callable[[str], str] | None = None,
    force: bool = False,
) -> None:
    """Apply the same licence and installation policy to checks and backend commands."""
    manager = manager or RuntimeManager()
    definitions = list(
        {definition.identity: definition for definition in definitions}.values()
    )
    missing = [d for d in definitions if force or not manager.ready(d)]
    licences = [d for d in missing if d.data.get("license")]
    accepted = bool(accept_eula)
    if licences and accept_eula is False:
        raise ConfigurationError(
            "EULA acceptance required for selected runtimes; use accept_eula=True "
            "(--accept-eula for backend install, "
            "or --i-agree-to-the-contained-eula for check)"
        )
    # Validate every supplied input before displaying terms or starting installation.
    for definition in missing:
        definition.verify_sources()
    if licences and not accepted:
        for definition in licences:
            _accept_runtime_license(definition, prompt)
        accepted = True
    for definition in definitions:
        manager.install(definition, accept_eula=accepted, force=force)


def _accept_runtime_license(
    definition: RuntimeDefinition, prompt: Callable[[str], str] | None
) -> None:
    """Retain paged CLI terms and full-text frontend consent prompts."""
    text = (
        f"End-user licence agreement for runtime '{definition.name}'\n\n"
        + definition.license_text()
    )
    try:
        if prompt is not None:
            answer = prompt(text + "\n\nAccept these terms and install the runtime?")
            accepted = answer.strip().lower() in {"y", "yes"}
        else:
            pydoc.pager(text)
            while True:
                answer = input(
                    "Do you agree to the above terms and conditions? [yes/no]: "
                )
                answer = answer.strip().casefold()
                if answer in {"yes", "no", "quit"}:
                    break
            accepted = answer == "yes"
    except (EOFError, StopIteration) as exc:
        raise ConfigurationError(
            f"Input ended before the EULA for {definition.name} was accepted"
        ) from exc
    if not accepted:
        raise ConfigurationError(
            f"Runtime licence acceptance declined for {definition.name}"
        )
