# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Install backend packages in separate virtual environments and run their jobs.

The manager reuses verified installations, restores them after failed rebuilds,
and prevents removal during execution. Each job exchanges JSON and files with
a worker subprocess; only matching results and files inside its job are accepted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast
from uuid import uuid4

from uv import find_uv_bin

from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import RuntimeDefinition, contained, digest
from mlia.utils.logging import log_boundary_action

logger = logging.getLogger(__name__)


def _environment() -> dict[str, str]:
    """Avoid ambient Python/index configuration leaking into a runtime."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "PIP_", "UV_"))
        and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}
    }


def _run(
    command: list[str],
    log: Path,
    timeout: float,
    *,
    env: dict[str, str],
    cwd: Path | None = None,
    progress: str | None = None,
) -> int:
    """Supervise a whole process group, preserving diagnostics on failure."""
    with (
        log.open("ab") as output,
        subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            start_new_session=True,
        ) as process,
    ):
        try:
            return _wait_for_process(process, timeout, progress, log)
        finally:
            _stop_process_group(process)


def _wait_for_process(
    process: subprocess.Popen[bytes],
    timeout: float,
    progress: str | None,
    log: Path,
) -> int:
    """Wait for completion, reporting progress during long provisioning steps."""
    if progress is None:
        return process.wait(timeout=timeout)
    started = time.monotonic()
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(10, remaining))
        except subprocess.TimeoutExpired:
            logger.info(
                "%s — still working (%ds elapsed). Log: %s",
                progress,
                int(time.monotonic() - started),
                log,
            )


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop descendants even if the worker has already exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Finish cleanup within the frontend worker shutdown grace period.
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # The parent can exit on TERM while descendants ignore it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8000))
        return stream.read().decode("utf-8", errors="replace")


class RuntimeManager:
    """Manage exact installations without modifying the frontend environment."""

    def __init__(self, root: Path | None = None) -> None:
        """Use an explicit runtime directory or the configured user cache."""
        self.root = (
            root
            or Path(
                os.environ.get("MLIA_RUNTIME_HOME", "~/.mlia/runtimes")
            ).expanduser()
        ).resolve()

    def location(self, definition: RuntimeDefinition) -> Path:
        """Return the named installation directory for this runtime and host."""
        definition.verify_identity()
        parts = definition.name.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ConfigurationError("Invalid runtime name for an installation path")
        location = self.root.joinpath(*parts, platform.machine())
        if not location.resolve().is_relative_to(self.root):
            raise ConfigurationError("Runtime location escapes its store")
        if any(
            path.is_symlink()
            for path in (location, *location.parents)
            if path != self.root and path.is_relative_to(self.root)
        ):
            raise ConfigurationError(
                f"Runtime location is a symlink: {location}. "
                "Remove the obsolete runtime alias and retry installation."
            )
        return location

    @staticmethod
    def supports(definition: RuntimeDefinition) -> bool:
        """Return whether this host can provision and execute the runtime."""
        return sys.platform == "linux" and definition.data.get("architecture") in (
            None,
            platform.machine(),
        )

    def ready(self, definition: RuntimeDefinition) -> bool:
        """Recognize only matching, completed installations."""
        if not self.supports(definition):
            return False
        location = self.location(definition)
        try:
            record = json.loads((location / "ready.json").read_text())
            return (
                record["identity"] == definition.identity
                and (location / "venv/bin/python").is_file()
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def install(
        self,
        definition: RuntimeDefinition,
        *,
        accept_eula: bool = False,
        force: bool = False,
    ) -> Path:
        """Provision a verified environment once and publish readiness last."""
        if (
            definition.data.get("architecture", platform.machine())
            != platform.machine()
        ):
            raise ConfigurationError(
                f"Runtime {definition.name} requires architecture "
                f"{definition.data['architecture']}"
            )
        with self._lock(definition):
            destination = self.location(definition)
            if self.ready(definition) and not force:
                return destination
            if definition.data.get("license") and not accept_eula:
                raise ConfigurationError(
                    f"EULA acceptance required for runtime "
                    f"{definition.name}: {definition.data['license']}"
                )
            definition.verify_sources()
            log_boundary_action(
                logger, f"Installing runtime {definition.name} into {destination}"
            )
            with self._replacement(destination):
                destination.mkdir(parents=True)
                env = _environment()
                env["UV_CACHE_DIR"] = str(self.root / ".cache")
                log = destination / "install.log"
                try:
                    self._create_environment(definition, destination, env)
                    self._install_package(definition, destination, env)
                    definition.verify_sources()
                    (destination / "manifest.json").write_text(
                        json.dumps(definition.data, indent=2)
                    )
                    self._validate_installation(definition, destination)
                except subprocess.TimeoutExpired as exc:
                    raise ConfigurationError(
                        f"Runtime installation timed out: {_tail(log)}"
                    ) from exc
                return destination

    def execute(
        self,
        definition: RuntimeDefinition,
        operation: str,
        payload: dict[str, Any],
        output_dir: Path,
        *,
        timeout: float = 3600,
    ) -> dict[str, Any]:
        """Execute a ready runtime while preventing concurrent removal."""
        log_boundary_action(logger, f"Executing runtime {definition.name}: {operation}")
        with self._lock(definition, shared=True):
            if not self.ready(definition):
                raise ConfigurationError(f"Runtime is not installed: {definition.name}")
            return self._execute(
                definition, operation, payload, output_dir, timeout=timeout
            )

    def remove(self, definition: RuntimeDefinition) -> None:
        """Wait for active uses before removing an exact installation."""
        with self._lock(definition):
            location = self.location(definition)
            if not location.exists():
                raise ConfigurationError(f"Runtime is not installed: {definition.name}")
            record = location / "ready.json"
            if record.exists():
                try:
                    identity = json.loads(record.read_text())["identity"]
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise ConfigurationError(
                        f"Cannot identify runtime at {location}"
                    ) from exc
                if identity != definition.identity:
                    raise ConfigurationError(
                        f"A different runtime is installed at {location}; "
                        "refresh the catalog before removing it"
                    )
            log_boundary_action(
                logger, f"Removing runtime {definition.name} from {location}"
            )
            shutil.rmtree(location)

    @contextmanager
    def _lock(
        self, definition: RuntimeDefinition, *, shared: bool = False
    ) -> Iterator[None]:
        if sys.platform != "linux":
            raise ConfigurationError(
                "Managed backend runtimes currently support Linux only"
            )
        # Keep discovery importable on hosts without Unix file locking.
        import fcntl

        locks = self.root / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        # Different contents under the same qualified name share a destination.
        lock_id = hashlib.sha256(
            f"{definition.name}:{platform.machine()}".encode()
        ).hexdigest()
        with (locks / f"{lock_id}.lock").open("a") as stream:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(stream, mode)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @contextmanager
    def _replacement(self, destination: Path) -> Iterator[None]:
        """Keep the failure log and restore any prior runtime after cleaning up."""
        backup = destination.with_name(f".{destination.name}-{uuid4().hex}.backup")
        previous = destination.exists()
        if previous:
            destination.rename(backup)
        try:
            yield
        except BaseException as exc:
            self._retain_installation_log(destination, exc)
            if destination.exists():
                shutil.rmtree(destination)
            if previous:
                backup.rename(destination)
            raise
        else:
            if previous:
                shutil.rmtree(backup)

    @staticmethod
    def _retain_installation_log(destination: Path, error: BaseException) -> None:
        """Retain failure diagnostics without letting IO errors block recovery."""
        try:
            log = destination / "install.log"
            if log.is_file():
                with log.open("a", encoding="utf-8") as stream:
                    for validation_log in (destination / ".validation").glob(
                        "*/worker.log"
                    ):
                        stream.write(
                            validation_log.read_text(encoding="utf-8", errors="replace")
                        )
                    stream.write(f"\nInstallation failed: {error}\n")
                failed_log = destination.with_name(f"{destination.name}.failed.log")
                log.replace(failed_log)
                logger.error(
                    "Runtime installation diagnostics retained at %s", failed_log
                )
        except OSError:
            logger.warning("Could not retain runtime installation log", exc_info=True)

    @staticmethod
    def _create_environment(
        definition: RuntimeDefinition, destination: Path, env: dict[str, str]
    ) -> None:
        """Obtain the requested Python and create the runtime's virtual environment."""
        log = destination / "install.log"
        logger.info("Creating runtime Python environment for %s", definition.name)
        code = _run(
            [
                find_uv_bin(),
                "--no-config",
                "venv",
                "--python",
                definition.data["python"],
                str(destination / "venv"),
            ],
            log,
            600,
            env=env,
            progress=f"Preparing Python for {definition.name}",
        )
        if code:
            raise ConfigurationError(
                f"Runtime environment creation failed: {_tail(log)}"
            )

    @staticmethod
    def _install_package(
        definition: RuntimeDefinition, destination: Path, env: dict[str, str]
    ) -> None:
        """Install the pinned worker and its dependencies using supplied wheels."""
        log = destination / "install.log"
        logger.info(
            "Installing runtime package for %s (downloads can take several minutes)",
            definition.name,
        )
        command = [
            find_uv_bin(),
            "--no-config",
            "pip",
            "install",
            "--python",
            str(destination / "venv/bin/python"),
            "--default-index",
            "https://pypi.org/simple",
        ]
        for index in definition.data.get("indexes", []):
            command.extend(["--index", index])
        command.append(definition.data["package"])
        command.extend(
            str(contained(definition.path.parent, wheel["source"]))
            for wheel in definition.data.get("wheels", [])
        )
        code = _run(
            command,
            log,
            3600,
            env=env,
            progress=f"Installing dependencies for {definition.name}",
        )
        if code:
            raise ConfigurationError(
                f"Runtime dependency installation failed: {_tail(log)}"
            )

    def _validate_installation(
        self, definition: RuntimeDefinition, destination: Path
    ) -> None:
        """Check every operation before marking the installation ready for use."""
        logger.info("Checking runtime %s", definition.name)
        handshake = {}
        for operation in definition.data["operations"]:
            handshake = self._execute(
                definition,
                operation,
                {},
                destination / ".validation",
                handshake=True,
            )
        temporary = destination / "ready.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "identity": definition.identity,
                    "id": definition.name,
                    "interpreter": handshake["payload"],
                }
            )
        )
        temporary.replace(destination / "ready.json")
        logger.info("Runtime %s is ready", definition.name)

    def _execute(
        self,
        definition: RuntimeDefinition,
        operation: str,
        payload: dict[str, Any],
        output_dir: Path,
        *,
        timeout: float = 3600,
        handshake: bool = False,
    ) -> dict[str, Any]:
        if operation not in definition.data["operations"]:
            raise ConfigurationError(
                f"Runtime {definition.name} does not provide {operation}"
            )
        spec = definition.data["operations"][operation]
        job_id = uuid4().hex
        job = output_dir.resolve() / job_id
        job.mkdir(parents=True)
        runtime_dir = self.location(definition)
        request = {
            "protocol": 1,
            "job_id": job_id,
            "runtime_id": definition.identity,
            "operation": operation,
            "result_schema": spec["result_schema"],
            "entrypoint": spec["entrypoint"],
            "runtime_dir": str(runtime_dir),
            "output_dir": str(job),
            "payload": payload,
            "handshake": handshake,
        }
        request_path, result_path, log = (
            job / "request.json",
            job / "result.json",
            job / "worker.log",
        )
        request_path.write_text(json.dumps(request, allow_nan=False))
        env = _environment()
        env["PATH"] = str(runtime_dir / "venv/bin") + os.pathsep + env.get("PATH", "")
        env["MLIA_RUNTIME_WORKER"] = "1"
        try:
            code = _run(
                [
                    str(runtime_dir / "venv/bin/python"),
                    "-I",
                    str(Path(__file__).with_name("worker.py")),
                    str(request_path),
                    str(result_path),
                ],
                log,
                timeout,
                env=env,
                cwd=job,
                progress=f"Checking runtime {definition.name}" if handshake else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise ConfigurationError(
                f"Runtime {definition.name} timed out; see {log}"
            ) from exc
        return self._read_result(definition, request, job, code)

    @staticmethod
    def _read_result(
        definition: RuntimeDefinition, request: dict[str, Any], job: Path, code: int
    ) -> dict[str, Any]:
        """Validate the response envelope and locate its declared output files."""
        result_path = job / "result.json"
        log = job / "worker.log"
        operation = request["operation"]
        try:
            result = json.loads(result_path.read_text())
            for key in (
                "protocol",
                "job_id",
                "runtime_id",
                "operation",
                "result_schema",
            ):
                if result[key] != request[key]:
                    raise ValueError(f"Mismatched runtime result {key}")
            if code or result.get("status") != "success":
                error = result.get("error", {})
                if not isinstance(error, dict):
                    raise ValueError("Invalid runtime error payload")
                message = error.get("message", _tail(log))
                raise ConfigurationError(
                    f"Runtime {definition.name} operation "
                    f"{operation} failed: {message}. Log: {log}"
                )
            if not isinstance(result["payload"], dict) or not isinstance(
                result["artifacts"], list
            ):
                raise ValueError("Invalid runtime result payload/artifacts")
            for artifact in result["artifacts"]:
                path = Path(artifact["path"])
                path = (job / path).resolve()
                if path in {
                    job / "request.json",
                    result_path,
                    result_path.with_suffix(".tmp"),
                    log,
                }:
                    raise ConfigurationError(
                        f"Worker artifact uses a reserved protocol file: {path}"
                    )
                if not path.is_relative_to(job) or not path.is_file():
                    raise ConfigurationError(
                        f"Worker artifact is missing or outside its job: {path}"
                    )
                artifact["path"] = str(path)
                artifact["sha256"] = digest(path)
            return cast(dict[str, Any], result)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise ConfigurationError(
                f"Invalid runtime result from {definition.name}: {exc}. Log: {log}"
            ) from exc
