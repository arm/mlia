# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Exercise catalog selection, package isolation and file-based worker results."""

import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import pytest

from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import RuntimeDefinition, load_profile
from mlia.runtime.manager import RuntimeManager


def wheel_file(
    directory: Path,
    name: str,
    version: str,
    files: dict[str, str],
    dependencies: tuple[str, ...] = (),
) -> Path:
    """Make a small real wheel to exercise installation without network dependencies."""
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w") as stream:
        for filename, content in files.items():
            stream.writestr(filename, content)
        stream.writestr(
            f"{metadata}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {dep}\n" for dep in dependencies),
        )
        stream.writestr(
            f"{metadata}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        stream.writestr(f"{metadata}/RECORD", "")
    return wheel


def supplied_wheel(path: Path) -> dict[str, str]:
    """Describe one local wheel."""
    return {
        "source": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def runtime(tmp_path: Path, source: str, variant: str = "public") -> RuntimeDefinition:
    """Create a dependency-free, locally supplied worker package."""
    bundle = tmp_path / variant
    bundle.mkdir()
    wheel = wheel_file(
        bundle,
        "test_worker",
        "1",
        {"worker.py": source, "EULA.txt": "Full test license"},
    )
    manifest = bundle / "runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": f"test/echo/{variant}@1",
                "python": sys.executable,
                "platform": "linux",
                "package": "test-worker==1",
                "wheels": [supplied_wheel(wheel)],
                "operations": {
                    "echo": {"entrypoint": "worker:run", "result_schema": "echo/1"}
                },
            }
        )
    )
    return RuntimeDefinition.load(manifest)


def test_runtime_isolated_and_reused(tmp_path: Path) -> None:
    definition = runtime(
        tmp_path,
        "import sys\ndef run(request):\n"
        "    return {'prefix': sys.prefix, 'base': sys.base_prefix}, []\n",
    )
    manager = RuntimeManager(tmp_path / "installed")
    installed = manager.install(definition)
    assert installed == manager.root / definition.name / platform.machine()
    result = manager.execute(definition, "echo", {}, tmp_path / "jobs")
    assert result["payload"]["prefix"] == str(installed / "venv")
    assert result["payload"]["prefix"] != sys.prefix
    assert result["payload"]["base"] != result["payload"]["prefix"]
    ready = (installed / "ready.json").stat().st_mtime_ns
    assert manager.install(definition) == installed
    assert (installed / "ready.json").stat().st_mtime_ns == ready


def test_variants_do_not_share_installations(tmp_path: Path) -> None:
    source = "def run(request):\n    return {}, []\n"
    public = runtime(tmp_path, source)
    partner = runtime(tmp_path, source, "partner")
    manager = RuntimeManager(tmp_path / "installed")
    assert manager.install(public) != manager.install(partner)
    manager.remove(public)
    assert not manager.ready(public)
    assert manager.execute(partner, "echo", {}, tmp_path / "jobs")["payload"] == {}


def test_profile_selected_worker_processes_input_file(tmp_path: Path) -> None:
    """A profile binds an operation whose payload and output survive the boundary."""
    definition = runtime(
        tmp_path,
        """from pathlib import Path

def run(request):
    payload = request['payload']
    text = Path(payload['model']).read_text() * payload['options']['copies']
    output = Path(request['output_dir']) / 'processed.txt'
    output.write_text(text)
    return {'characters': len(text)}, [{'path': output.name, 'kind': 'model'}]
""",
    )
    profile_path = tmp_path / "process.profile.toml"
    profile_path.write_text(
        'schema_version = 1\nid = "test/process"\n'
        '[stages.process]\noperation = "echo"\n'
        f'runtime = "{definition.name}"\n'
        "[stages.process.options]\noptions = { copies = 2 }\n"
    )
    manager = RuntimeManager(tmp_path / "installed")
    profile = load_profile(profile_path, {definition.name: definition})
    assert not manager.root.exists()
    operation, selected = profile.stages["process"]
    model = tmp_path / "input.txt"
    model.write_text("example input\n")
    payload = {"model": str(model), **profile.data["stages"]["process"]["options"]}
    manager.install(selected)
    result = manager.execute(selected, operation, payload, tmp_path / "jobs")
    expected = "example input\nexample input\n"
    assert result["result_schema"] == "echo/1"
    assert result["payload"] == {"characters": len(expected)}
    artifact = result["artifacts"][0]
    output = Path(artifact["path"])
    assert output.is_relative_to(tmp_path / "jobs")
    assert output.read_text() == expected
    assert artifact["sha256"] == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize(
    ("previous_install", "save_log"), [(False, True), (True, True), (True, False)]
)
def test_failed_install_preserves_working_runtime_and_log(
    tmp_path: Path, previous_install: bool, save_log: bool
) -> None:
    """Retries keep diagnostics and any working runtime without partial environments."""
    definition = runtime(tmp_path, "def run(request): return {'value': 42}, []\n")
    manager = RuntimeManager(tmp_path / "installed")
    if previous_install:
        manager.install(definition)
    wheel = wheel_file(
        definition.path.parent,
        "test_worker",
        "1",
        {
            "worker.py": "print('Preflight diagnostic', flush=True)\n"
            "raise RuntimeError('broken replacement')\n"
        },
    )
    data = {**definition.data, "wheels": [supplied_wheel(wheel)]}
    definition.path.write_text(json.dumps(data))
    replacement = RuntimeDefinition.load(definition.path)
    location = manager.location(definition)
    log = location.with_name(f"{location.name}.failed.log")
    if not save_log:
        log.mkdir()  # An unusable diagnostic destination must not prevent rollback.
    for _ in range(2):
        with pytest.raises(ConfigurationError, match="broken replacement"):
            manager.install(replacement, force=True)
        assert not manager.ready(replacement)
        assert location.exists() == previous_install
        if save_log:
            assert "broken replacement" in log.read_text()
            assert "Preflight diagnostic" in log.read_text()
        assert set(location.parent.iterdir()) == (
            {location, log} if previous_install else {log}
        )
    if previous_install:
        assert manager.ready(definition)
        assert manager.execute(definition, "echo", {}, tmp_path / "jobs")[
            "payload"
        ] == {"value": 42}


@pytest.mark.parametrize("entrypoint", ["foo.:run", "foo..bar:run", "foo.1bar:run"])
def test_invalid_entrypoint_rejected_before_provisioning(
    tmp_path: Path, entrypoint: str
) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    data = definition.data
    data["operations"]["echo"]["entrypoint"] = entrypoint
    definition.path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="Invalid operation entrypoint"):
        RuntimeDefinition.load(definition.path)


@pytest.mark.parametrize("default", ['"false"', "1", "[]"])
def test_profile_default_requires_boolean(tmp_path: Path, default: str) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    profile = tmp_path / "default.profile.toml"
    profile.write_text(
        'schema_version = 1\nid = "test"\n'
        f'default = {default}\n[stages.echo]\noperation = "echo"\n'
        f'runtime = "{definition.name}"\n'
    )
    with pytest.raises(ConfigurationError, match="default must be a boolean"):
        load_profile(profile, {definition.name: definition})


def test_changed_artifact_is_rejected_before_install(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request):\n    return {}, []\n")
    (definition.path.parent / definition.data["wheels"][0]["source"]).write_bytes(
        b"changed"
    )
    manager = RuntimeManager(tmp_path / "installed")
    with pytest.raises(ConfigurationError, match="hash"):
        manager.install(definition)
    assert not list((tmp_path / "installed").glob("*/ready.json"))


@pytest.mark.parametrize(
    "source, message",
    [
        (
            "def run(request):\n    raise ValueError('incompatible representation')\n",
            "incompatible representation",
        ),
        (
            "def run(request):\n"
            "    return {}, [{'path': '/etc/passwd', 'kind': 'model'}]\n",
            "outside",
        ),
        (
            "from pathlib import Path\ndef run(request):\n"
            "    path = Path(request['output_dir']) / 'loop'\n"
            "    path.symlink_to(path)\n"
            "    return {}, [{'path': str(path), 'kind': 'model'}]\n",
            "Invalid runtime result|Worker artifact is missing",
        ),
    ],
)
def test_failed_or_escaping_worker_result_is_rejected(
    tmp_path: Path, source: str, message: str
) -> None:
    definition = runtime(tmp_path, source)
    manager = RuntimeManager(tmp_path / "installed")
    manager.install(definition)
    with pytest.raises(ConfigurationError, match=message):
        manager.execute(definition, "echo", {}, tmp_path / "jobs")


def test_profile_requires_known_runtime_without_provisioning(tmp_path: Path) -> None:
    profile = tmp_path / "path.toml"
    profile.write_text(
        "schema_version = 1\nid = "
        '"test/path"\n[stages.estimate]\noperation = "echo"\nruntime = "unknown@1"\n'
    )
    with pytest.raises(ConfigurationError, match="Unknown runtime"):
        load_profile(profile, {})


@pytest.mark.parametrize(
    "source,message",
    [
        ("import os\ndef run(request):\n    os._exit(0)\n", "Invalid runtime result"),
        ("import time\ndef run(request):\n    time.sleep(60)\n", "timed out"),
    ],
)
def test_worker_exit_and_timeout_are_actionable(
    tmp_path: Path, source: str, message: str
) -> None:
    definition = runtime(tmp_path, source)
    manager = RuntimeManager(tmp_path / "installed")
    manager.install(definition)
    with pytest.raises(ConfigurationError, match=message):
        manager.execute(definition, "echo", {}, tmp_path / "jobs", timeout=0.2)
    manager.remove(definition)
    assert not manager.ready(definition)


@pytest.mark.parametrize("outcome", ["timeout", "success", "failure"])
def test_worker_terminates_children_even_when_worker_exits(
    tmp_path: Path, outcome: str
) -> None:
    """A child ignoring TERM must not survive its supervising worker."""
    import os
    import signal
    import time

    definition = runtime(
        tmp_path,
        """import subprocess, sys, time
from pathlib import Path

def run(request):
    child = subprocess.Popen([sys.executable, '-c',
        'import signal,time;'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
    Path(request['output_dir'], 'child.pid').write_text(str(child.pid))
    OUTCOME
""".replace(
            "OUTCOME",
            {
                "timeout": "time.sleep(60)",
                "success": "return {}, []",
                "failure": "raise ValueError('worker failed')",
            }[outcome],
        ),
    )
    manager = RuntimeManager(tmp_path / "installed")
    manager.install(definition)
    if outcome == "success":
        manager.execute(definition, "echo", {}, tmp_path / "jobs", timeout=1)
    else:
        with pytest.raises(
            ConfigurationError,
            match="timed out" if outcome == "timeout" else "worker failed",
        ):
            manager.execute(definition, "echo", {}, tmp_path / "jobs", timeout=1)
    pid = int(next((tmp_path / "jobs").rglob("child.pid")).read_text())
    status = Path(f"/proc/{pid}/stat")
    try:
        for _ in range(100):
            try:
                if status.read_text().split()[2] == "Z":
                    break
            except (FileNotFoundError, ProcessLookupError):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Runtime child survived timeout cleanup")
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_conflicting_dependencies_are_isolated(tmp_path: Path) -> None:
    """Two runtime packages can depend on incompatible versions of the same package."""
    manager = RuntimeManager(tmp_path / "installed with spaces")
    for version in ("1", "2"):
        source = (
            "import local_runtime_dependency\ndef run(request):\n"
            "    return {'value': local_runtime_dependency.VALUE}, []\n"
        )
        definition = runtime(tmp_path, source, f"v{version}")
        bundle = definition.path.parent
        worker = wheel_file(
            bundle,
            "test_worker",
            "1",
            {"worker.py": source},
            (f"local-runtime-dependency=={version}",),
        )
        dependency = wheel_file(
            bundle,
            "local_runtime_dependency",
            version,
            {"local_runtime_dependency.py": f"VALUE = {version}\n"},
        )
        definition.data["wheels"] = [supplied_wheel(worker), supplied_wheel(dependency)]
        definition.path.write_text(json.dumps(definition.data))
        definition = RuntimeDefinition.load(definition.path)
        installed = manager.install(definition)
        assert manager.execute(definition, "echo", {}, tmp_path / "jobs")[
            "payload"
        ] == {"value": int(version)}
        inventory = json.loads((installed / "ready.json").read_text())["interpreter"][
            "packages"
        ]
        assert inventory["local_runtime_dependency"] == version
    import importlib.util

    assert importlib.util.find_spec("local_runtime_dependency") is None


def test_wheel_license_is_available_before_installation(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    definition.data["license"] = {
        "wheel": definition.data["wheels"][0]["source"],
        "path": "EULA.txt",
    }
    definition.path.write_text(json.dumps(definition.data))
    definition = RuntimeDefinition.load(definition.path)
    manager = RuntimeManager(tmp_path / "installed")
    assert definition.license_text() == "Full test license"
    with pytest.raises(ConfigurationError, match="EULA acceptance"):
        manager.install(definition)
    manager.install(definition, accept_eula=True)
    assert manager.ready(definition)


@pytest.mark.parametrize("installed", [False, True])
def test_missing_runtime_identifies_provider_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: bool
) -> None:
    """Explain both an absent provider and an installed incompatible provider."""
    import importlib.metadata

    def version(name: str) -> str:
        assert name == "test-converter"
        if not installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return "0.1"

    monkeypatch.setattr(importlib.metadata, "version", version)
    profile = tmp_path / "path.toml"
    profile.write_text(
        'schema_version = 1\nid = "test/path"\n'
        '[stages.convert]\noperation = "convert"\nruntime = "test/runtime@1"\n'
        'provider_package = "test-converter"\n'
    )
    with pytest.raises(ConfigurationError) as error:
        load_profile(profile, {})
    message = str(error.value)
    assert "test-converter" in message
    assert "test/path" in message
    assert "test/runtime@1" in message
    assert "compatible release" in message if installed else "not installed" in message


@pytest.mark.parametrize("by_name", [False, True])
@pytest.mark.parametrize(
    "contents",
    [
        'schema_version = 1\nid = "test/path"\nstages = []\n',
        'schema_version = 1\nid = "test/path"\n[stages]\nconvert = "invalid"\n',
        'schema_version = 1\nid = "test/path"\nbackend = ["invalid"]\n',
        "schema_version = [\n",
    ],
)
def test_invalid_profile_reports_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_name: bool, contents: str
) -> None:
    """File and catalog selection report invalid structure without tracebacks."""
    from mlia.runtime import catalog

    profile = tmp_path / "invalid.profile.toml"
    profile.write_text(contents)
    monkeypatch.setattr(catalog, "catalog_directories", lambda: [tmp_path])
    with pytest.raises(ConfigurationError, match="Invalid execution profile"):
        load_profile("test/path" if by_name else profile, {})


def test_editable_catalog_prefers_built_definition_over_source_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated worker resources supersede templates, but published conflicts fail."""
    from mlia.runtime import catalog

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    data = definition.data.copy()
    data["provider_package"] = "test-provider"
    published = tmp_path / "published.runtime.json"
    published.write_text(json.dumps(data))
    data["id"] = "test/echo/public@{version}"
    data["python"] = "3.12"  # Previous source template inputs.
    template = tmp_path / "source.runtime.json"
    template.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "catalog_directories", lambda: [tmp_path])
    monkeypatch.setattr(catalog.importlib.metadata, "version", lambda _: "1")
    assert catalog.load_catalog()[definition.name].path == published
    # An additional concrete publication with changed contents remains an error.
    data["id"] = definition.name
    (tmp_path / "conflicting.runtime.json").write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="Conflicting runtime definition"):
        catalog.load_catalog()


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_catalog_selects_host_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """One runtime identity may supply separate artifacts for supported machines."""
    from mlia.runtime import catalog

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    for arch in ("x86_64", "aarch64"):
        data = {**definition.data, "architecture": arch}
        definition.path.with_name(f"{arch}.runtime.json").write_text(json.dumps(data))
    definition.path.unlink()
    monkeypatch.setattr(catalog, "catalog_directories", lambda: [tmp_path])
    monkeypatch.setattr(platform, "machine", lambda: architecture)
    assert catalog.load_catalog()[definition.name].data["architecture"] == architecture


def test_handshake_rejects_noncallable_entrypoint(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "run = 1\n")
    manager = RuntimeManager(tmp_path / "installed")
    with pytest.raises(ConfigurationError, match="callable"):
        manager.install(definition)
    assert not manager.ready(definition)


def test_mutated_definition_cannot_be_installed(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    definition.data["operations"]["echo"]["entrypoint"] = "other:run"
    with pytest.raises(ConfigurationError, match="changed"):
        RuntimeManager(tmp_path / "installed").install(definition)


def test_intermediate_runtime_symlink_is_rejected(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    root = tmp_path / "installed"
    (root / "other").mkdir(parents=True)
    (root / "test").symlink_to(root / "other", target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symlink"):
        RuntimeManager(root).location(definition)


def test_profile_digest_describes_loaded_content(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    path = tmp_path / "profile.toml"
    path.write_text(
        'schema_version = 1\nid = "test"\n[stages.echo]\n'
        'operation = "echo"\nruntime = "test/echo/public@1"\n'
    )
    profile = load_profile(path, {definition.name: definition})
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("changed")
    assert profile.describe()["profile_sha256"] == expected


def test_remove_rejects_stale_definition(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    manager = RuntimeManager(tmp_path / "installed")
    installed = manager.install(definition)
    data = definition.data
    data["python"] = "3.12"
    definition.path.write_text(json.dumps(data))
    stale = RuntimeDefinition.load(definition.path)
    with pytest.raises(ConfigurationError, match="different"):
        manager.remove(stale)
    assert (installed / "ready.json").is_file()


@pytest.mark.parametrize("architecture", [None, platform.machine()])
def test_host_or_universal_catalog_preferred_over_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str | None
) -> None:
    from mlia.runtime.catalog import load_catalog

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    for name, arch in [("a", "foreign"), ("b", architecture)]:
        data = dict(definition.data)
        if arch:
            data["architecture"] = arch
        (definition.path.parent / f"{name}.runtime.json").write_text(json.dumps(data))
    monkeypatch.setattr(
        "mlia.runtime.catalog.catalog_directories", lambda: [definition.path.parent]
    )
    selected = load_catalog()[definition.name]
    assert selected.data.get("architecture") == architecture


def test_runtime_epoch_version(tmp_path: Path) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    data = definition.data
    data["id"] = "test/echo/public@1!2.0"
    definition.path.write_text(json.dumps(data))
    assert RuntimeDefinition.load(definition.path).name == data["id"]


@pytest.mark.parametrize("setting", ['backend = ""\nbackends = ["echo"]', ""])
def test_invalid_profile_options_fail_before_provisioning(
    tmp_path: Path, setting: str
) -> None:
    path = tmp_path / "profile.toml"
    path.write_text(
        'schema_version = 1\nid = "test"\n'
        + setting
        + '\n[stages.echo]\noperation = "echo"\nruntime = "unknown@1"\n'
        'options = { options = "bad" }\n'
    )
    with pytest.raises(ConfigurationError, match="unique backends|nested options"):
        load_profile(path, {})


def test_combined_profile_backend_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlia.runtime.catalog import default_backend_runtime

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    definition.path.rename(definition.path.with_name("definition.runtime.json"))
    (definition.path.parent / "default.profile.toml").write_text(
        'schema_version = 1\nid = "test"\ndefault = true\n'
        'backends = ["first", "second"]\n'
        '[stages.echo]\noperation = "echo"\nruntime = "test/echo/public@1"\n'
    )
    monkeypatch.setattr(
        "mlia.runtime.catalog.catalog_directories", lambda: [definition.path.parent]
    )
    assert default_backend_runtime("second", "echo").identity == definition.identity


@pytest.mark.parametrize("architecture", [[], {}, "", None])
def test_invalid_architecture_is_a_configuration_error(
    tmp_path: Path, architecture: object
) -> None:
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    data = definition.data
    data["architecture"] = architecture
    definition.path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="architecture"):
        RuntimeDefinition.load(definition.path)


def test_host_selection_does_not_hide_universal_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlia.runtime.catalog import load_catalog

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    variants: list[tuple[str, dict[str, str]]] = [
        ("a", {}),
        ("b", {"architecture": platform.machine()}),
        ("c", {"python": "3.13"}),
    ]
    for name, changes in variants:
        (definition.path.parent / f"{name}.runtime.json").write_text(
            json.dumps({**definition.data, **changes})
        )
    monkeypatch.setattr(
        "mlia.runtime.catalog.catalog_directories", lambda: [definition.path.parent]
    )
    with pytest.raises(ConfigurationError, match="Conflicting"):
        load_catalog()


def test_local_registration_retains_multiple_architectures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlia.runtime.catalog import load_catalog_variants
    from mlia.runtime.local_install import register_local_runtime

    monkeypatch.setenv("MLIA_RUNTIME_HOME", str(tmp_path / "store"))
    monkeypatch.delenv("MLIA_RUNTIME_CATALOG", raising=False)
    monkeypatch.setattr("mlia.runtime.catalog.get_mlia_resource_dirs", lambda: [])
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    for architecture in ("x86_64", "aarch64"):
        data = {**definition.data, "architecture": architecture}
        definition.path.write_text(json.dumps(data))
        register_local_runtime(RuntimeDefinition.load(definition.path))
    assert set(load_catalog_variants()) == {
        (definition.name, "x86_64"),
        (definition.name, "aarch64"),
    }


def test_local_registration_rejects_symlinked_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registration must not write through a catalog link outside its runtime store."""
    from mlia.runtime.local_install import register_local_runtime

    store = tmp_path / "store"
    outside = tmp_path / "outside"
    store.mkdir()
    outside.mkdir()
    (store / ".catalog").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("MLIA_RUNTIME_HOME", str(store))
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    with pytest.raises(ConfigurationError, match="catalog.*symlink"):
        register_local_runtime(definition)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("value", [True, 1.0])
def test_schema_version_requires_integer(tmp_path: Path, value: object) -> None:
    """JSON and TOML schema versions cannot use boolean or floating-point aliases."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    definition.data["schema_version"] = value
    definition.path.write_text(json.dumps(definition.data))
    with pytest.raises(ConfigurationError, match="schema"):
        RuntimeDefinition.load(definition.path)
    profile = tmp_path / "invalid.profile.toml"
    profile.write_text(
        f'schema_version = {json.dumps(value)}\nid = "test"\n'
        f'[stages.echo]\noperation = "echo"\nruntime = "{definition.name}"\n'
    )
    with pytest.raises(ConfigurationError, match="schema"):
        load_profile(profile)


def test_identical_profile_copies_can_be_selected_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source and built resources can expose one profile twice, but not two choices."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    monkeypatch.setenv("MLIA_RUNTIME_CATALOG", str(tmp_path))
    text = (
        'schema_version = 1\nid = "test/shared"\n'
        f'[stages.run]\noperation = "echo"\nruntime = "{definition.name}"\n'
        "options = { copies = 1 }\n"
    )
    for directory in ("source", "built"):
        path = tmp_path / directory / "shared.profile.toml"
        path.parent.mkdir()
        path.write_text(text)
    catalog = {definition.name: definition}
    assert load_profile("test/shared", catalog).stages["run"][1] == definition
    path.write_text(text.replace("copies = 1", "copies = 2"))
    with pytest.raises(ConfigurationError, match="ambiguous"):
        load_profile("test/shared", catalog)


@pytest.mark.parametrize(
    "filename", ["request.json", "result.json", "result.tmp", "worker.log"]
)
def test_protocol_files_cannot_be_returned_as_artifacts(
    tmp_path: Path, filename: str
) -> None:
    """Worker outputs must not alias the request, response or process log."""
    definition = runtime(
        tmp_path,
        f"""from pathlib import Path
def run(request):
    path = Path(request['output_dir']) / {filename!r}
    path.write_text('model content')
    return {{}}, [{{'path': str(path), 'kind': 'model'}}]
""",
    )
    manager = RuntimeManager(tmp_path / "installed")
    manager.install(definition)
    with pytest.raises(ConfigurationError, match="reserved protocol file"):
        manager.execute(definition, "echo", {}, tmp_path / "jobs")


def test_removing_uninstalled_runtime_reports_error(tmp_path: Path) -> None:
    """An uninstall request must report when there is nothing to remove."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    with pytest.raises(ConfigurationError, match="not installed"):
        RuntimeManager(tmp_path / "installed").remove(definition)


@pytest.mark.parametrize("wheels", [{}, ""])
def test_malformed_wheel_list_is_rejected_before_installation(
    tmp_path: Path, wheels: object
) -> None:
    """An invalid catalog must not silently fall back to downloading its package."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    definition.path.write_text(json.dumps({**definition.data, "wheels": wheels}))
    with pytest.raises(ConfigurationError, match="wheels must be a list"):
        RuntimeDefinition.load(definition.path)


def test_looping_wheel_source_is_rejected(tmp_path: Path) -> None:
    """A broken supplied wheel is a configuration error on every supported Python."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    wheel = definition.path.parent / definition.data["wheels"][0]["source"]
    wheel.unlink()
    wheel.symlink_to(wheel)
    with pytest.raises(ConfigurationError):
        RuntimeDefinition.load(definition.path).verify_sources()


def test_malformed_worker_error_is_rejected(tmp_path: Path) -> None:
    """Malformed error envelopes must produce an MLIA error rather than a traceback."""
    definition = runtime(
        tmp_path,
        """import json, os
from pathlib import Path
def run(request):
    request.update(status='error', error=None)
    Path(request['output_dir'], 'result.json').write_text(json.dumps(request))
    os._exit(1)
""",
    )
    manager = RuntimeManager(tmp_path / "installed")
    manager.install(definition)
    with pytest.raises(ConfigurationError, match="Invalid runtime result"):
        manager.execute(definition, "echo", {}, tmp_path / "jobs")
