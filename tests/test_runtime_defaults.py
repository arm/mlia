# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Check backend installation and selection through their public API."""

import json
import platform
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from mlia.api import get_advice, install_backends, list_backends, uninstall_backends
from mlia.backend.config import BackendConfiguration, BackendType
from mlia.backend.manager import ensure_backends_installed
from mlia.backend.registry import ensure_backend_plugins_loaded, registry
from mlia.core.common import AdviceCategory
from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import (
    ExecutionProfile,
    RuntimeDefinition,
    load_catalog,
    load_catalog_variants,
    load_profile,
)
from mlia.runtime.local_install import load_local_runtime
from mlia.runtime.manager import RuntimeManager
from mlia.runtime.session import profile_backends, select_profile
from mlia.target.config import TargetInfo, TargetProfile
from mlia.target.registry import registry as targets
from tests.test_runtime_execution import runtime


@pytest.fixture
def managed_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Register a migrated backend with a catalog scoped to the test."""
    ensure_backend_plugins_loaded()
    monkeypatch.setitem(
        registry.items,
        "managed-tool",
        BackendConfiguration(
            supported_advice=[AdviceCategory.PERFORMANCE],
            supported_systems=None,
            backend_type=BackendType.CUSTOM,
            installation=None,
            runtime_operation="echo",
        ),
    )
    monkeypatch.setenv("MLIA_RUNTIME_CATALOG", str(tmp_path))
    monkeypatch.setenv("MLIA_RUNTIME_HOME", str(tmp_path / "installed"))
    monkeypatch.delenv("MLIA_EXECUTION_PROFILE", raising=False)


def default_profile(
    tmp_path: Path, definition: RuntimeDefinition, suffix: str = "vgf"
) -> Path:
    """Publish the supplied runtime as the default for one input format."""
    manifest = definition.path.with_name("definition.runtime.json")
    definition.path.rename(manifest)
    path = tmp_path / f"default-{suffix}.profile.toml"
    path.write_text(f'''schema_version = 1
id = "test/{suffix}"
backend = "managed-tool"
default = true
inputs = ["{suffix}"]
[stages.estimate]
operation = "echo"
runtime = "{definition.name}"
''')
    return path


def test_missing_default_does_not_fall_back_to_legacy(managed_backend: None) -> None:
    """Missing runtime delivery must not invoke the old installer."""
    with pytest.raises(ConfigurationError, match="runtime catalog"):
        select_profile(None, ["managed-tool"], "model.vgf")


@pytest.mark.parametrize("by_name", [False, True])
def test_check_keeps_profile_and_runtime_selection_until_next_call(
    managed_backend: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    by_name: bool,
) -> None:
    """A catalog edit during backend validation takes effect on the next check."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    selection = default_profile(tmp_path, definition)
    original = selection.read_text() + '\n[stages.estimate.options]\nvalue = "before"\n'
    selection.write_text(original)

    class EditingTarget(TargetProfile):
        def verify(self) -> None:
            super().verify()
            selection.write_text(original.replace('"before"', '"after"'))
            updated = dict(definition.data)
            updated["description"] = "Updated runtime"
            (definition.path.parent / "definition.runtime.json").write_text(
                json.dumps(updated)
            )

    class PreflightComplete(Exception):
        pass

    selections = []

    def validate(
        profile: ExecutionProfile,
        model: str | Path | None,
        categories: set[AdviceCategory],
        config: TargetProfile,
    ) -> None:
        selections.append(
            (
                profile.data["stages"]["estimate"]["options"]["value"],
                profile.stages["estimate"][1].data.get("description"),
            )
        )
        raise PreflightComplete

    monkeypatch.setitem(
        targets.items,
        "editing-target",
        TargetInfo(
            supported_backends=["managed-tool"],
            default_backends=["managed-tool"],
            target_profile_cls=EditingTarget,
            advisor_factory_func=lambda *a, **kw: pytest.fail("Preflight only"),
            runtime_validator=validate,
        ),
    )
    target = tmp_path / "target.toml"
    target.write_text('target = "editing-target"\n')
    for _ in range(2):
        with pytest.raises(PreflightComplete):
            get_advice(
                str(target),
                "model.vgf",
                {"performance"},
                execution_profile="test/vgf" if by_name else selection,
            )
    assert selections == [("before", None), ("after", "Updated runtime")]


@pytest.mark.parametrize("selection", [None, "test/vgf"])
@pytest.mark.parametrize("model", ["MODEL.VGF", None])
def test_profile_only_requires_providers_for_selected_input(
    managed_backend: None, tmp_path: Path, selection: str | None, model: str | None
) -> None:
    """A shared profile requires its converter only for inputs needing conversion."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    path = default_profile(tmp_path, definition)
    path.write_text(
        path.read_text().replace('inputs = ["vgf"]', "")
        + '\n[stages.convert]\ninputs = ["pt2"]\n'
        'operation = "convert"\nruntime = "test/converter@1"\n'
    )

    selected = select_profile(selection, ["managed-tool"], model)
    assert selected is not None
    assert list(selected.stages) == ["estimate"]
    assert profile_backends("test/vgf", None, model=model) == ["managed-tool"]

    with pytest.raises(ConfigurationError, match="Unknown runtime test/converter@1"):
        select_profile(selection, ["managed-tool"], "model.pt2")


def test_runtime_listing_reports_invalid_catalog_without_traceback(
    managed_backend: None, tmp_path: Path
) -> None:
    """Invalid user catalogs produce a normal CLI error and a nonzero exit."""
    from typer.testing import CliRunner

    from mlia.cli.commands import mlia_app

    (tmp_path / "broken.runtime.json").write_text("{}")
    result = CliRunner().invoke(mlia_app, ["backend", "list", "--runtimes"])
    assert result.exit_code == 1
    assert "Error: Invalid runtime definition" in result.output
    assert isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("source", ["catalog", "local", "automatic"])
def test_backend_alias_installs_lists_and_removes_default(
    managed_backend: None, tmp_path: Path, source: str
) -> None:
    """Ordinary backend commands share the published runtime selection."""

    definition = runtime(tmp_path, "def run(request):\n    return {}, []\n")
    default_profile(tmp_path, definition)
    assert select_profile(None, ["managed-tool"], "MODEL.VGF") is not None
    if source == "automatic":
        ensure_backends_installed(["managed-tool"], accept_eula=False)
    else:
        local = definition.path.parent if source == "local" else None
        install_backends(["MANAGED-TOOL"], path=local, noninteractive=True)
    manager = RuntimeManager()
    assert manager.ready(definition)
    marker = manager.location(definition) / "previous-installation"
    marker.touch()
    install_backends(["managed-tool"], noninteractive=True, force=True)
    assert manager.ready(definition) and not marker.exists()
    item = next(item for item in list_backends() if item["name"] == "managed-tool")
    assert item["installed"]
    uninstall_backends(["managed-tool"])
    assert not RuntimeManager().ready(definition)


def test_backend_alias_rejects_conflicting_defaults(
    managed_backend: None, tmp_path: Path
) -> None:
    """Different primary runtime defaults require an explicit choice."""

    for suffix in ("vgf", "tosa"):
        definition = runtime(tmp_path, "def run(request):\n    return {}, []\n", suffix)
        default_profile(tmp_path, definition, suffix)
    with pytest.raises(ConfigurationError, match="Multiple default runtimes"):
        install_backends(["managed-tool"], noninteractive=True)
    assert not (tmp_path / "installed").exists()


def test_local_bundle_survives_source_removal(
    managed_backend: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    definition = runtime(tmp_path, "def run(request): return {'ok': True}, []\n")
    install_backends([definition.name], path=definition.path, noninteractive=True)
    shutil.rmtree(definition.path.parent)
    monkeypatch.delenv("MLIA_RUNTIME_CATALOG")
    installed = load_catalog()[definition.name]
    profile = tmp_path / "custom.profile.toml"
    profile.write_text(
        'schema_version = 1\nid = "custom"\n[stages.run]\n'
        f'operation = "echo"\nruntime = "{definition.name}"\n'
    )
    assert load_profile(profile).stages["run"][1].identity == definition.identity
    manager = RuntimeManager()
    assert manager.execute(installed, "echo", {}, tmp_path / "jobs")["payload"] == {
        "ok": True
    }
    install_backends([definition.name], noninteractive=True, force=True)
    uninstall_backends([definition.name])
    assert not manager.ready(installed)
    # Keep the registered definition available for automatic reinstallation.
    install_backends([definition.name], noninteractive=True)
    assert manager.ready(installed)


def test_local_runtime_requires_the_requested_version(
    managed_backend: None, tmp_path: Path
) -> None:
    """Aliases select their default; another version needs its qualified runtime ID."""
    source = "def run(request): return {}, []\n"
    default_profile(tmp_path, runtime(tmp_path, source))
    definition = runtime(tmp_path, source, "partner")
    for name in ("test/other@1", "managed-tool"):
        with pytest.raises(ConfigurationError, match="does not match"):
            install_backends([name], path=definition.path, noninteractive=True)
        assert not (tmp_path / "installed").exists()
    install_backends([definition.name], path=definition.path, noninteractive=True)
    assert RuntimeManager().ready(definition)


def test_local_registration_failure_can_be_retried_without_reinstalling(
    managed_backend: None, tmp_path: Path
) -> None:
    """A catalog error explains the partial success and leaves installation reusable."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    manager = RuntimeManager()
    manager.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    catalog = manager.root / ".catalog"
    catalog.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="installed.*registration failed"):
        install_backends([definition.name], path=definition.path, noninteractive=True)
    assert manager.ready(definition)
    assert not list(outside.iterdir())
    marker = manager.location(definition) / "preserved-installation"
    marker.touch()
    catalog.unlink()
    install_backends([definition.name], path=definition.path, noninteractive=True)
    assert marker.exists()
    assert load_catalog()[definition.name].identity == definition.identity


def test_local_install_allows_another_architecture(
    managed_backend: None, tmp_path: Path
) -> None:

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    data = definition.data
    (definition.path.parent / "foreign.runtime.json").write_text(
        json.dumps({**data, "architecture": "foreign"})
    )
    data["architecture"] = platform.machine()
    definition.path.write_text(json.dumps(data))
    definition = RuntimeDefinition.load(definition.path)
    install_backends([definition.name], path=definition.path, noninteractive=True)
    assert RuntimeManager().ready(definition)
    assert (definition.name, "foreign") in load_catalog_variants()
    assert (definition.name, platform.machine()) in load_catalog_variants()


def test_mixed_backend_install_and_uninstall(
    managed_backend: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    default_profile(tmp_path, definition)
    manager = Mock()
    monkeypatch.setattr("mlia.api.get_installation_manager", Mock(return_value=manager))
    install_backends(
        ["managed-tool", "legacy-tool"], noninteractive=True, accept_eula=True
    )
    assert RuntimeManager().ready(definition)
    manager.download_and_install.assert_called_once_with(["legacy-tool"], True, False)
    uninstall_backends(["managed-tool", "legacy-tool"])
    assert not RuntimeManager().ready(definition)
    manager.uninstall.assert_called_once_with(["legacy-tool"])


def test_legacy_local_install_forwards_consent(
    managed_backend: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    manager = Mock()
    monkeypatch.setattr("mlia.api.get_installation_manager", Mock(return_value=manager))
    install_backends(["legacy-tool"], path=tmp_path, noninteractive=True)
    manager.install_from.assert_called_once_with(
        tmp_path, "legacy-tool", False, eula_agreement=False
    )


def test_nonexistent_local_path_is_not_replaced_by_sibling_bundle(
    managed_backend: None, tmp_path: Path
) -> None:

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    default_profile(tmp_path, definition)
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_local_runtime(tmp_path / "public/missing.tar.gz")


def test_profile_runtime_uses_provider_wheel_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default bindings follow their own provider, including dev/local versions."""

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    provider_version = "0.4.0.dev3+abc.dirty"
    data = definition.data.copy()
    data["id"] = "test/tool/public@" + provider_version
    data["provider_package"] = "test-provider"
    definition.path.write_text(json.dumps(data))
    concrete = RuntimeDefinition.load(definition.path)
    data["id"] = "test/tool/public@{version}"
    definition.path.write_text(json.dumps(data))
    metadata = tmp_path / "test_provider.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Name: test-provider\nVersion: {provider_version}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    resolved = RuntimeDefinition.load(definition.path)
    assert resolved.identity == concrete.identity
    profile = tmp_path / "provider.profile.toml"
    profile.write_text(
        'schema_version = 1\nid = "test/provider"\n'
        '[stages.run]\noperation = "echo"\n'
        'runtime = "test/tool/public@{version}"\n'
        'provider_package = "test-provider"\n'
    )
    selected = load_profile(profile, {resolved.name: resolved})
    assert selected.describe()["stages"][0]["runtime"] == concrete.name


def test_backend_inventory_respects_runtime_host_support(
    managed_backend: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    default_profile(tmp_path, definition)
    monkeypatch.setattr("mlia.runtime.manager.sys.platform", "darwin")
    info = next(item for item in list_backends() if item["name"] == "managed-tool")
    assert not info["installed"]
    assert not info["could_be_installed"]


@pytest.mark.parametrize("inputs", [None, []])
@pytest.mark.parametrize("model", [None, "model.vgf"])
def test_unrestricted_default_profile_matches_model_and_profiling_inputs(
    managed_backend: None, tmp_path: Path, inputs: list[str] | None, model: str | None
) -> None:
    """An unrestricted default uses the same input policy as explicit selection."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    path = default_profile(tmp_path, definition)
    text = path.read_text().replace(
        'inputs = ["vgf"]', "inputs = []" if inputs is not None else ""
    )
    path.write_text(text)
    selected = select_profile(None, ["managed-tool"], model)
    assert selected is not None and selected.path == path


@pytest.mark.parametrize("existing_architecture", [None, platform.machine()])
def test_local_install_rejects_overlapping_host_variant(
    managed_backend: None, tmp_path: Path, existing_architecture: str | None
) -> None:
    """A universal bundle must not replace a different host-specific environment."""
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    existing_data = dict(definition.data)
    if existing_architecture:
        existing_data["architecture"] = existing_architecture
    definition.path.write_text(json.dumps(existing_data))
    manifest = definition.path.with_name("existing.runtime.json")
    definition.path.rename(manifest)
    existing = RuntimeDefinition.load(manifest)
    manager = RuntimeManager()
    manager.install(existing)
    marker = manager.location(existing) / "preserved"
    marker.touch()
    supplied = manifest.parent / "replacement.json"
    incoming_data = dict(definition.data)
    if existing_architecture is None:
        incoming_data["architecture"] = platform.machine()
    supplied.write_text(json.dumps(incoming_data))
    with pytest.raises(ConfigurationError, match="Conflicting runtime definition"):
        install_backends([existing.name], path=supplied, noninteractive=True)
    assert manager.ready(existing)
    assert marker.exists()


def test_plan_preview_validates_resolved_options_without_installing(
    managed_backend: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview validates resolved options without creating runtime environments."""
    from typing import Any

    from mlia.check import CheckRequest, CheckSettings, execute_check
    from mlia.plugins.analysis import AnalysisPluginRegistry
    from mlia.target.config import TargetInfo, TargetProfile
    from mlia.target.registry import registry as targets

    class PreviewTarget(TargetProfile):
        def __init__(self, target: str, backend_config: dict | None = None, **_: Any):
            super().__init__(target, backend_config)

        def verify(self) -> None:
            super().verify()

    validated = []
    monkeypatch.setitem(
        targets.items,
        "preview-target",
        TargetInfo(
            supported_backends=["managed-tool"],
            default_backends=["managed-tool"],
            advisor_factory_func=lambda *a, **kw: pytest.fail(
                "Preview must not run analysis"
            ),
            target_profile_cls=PreviewTarget,
            runtime_validator=lambda p, m, c, config: validated.append(
                config.backend_config
            ),
        ),
    )
    target_path = tmp_path / "target.toml"
    target_path.write_text(
        'target = "preview-target"\n[backend.managed-tool]\ncount = 1\n'
    )
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    selection = default_profile(tmp_path, definition)
    result = execute_check(
        CheckRequest(
            target_profile=str(target_path),
            model="model.vgf",
            performance=True,
            execution_profile=selection,
            show_plan=True,
            output_dir=tmp_path / "output",
            settings=CheckSettings(backend_options={"managed-tool": {"count": 2}}),
        ),
        analysis_plugin_registry=AnalysisPluginRegistry(),
    )
    assert json.loads(result.report)["profile"] == "test/vgf"
    assert validated == [{"managed-tool": {"count": 2}}]
    assert not (tmp_path / "output").exists()
    assert not RuntimeManager().root.exists()
