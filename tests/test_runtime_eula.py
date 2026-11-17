# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Check that licence decisions determine whether a backend is installed."""

import json
from pathlib import Path

import pytest

from mlia.api import install_backends
from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import RuntimeDefinition, load_catalog
from mlia.runtime.manager import RuntimeManager
from mlia.runtime.session import provision_runtimes
from tests.test_runtime_execution import runtime


@pytest.fixture
def licensed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> RuntimeDefinition:
    """Supply a real worker wheel with its full licence and an isolated store."""
    monkeypatch.setenv("MLIA_RUNTIME_HOME", str(tmp_path / "installed"))
    definition = runtime(tmp_path, "def run(request): return {}, []\n")
    definition.data["license"] = {
        "wheel": definition.data["wheels"][0]["source"],
        "path": "EULA.txt",
    }
    definition.path.write_text(json.dumps(definition.data))
    return RuntimeDefinition.load(definition.path)


@pytest.mark.parametrize("response", ["yes", "no", "quit", "eof"])
def test_terminal_consent_controls_installation(
    licensed_runtime: RuntimeDefinition,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    response: str,
) -> None:
    """Display full terms, retry invalid answers, and honour refusal."""
    manager = RuntimeManager()
    monkeypatch.setattr("pydoc.pager", print)
    answers = iter(["maybe", response])

    def answer(message: str) -> str:
        assert not manager.ready(licensed_runtime)
        print(message)
        value = next(answers)
        if value == "eof":
            raise EOFError
        return value

    monkeypatch.setattr("builtins.input", answer)
    if response == "yes":
        provision_runtimes([licensed_runtime], accept_eula=None)
    else:
        with pytest.raises(ConfigurationError, match="declined|ended"):
            provision_runtimes([licensed_runtime], accept_eula=None)
    text = capsys.readouterr().out
    assert text.index(licensed_runtime.license_text()) < text.index("[yes/no]")
    assert manager.ready(licensed_runtime) == (response == "yes")


def test_callback_consent_and_reinstallation(
    licensed_runtime: RuntimeDefinition,
) -> None:
    """Show full terms once; require consent again when replacing the installation."""
    manager = RuntimeManager()
    prompts = []

    def accept(text: str) -> str:
        assert not manager.ready(licensed_runtime)
        prompts.append(text)
        return "yes"

    provision_runtimes([licensed_runtime], accept_eula=None, prompt=accept)
    assert manager.ready(licensed_runtime)
    assert licensed_runtime.license_text() in prompts[0]
    provision_runtimes([licensed_runtime], accept_eula=None, prompt=accept)
    assert len(prompts) == 1
    with pytest.raises(ConfigurationError, match="EULA acceptance required"):
        provision_runtimes([licensed_runtime], accept_eula=False, force=True)
    assert manager.ready(licensed_runtime)
    marker = manager.location(licensed_runtime) / "old-installation"
    marker.touch()
    provision_runtimes([licensed_runtime], accept_eula=True, force=True)
    assert manager.ready(licensed_runtime) and not marker.exists()


def test_noninteractive_local_install_requires_consent(
    licensed_runtime: RuntimeDefinition,
) -> None:
    """The backend API requires consent and retains the licence for later inspection."""
    with pytest.raises(ConfigurationError, match="EULA acceptance required"):
        install_backends(
            [licensed_runtime.name], path=licensed_runtime.path, noninteractive=True
        )
    assert not RuntimeManager().root.exists()
    install_backends(
        [licensed_runtime.name],
        path=licensed_runtime.path,
        noninteractive=True,
        accept_eula=True,
    )
    retained = load_catalog()[licensed_runtime.name]
    assert retained.license_text() == licensed_runtime.license_text()
    assert RuntimeManager().ready(retained)


def test_changed_eula_rejected_before_display(
    licensed_runtime: RuntimeDefinition,
) -> None:
    wheel = licensed_runtime.path.parent / licensed_runtime.data["wheels"][0]["source"]
    wheel.write_bytes(b"Substituted terms")

    def unexpected_prompt(text: str) -> str:
        pytest.fail("Unverified terms must not be presented for consent")

    with pytest.raises(ConfigurationError, match="hash mismatch"):
        provision_runtimes(
            [licensed_runtime], accept_eula=None, prompt=unexpected_prompt
        )
    assert not RuntimeManager().ready(licensed_runtime)
