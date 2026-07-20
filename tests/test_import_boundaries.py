# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process tests for completion import boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETION_ENVIRONMENT_VARIABLES = (
    "_MLIA_COMPLETE",
    "COMP_WORDS",
    "COMP_CWORD",
    "_TYPER_COMPLETE_ARGS",
)


def _run_python(code: str) -> dict[str, object]:
    """Run code in a fresh Python process and return its JSON result."""
    environment = os.environ.copy()
    for variable in COMPLETION_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    source_dir = os.fspath(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_dir
        if not environment.get("PYTHONPATH")
        else os.pathsep.join((source_dir, environment["PYTHONPATH"]))
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(json.loads(result.stdout))


def test_package_defers_runtime_heavy_public_exports() -> None:
    """Importing MLIA should keep runtime-heavy public exports lazy."""
    result = _run_python(
        """
        import json
        import sys
        import mlia

        print(json.dumps({
            "api_loaded": "mlia.api" in sys.modules,
            "backends_loaded": "mlia.backends" in sys.modules,
            "targets_loaded": "mlia.targets" in sys.modules,
            "target_profiles_loaded": "mlia.target_profiles" in sys.modules,
            "error_available": issubclass(mlia.ConfigurationError, Exception),
            "public_exports_visible": set(mlia.__all__).issubset(dir(mlia)),
        }))
        """
    )

    assert result == {
        "api_loaded": False,
        "backends_loaded": False,
        "targets_loaded": False,
        "target_profiles_loaded": False,
        "error_available": True,
        "public_exports_visible": True,
    }


def test_lazy_public_api_exports_remain_compatible() -> None:
    """Accessing lazy exports should preserve the public package API."""
    result = _run_python(
        """
        import json
        import sys
        import mlia

        api_loaded_before = "mlia.api" in sys.modules
        from mlia import ValidationMode, list_backends, run_advisor
        print(json.dumps({
            "api_loaded_before": api_loaded_before,
            "api_loaded_after": "mlia.api" in sys.modules,
            "validation_mode": ValidationMode.__name__,
            "list_backends_callable": callable(list_backends),
            "run_advisor_callable": callable(run_advisor),
        }))
        """
    )

    assert result == {
        "api_loaded_before": False,
        "api_loaded_after": True,
        "validation_mode": "ValidationMode",
        "list_backends_callable": True,
        "run_advisor_callable": True,
    }


@pytest.mark.parametrize(
    ("comp_words", "comp_cword", "expected_candidate"),
    [
        ("mlia ", "1", "check"),
        ("mlia check --backend-a", "2", "--backend-a.optimization-level"),
    ],
)
def test_generic_completion_avoids_runtime_state_modules(
    comp_words: str, comp_cword: str, expected_candidate: str
) -> None:
    """Controlled backend registration should not initialize other runtime state."""
    result = _run_python(
        f"""
        import json
        import os
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import click
        from typer.testing import CliRunner

        os.environ["_MLIA_COMPLETE"] = "complete_bash"
        import mlia.plugins.plugins

        def register_backends(registry):
            registry.register(
                "backend-a",
                SimpleNamespace(
                    selectable=True,
                    cli_options={{
                        "optimization_level": click.Option(
                            ["--optimization-level"],
                            type=click.Choice(["0", "1", "2"]),
                        )
                    }},
                ),
            )
            registry.plugin_interface_versions["backend-a"] = "0.0.2"

        with patch.object(
            mlia.plugins.plugins,
            "load_backend_plugins",
            side_effect=register_backends,
        ) as load_backend_plugins:
            import mlia.cli.main

            completion = CliRunner().invoke(
                mlia.cli.main.mlia_app,
                [],
                prog_name="mlia",
                env={{
                    "_MLIA_COMPLETE": "complete_bash",
                    "COMP_WORDS": {comp_words!r},
                    "COMP_CWORD": {comp_cword!r},
                }},
            )
        modules = (
            "mlia.api",
            "mlia.backend.install",
            "mlia.cli.command_validators",
            "mlia.core.advisor",
            "mlia.core.output_validation",
            "mlia.target",
            "mlia.target.config",
            "mlia.target.registry",
        )
        print(json.dumps({{
            "completion_succeeded": completion.exit_code == 0,
            "expected_candidate_present": (
                {expected_candidate!r} in completion.stdout.splitlines()
            ),
            "backend_loaded": "mlia.backend" in sys.modules,
            "plugin_load_count": load_backend_plugins.call_count,
            **{{module: module in sys.modules for module in modules}},
        }}))
        """
    )

    assert result == {
        "completion_succeeded": True,
        "expected_candidate_present": True,
        "backend_loaded": True,
        "plugin_load_count": 1,
        "mlia.api": False,
        "mlia.backend.install": False,
        "mlia.cli.command_validators": False,
        "mlia.core.advisor": False,
        "mlia.core.output_validation": False,
        "mlia.target": False,
        "mlia.target.config": False,
        "mlia.target.registry": False,
    }


@pytest.mark.parametrize(
    ("comp_words", "comp_cword"),
    [
        ("mlia check model.tflite --target-profile profile --backend backend-", "6"),
        ("mlia backend install backend-", "3"),
        ("mlia backend uninstall backend-", "3"),
    ],
)
def test_backend_completion_uses_package_plugin_boundary(
    comp_words: str, comp_cword: str
) -> None:
    """Core backend completion should use registered names without state checks."""
    result = _run_python(
        f"""
        import json
        import os
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from typer.testing import CliRunner

        os.environ["_MLIA_COMPLETE"] = "complete_bash"
        import mlia.plugins.plugins

        def register_backends(registry):
            registry.register(
                "backend-b", SimpleNamespace(selectable=True, cli_options={{}})
            )
            registry.register(
                "hidden", SimpleNamespace(selectable=False, cli_options={{}})
            )
            registry.register(
                "backend-a", SimpleNamespace(selectable=True, cli_options={{}})
            )

        with patch.object(
            mlia.plugins.plugins,
            "load_backend_plugins",
            side_effect=register_backends,
        ) as load_backend_plugins:
            import mlia.cli.main

            completion = CliRunner().invoke(
                mlia.cli.main.mlia_app,
                [],
                prog_name="mlia",
                env={{
                    "_MLIA_COMPLETE": "complete_bash",
                    "COMP_WORDS": {comp_words!r},
                    "COMP_CWORD": {comp_cword!r},
                }},
            )

        modules = (
            "mlia.api",
            "mlia.backend.install",
            "mlia.backend.manager",
        )
        print(json.dumps({{
            "completion_succeeded": completion.exit_code == 0,
            "candidates": completion.stdout.splitlines(),
            "plugin_load_count": load_backend_plugins.call_count,
            **{{module: module in sys.modules for module in modules}},
        }}))
        """
    )

    assert result == {
        "completion_succeeded": True,
        "candidates": ["backend-a", "backend-b"],
        "plugin_load_count": 1,
        "mlia.api": False,
        "mlia.backend.install": False,
        "mlia.backend.manager": False,
    }


def test_target_profile_completion_avoids_target_runtime_modules() -> None:
    """Packaged profile completion should not initialize the target runtime."""
    result = _run_python(
        """
        import json
        import os
        import sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from typer.testing import CliRunner

        os.environ["_MLIA_COMPLETE"] = "complete_bash"
        import mlia
        import mlia.plugins.plugins

        with patch.object(mlia.plugins.plugins, "load_backend_plugins"):
            import mlia.cli.main

            with TemporaryDirectory() as temporary_dir:
                namespace_dir = Path(temporary_dir) / "mlia"
                profiles_dir = namespace_dir / "resources" / "target_profiles"
                profiles_dir.mkdir(parents=True)
                (profiles_dir / "target-a.toml").write_text(
                    'profile_name = "target-a"\\n',
                    encoding="utf-8",
                )
                mlia.__path__ = [str(namespace_dir)]
                completion = CliRunner().invoke(
                    mlia.cli.main.mlia_app,
                    [],
                    prog_name="mlia",
                    env={
                        "_MLIA_COMPLETE": "complete_bash",
                        "COMP_WORDS": "mlia check model.tflite --target-profile target",
                        "COMP_CWORD": "4",
                    },
                )

        modules = (
            "mlia.api",
            "mlia.backend.install",
            "mlia.target",
            "mlia.target.config",
            "mlia.target.registry",
        )
        print(json.dumps({
            "completion_succeeded": completion.exit_code == 0,
            "candidates": completion.stdout.splitlines(),
            **{module: module in sys.modules for module in modules},
        }))
        """
    )

    assert result == {
        "completion_succeeded": True,
        "candidates": ["target-a"],
        "mlia.api": False,
        "mlia.backend.install": False,
        "mlia.target": False,
        "mlia.target.config": False,
        "mlia.target.registry": False,
    }
