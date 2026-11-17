# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process discovery must not execute irrelevant plugin entry points."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("kind", "args"),
    [
        ("cli", args)
        for args in [
            ["--version"],
            ["--help"],
            [],
            ["backend", "--help"],
            ["target", "--help"],
            ["backend", "install", "--help"],
            ["backend", "uninstall", "--help"],
            ["backend", "list", "--help"],
            ["target", "list", "--help"],
            ["target", "list"],
        ]
    ]
    + [
        ("api", ["list_target_profiles"]),
        ("import", ["mlia.api"]),
        ("import", ["mlia.backend", "mlia.target", "mlia.transformers"]),
    ],
)
def test_metadata_discovery_does_not_load_plugins(kind: str, args: list[str]) -> None:
    """Even installed third-party entry points stay untouched during discovery."""
    script = """
import importlib.metadata as metadata
import importlib.abc
import json, sys
from types import SimpleNamespace
class ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy', 'torch', 'onnx', 'tensorflow'}:
            raise AssertionError('Unnecessary model-library import: ' + fullname)
sys.meta_path.insert(0, ImportGuard())
original = metadata.entry_points
attempts = []
class Sentinel:
    name = "startup-sentinel"
    value = "sentinel:Plugin"
    dist = None
    def __init__(self, group): self.group = group
    def load(self):
        attempts.append(self.group)
        return SimpleNamespace(
            plugin_interface_version="0.0.2", register=lambda registry: None
        )
def entry_points(**kwargs):
    group = kwargs.get("group", "")
    if group.startswith("mlia.plugin."):
        return [Sentinel(group)]
    return original(**kwargs)
metadata.entry_points = entry_points
kind, args = sys.argv[1], json.loads(sys.argv[2])
if kind == 'cli':
    sys.argv = ['mlia', *args]
    from mlia.cli.main import main
    try:
        main()
    except SystemExit as exc:
        assert exc.code == (0 if args else 2), exc.code
elif kind == 'api':
    import mlia.api as api
    getattr(api, args[0])()
else:
    for module in args:
        __import__(module)
assert not attempts, f"Unnecessary plugin loads: {attempts}"
"""
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, kind, json.dumps(args)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": source},
    )
    assert result.returncode == 0, result.stdout + result.stderr
