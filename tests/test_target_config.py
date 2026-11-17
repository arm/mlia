# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for target profile loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from mlia.target.config import TargetInfo, TargetProfile
from mlia.target.registry import create_target_profile, registry


class _DummyTargetProfile(TargetProfile):
    """Minimal concrete TargetProfile for load-path testing."""

    def __init__(
        self, target: str, backend_config: dict | None = None, **_kwargs: Any
    ) -> None:
        super().__init__(target, backend_config)

    def verify(self) -> None:
        super().verify()


def test_target_profile_load_forwards_override_backend_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File-based load should forward override options into load_data()."""
    monkeypatch.setattr(
        "mlia.target.config.load_profile",
        lambda _path: {"target": "dummy", "backend": {"vela": {"existing": True}}},
    )
    load_data = MagicMock(return_value=_DummyTargetProfile("dummy"))
    monkeypatch.setattr(_DummyTargetProfile, "load_data", load_data)

    profile = _DummyTargetProfile.load(
        "dummy.toml", {"vela": {"override": False}, "corstone": {"x": 1}}
    )

    assert isinstance(profile, _DummyTargetProfile)
    load_data.assert_called_once_with(
        {"target": "dummy", "backend": {"vela": {"existing": True}}},
        {"vela": {"override": False}, "corstone": {"x": 1}},
    )


def test_target_profile_load_data_merges_config_and_overrides() -> None:
    """Dictionary-based load should normalize config and merge overrides."""
    profile = _DummyTargetProfile.load_data(
        {
            "config": {"target": "dummy"},
            "backend": {"vela": {"existing": True}},
        },
        {"vela": {"override": False}, "corstone": {"x": 1}},
    )

    assert profile.target == "dummy"
    assert profile.backend_config == {
        "vela": {"existing": True, "override": False},
        "corstone": {"x": 1},
    }


def test_repeated_profile_load_preserves_cached_config(tmp_path: Path) -> None:
    """Loading a profile with overrides must not change subsequent loads."""
    path = tmp_path / "target.toml"
    path.write_text('[config]\ntarget = "dummy"\n[backend.vela]\ncount = 1\n')
    first = _DummyTargetProfile.load(path, {"vela": {"count": 2}})
    second = _DummyTargetProfile.load(path)
    assert first.target == second.target == "dummy"
    assert first.backend_config == {"vela": {"count": 2}}
    assert second.backend_config == {"vela": {"count": 1}}


def test_create_target_profile_uses_load_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registry profile creation should use the canonical load_data path."""
    create_target_profile.cache_clear()
    monkeypatch.setattr(
        "mlia.target.registry.load_profile",
        lambda _path: {"target": "dummy"},
    )
    load_data = MagicMock(return_value=_DummyTargetProfile("dummy"))
    monkeypatch.setitem(
        registry.items,
        "dummy",
        TargetInfo(
            supported_backends=[],
            default_backends=[],
            advisor_factory_func=cast(Any, MagicMock()),
            target_profile_cls=cast(Any, MagicMock(load_data=load_data)),
        ),
    )

    profile = create_target_profile(Path("dummy.toml"))

    assert isinstance(profile, _DummyTargetProfile)
    load_data.assert_called_once_with({"target": "dummy"})
