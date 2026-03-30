# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic constants modules."""

from __future__ import annotations

import importlib

import pytest

import mlia
from mlia._constants import exported_mapping, exported_value, normalize_constant_name
from mlia.api import list_backends, list_target_profiles, list_targets


def _values(module: object) -> list[str]:
    return [getattr(module, name) for name in getattr(module, "__all__", [])]


def test_constants_modules_expose_strings() -> None:
    """Ensure constants are non-empty strings."""
    targets = _values(mlia.targets)
    target_profiles = _values(mlia.target_profiles)
    backends = _values(mlia.backends)

    for group in (targets, target_profiles, backends):
        assert all(isinstance(value, str) and value for value in group)


def test_constants_match_current_discovery_outputs() -> None:
    """Installed discovery outputs should match exported constants exactly."""
    assert set(_values(mlia.targets)) == {entry["target"] for entry in list_targets()}
    assert set(_values(mlia.target_profiles)) == {
        entry["name"]
        for entries in list_target_profiles().values()
        for entry in entries
    }
    assert set(_values(mlia.backends)) == {entry["name"] for entry in list_backends()}


def test_constants_modules_support_dir() -> None:
    """dir() should include dynamically exported constants."""
    for module in (mlia.targets, mlia.target_profiles, mlia.backends):
        exported = module.__all__
        assert set(exported).issubset(set(dir(module)))


def test_normalize_constant_name_prefixes_leading_digits() -> None:
    """Constant normalization should keep names valid for Python attributes."""
    assert normalize_constant_name("123-profile") == "_123_PROFILE"


def test_normalize_constant_name_rejects_empty_values() -> None:
    """Empty constant names should fail deterministically."""
    with pytest.raises(ValueError, match="Cannot derive a constant name"):
        normalize_constant_name("!!!")


def test_dynamic_constants_support_attribute_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic facade modules should resolve exported attributes on demand."""
    monkeypatch.setattr(mlia.backends, "__all__", ["VELA"])
    monkeypatch.setattr(mlia.targets, "__all__", ["ETHOS_U55"])
    monkeypatch.setattr(mlia.target_profiles, "__all__", ["ETHOS_U55_256"])

    monkeypatch.setattr("mlia.backends.exported_value", lambda _kind, _name: "vela")
    monkeypatch.setattr("mlia.targets.exported_value", lambda _kind, _name: "ethos-u55")
    monkeypatch.setattr(
        "mlia.target_profiles.exported_value",
        lambda _kind, _name: "ethos-u55-256",
    )

    assert mlia.backends.VELA == "vela"
    assert mlia.targets.ETHOS_U55 == "ethos-u55"
    assert mlia.target_profiles.ETHOS_U55_256 == "ethos-u55-256"


def test_exported_value_rejects_missing_symbol() -> None:
    """Unknown exported symbols should raise AttributeError."""
    with pytest.raises(AttributeError, match="module has no attribute"):
        exported_value("targets", "MISSING")


def test_exported_value_returns_known_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known exported symbols should resolve to their canonical value."""
    constants = importlib.import_module("mlia._constants")
    constants.exported_value.cache_clear()
    monkeypatch.setattr(
        constants,
        "_entries_for_kind",
        lambda _kind: (constants.ConstantEntry(symbol="ETHOS_U55", value="ethos-u55"),),
    )

    assert exported_value("targets", "ETHOS_U55") == "ethos-u55"
    constants.exported_value.cache_clear()


def test_exported_mapping_rejects_unknown_kind() -> None:
    """Unknown constant groups should fail clearly."""
    with pytest.raises(ValueError, match="Unknown constants kind"):
        exported_mapping("unknown-kind")


def test_dynamic_constants_reject_symbol_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conflicting normalized names should fail deterministically."""
    constants = importlib.import_module("mlia._constants")
    constants.exported_symbols.cache_clear()
    monkeypatch.setattr(
        constants,
        "_entries_for_kind",
        lambda kind: constants._build_entries(["foo-bar", "foo_bar"]),
    )
    with pytest.raises(ValueError, match="Constant symbol collision"):
        constants.exported_symbols("targets")
    constants.exported_symbols.cache_clear()
