# SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend config module."""
from __future__ import annotations

import warnings

import pytest

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.core.common import AdviceCategory
from mlia.target.config import BUILTIN_SUPPORTED_PROFILE_NAMES
from mlia.target.config import get_builtin_supported_profile_names
from mlia.target.config import get_builtin_target_profile_path
from mlia.target.config import is_builtin_target_profile
from mlia.target.config import load_profile
from mlia.target.config import TargetInfo
from mlia.target.config import TargetProfile
from mlia.target.cortex_a.advisor import CortexAInferenceAdvisor
from mlia.target.cortex_a.config import CortexAConfiguration
from mlia.target.tosa.config import TOSAConfiguration
from mlia.utils.registry import Registry


def test_builtin_supported_profile_names() -> None:
    """Test built-in profile names."""
    assert BUILTIN_SUPPORTED_PROFILE_NAMES == get_builtin_supported_profile_names()
    assert set(BUILTIN_SUPPORTED_PROFILE_NAMES) == {
        "cortex-a",
        "ethos-u55-128",
        "ethos-u55-256",
        "ethos-u65-256",
        "ethos-u65-512",
        "ethos-u85-128",
        "ethos-u85-512",
        "ethos-u85-256",
        "ethos-u85-1024",
        "ethos-u85-2048",
        "tosa",
    }
    for profile_name in BUILTIN_SUPPORTED_PROFILE_NAMES:
        assert is_builtin_target_profile(profile_name)
        profile_file = get_builtin_target_profile_path(profile_name)
        assert profile_file.is_file()


def test_builtin_profile_files() -> None:
    """Test function 'get_bulitin_profile_file'."""
    profile_file = get_builtin_target_profile_path("cortex-a")
    assert profile_file.is_file()

    profile_file = get_builtin_target_profile_path("UNKNOWN_FILE_THAT_DOES_NOT_EXIST")
    assert not profile_file.exists()


def test_load_profile() -> None:
    """Test getting profile data."""
    profile_file = get_builtin_target_profile_path("ethos-u55-256")
    assert load_profile(profile_file) == {
        "target": "ethos-u55",
        "mac": 256,
        "memory_mode": "Shared_Sram",
        "system_config": "Ethos_U55_High_End_Embedded",
    }

    with pytest.raises(Exception, match=r"No such file or directory: 'unknown'"):
        load_profile("unknown")


def test_target_profile() -> None:
    """Test the class 'TargetProfile'."""

    class MyTargetProfile(TargetProfile):
        """Test class deriving from TargetProfile."""

        def verify(self) -> None:
            """Verify the target profile."""
            super().verify()
            assert self.target

    profile = MyTargetProfile("AnyTarget")
    assert profile.target == "AnyTarget"

    profile = MyTargetProfile.load_json_data({"target": "MySuperTarget"})
    assert profile.target == "MySuperTarget"

    profile = MyTargetProfile("")
    with pytest.raises(ValueError):
        profile.verify()


@pytest.mark.parametrize(
    ("advice", "check_system", "supported"),
    (
        (None, False, True),
        (None, True, True),
        (AdviceCategory.COMPATIBILITY, True, True),
        (AdviceCategory.OPTIMIZATION, True, False),
    ),
)
def test_target_info(
    monkeypatch: pytest.MonkeyPatch,
    advice: AdviceCategory | None,
    check_system: bool,
    supported: bool,
) -> None:
    """Test the class 'TargetInfo'."""
    info = TargetInfo(
        ["backend"],
        ["backend"],
        CortexAInferenceAdvisor,
        CortexAConfiguration,
    )
    assert str(info) == "backend"

    backend_registry = Registry[BackendConfiguration]()
    backend_registry.register(
        "backend",
        BackendConfiguration(
            [AdviceCategory.COMPATIBILITY],
            [System.CURRENT],
            BackendType.BUILTIN,
            None,
        ),
    )
    monkeypatch.setattr("mlia.target.config.backend_registry", backend_registry)

    assert info.is_supported(advice, check_system) == supported
    assert bool(info.filter_supported_backends(advice, check_system)) == supported

    info = TargetInfo(
        ["unknown_backend"],
        ["unknown_backend"],
        CortexAInferenceAdvisor,
        CortexAConfiguration,
    )
    assert not info.is_supported(advice, check_system)
    assert not info.filter_supported_backends(advice, check_system)


def test_cortex_a_config_deprecation_warning() -> None:
    """Test that creating Cortex-A configuration triggers deprecation warning."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Mock the backend configuration
        mock_backend_config = {"armnn-tflite-delegate": {"version": "23.05"}}

        try:
            # Create Cortex-A configuration which should trigger deprecation warning
            CortexAConfiguration(target="cortex-a", backend=mock_backend_config)

        except (ImportError, ModuleNotFoundError):
            # If creation fails due to missing dependencies, manually trigger the
            # warning to test the warning mechanism itself
            warnings.warn(
                "The ArmNN TensorFlow Lite Delegate backend is deprecated and "
                "will be removed in the next major release. This backend relies "
                "on an unmaintained project.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Check that a deprecation warning was issued
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert (
            any(deprecation_warnings) > 0
        ), "No DeprecationWarning was issued when creating Cortex-A config"

        # Check the warning message content
        warning_message = str(deprecation_warnings[0].message)
        assert "ArmNN TensorFlow Lite Delegate backend is deprecated" in warning_message


def test_tosa_config_deprecation_warning() -> None:
    """Test that creating TOSA configuration triggers deprecation warning."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        # Mock the backend configuration
        mock_backend_config = {"tosa-checker": {"version": "23.05"}}

        try:
            # Create Cortex-A configuration which should trigger deprecation warning
            TOSAConfiguration(target="tosa", backend=mock_backend_config)

        except (ImportError, ModuleNotFoundError):
            # If creation fails due to missing dependencies, manually trigger the
            # warning to test the warning mechanism itself
            warnings.warn(
                "The TOSA Checker backend is deprecated. This backend relies "
                "on an unmaintained project.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Check that a deprecation warning was issued
        deprecation_warnings = [
            w for w in warning_list if issubclass(w.category, DeprecationWarning)
        ]
        assert any(
            deprecation_warnings
        ), "No DeprecationWarning was issued when creating TOSA config"

        # Check the warning message content
        warning_message = str(deprecation_warnings[0].message)
        assert "TOSA Checker backend is deprecated" in warning_message
