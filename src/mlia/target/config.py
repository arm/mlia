# SPDX-FileCopyrightText: Copyright 2022-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target configuration module."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import TypeVar

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory
from mlia.core.advisor import InferenceAdvisor
from mlia.utils.filesystem import get_mlia_target_profiles_dir
from mlia.utils.filesystem import get_mlia_target_optimization_dir


def get_builtin_target_profile_path(target_profile: str) -> Path:
    """
    Construct the path to the built-in target profile file.

    No checks are performed.
    """
    return get_mlia_target_profiles_dir() / f"{target_profile}.toml"


def get_builtin_optimization_profile_path(optimization_profile: str) -> Path:
    """
    Construct the path to the built-in target profile file.

    No checks are performed.
    """
    return get_mlia_target_optimization_dir() / f"{optimization_profile}.toml"


@lru_cache
def load_profile(path: str | Path) -> dict[str, Any]:
    """Get settings for the provided target profile."""
    with open(path, "rb") as file:
        profile = tomllib.load(file)

    return cast(dict, profile)


def get_builtin_supported_profile_names() -> list[str]:
    """Return list of default profiles in the target profiles directory."""
    return sorted(
        [
            item.stem
            for item in get_mlia_target_profiles_dir().iterdir()
            if item.is_file() and item.suffix == ".toml"
        ]
    )


BUILTIN_SUPPORTED_PROFILE_NAMES = get_builtin_supported_profile_names()


def is_builtin_target_profile(profile_name: str | Path) -> bool:
    """Check if the given profile name belongs to a built-in profile."""
    return profile_name in BUILTIN_SUPPORTED_PROFILE_NAMES


BUILTIN_SUPPORTED_OPTIMIZATION_NAMES = ["optimization"]


def is_builtin_optimization_profile(optimization_name: str | Path) -> bool:
    """Check if the given optimization name belongs to a built-in optimization."""
    return optimization_name in BUILTIN_SUPPORTED_OPTIMIZATION_NAMES


T = TypeVar("T", bound="TargetProfile")


class TargetProfile(ABC):
    """Base class for target profiles."""

    def __init__(self, target: str, backend_config: dict | None = None) -> None:
        """Init TargetProfile instance with the target name."""
        self.target = target
        # Load backend config(s) to be handled by the backend(s) later.
        self.backend_config = {} if backend_config is None else backend_config

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load and verify a target profile from file and return new instance."""
        profile_data = load_profile(path)

        try:
            new_instance = cls.load_json_data(profile_data)
        except KeyError as ex:
            raise KeyError(f"Missing key in file {path}.") from ex

        return new_instance

    @classmethod
    def load_json_data(cls: type[T], profile_data: dict) -> T:
        """Load a target profile from the JSON data."""
        new_instance = cls(**profile_data)
        new_instance.verify()
        return new_instance

    @classmethod
    def load_profile(cls: type[T], target_profile: str | Path) -> T:
        """Load a target profile from built-in target profile name or file path."""
        if is_builtin_target_profile(target_profile):
            profile_file = get_builtin_target_profile_path(cast(str, target_profile))
        else:
            profile_file = Path(target_profile)
        return cls.load(profile_file)

    def save(self, path: str | Path) -> None:
        """Save this target profile to a file."""
        raise NotImplementedError("Saving target profiles is currently not supported.")

    @abstractmethod
    def verify(self) -> None:
        """
        Check that all attributes contain valid values etc.

        Raises a ValueError, if an issue is detected.
        """
        if not self.target:
            raise ValueError(f"Invalid target name: {self.target}")


@dataclass
class TargetInfo:
    """Collect information about supported targets."""

    supported_backends: list[str]
    default_backends: list[str]
    advisor_factory_func: Callable[..., InferenceAdvisor]
    target_profile_cls: type[TargetProfile]

    def __str__(self) -> str:
        """List supported backends."""
        return ", ".join(sorted(self.supported_backends))

    def is_supported(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> bool:
        """Check if any of the supported backends support this kind of advice."""
        return any(
            name in backend_registry.items
            and backend_registry.items[name].is_supported(advice, check_system)
            for name in self.supported_backends
        )

    def filter_supported_backends(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> list[str]:
        """Get the list of supported backends filtered by the given arguments."""
        return [
            name
            for name in self.supported_backends
            if name in backend_registry.items
            and backend_registry.items[name].is_supported(advice, check_system)
        ]
