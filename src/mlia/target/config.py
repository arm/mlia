# SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target configuration module."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast
from typing import TypeVar

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from mlia.backend.registry import registry as backend_registry
from mlia.core.common import AdviceCategory
from mlia.utils.filesystem import get_mlia_target_profiles_dir


def get_profile_file(target_profile: str | Path) -> Path:
    """Get the target profile toml file."""
    if not target_profile:
        raise Exception("Target profile is not provided.")

    profile_file = Path(get_mlia_target_profiles_dir() / f"{target_profile}.toml")
    if not profile_file.is_file():
        profile_file = Path(target_profile)

    if not profile_file.exists():
        raise Exception(f"File not found: {profile_file}.")
    return profile_file


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


def get_target(target_profile: str | Path) -> str:
    """Return target for the provided target_profile."""
    profile_file = get_profile_file(target_profile)
    profile = load_profile(profile_file)
    return cast(str, profile["target"])


T = TypeVar("T", bound="TargetProfile")


class TargetProfile(ABC):
    """Base class for target profiles."""

    def __init__(self, target: str) -> None:
        """Init TargetProfile instance with the target name."""
        self.target = target

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load and verify a target profile from file and return new instance."""
        profile = load_profile(path)

        try:
            new_instance = cls(**profile)
        except KeyError as ex:
            raise KeyError(f"Missing key in file {path}.") from ex

        new_instance.verify()

        return new_instance

    @classmethod
    def load_profile(cls: type[T], target_profile: str) -> T:
        """Load a target profile by name."""
        profile_file = get_profile_file(target_profile)
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

    def __str__(self) -> str:
        """List supported backends."""
        return ", ".join(sorted(self.supported_backends))

    def is_supported(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> bool:
        """Check if any of the supported backends support this kind of advice."""
        return any(
            backend_registry.items[name].is_supported(advice, check_system)
            for name in self.supported_backends
        )

    def filter_supported_backends(
        self, advice: AdviceCategory | None = None, check_system: bool = False
    ) -> list[str]:
        """Get the list of supported backends filtered by the given arguments."""
        return [
            name
            for name in self.supported_backends
            if backend_registry.items[name].is_supported(advice, check_system)
        ]
