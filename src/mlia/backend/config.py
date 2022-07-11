# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Contain definition of backend configuration."""
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict
from typing import Union


class UserParamConfig(TypedDict, total=False):
    """User parameter configuration."""

    name: Optional[str]
    default_value: str
    values: List[str]
    description: str
    alias: str


UserParamsConfig = Dict[str, List[UserParamConfig]]


class ExecutionConfig(TypedDict, total=False):
    """Execution configuration."""

    commands: Dict[str, List[str]]
    user_params: UserParamsConfig
    variables: Dict[str, str]


class NamedExecutionConfig(ExecutionConfig):
    """Execution configuration with name."""

    name: str


class BaseBackendConfig(ExecutionConfig, total=False):
    """Base backend configuration."""

    name: str
    description: str
    config_location: Path
    annotations: Dict[str, Union[str, List[str]]]


class ApplicationConfig(BaseBackendConfig, total=False):
    """Application configuration."""

    supported_systems: List[str]


class ExtendedApplicationConfig(BaseBackendConfig, total=False):
    """Extended application configuration."""

    supported_systems: List[NamedExecutionConfig]


class SystemConfig(BaseBackendConfig, total=False):
    """System configuration."""

    reporting: Dict[str, Dict]


BackendItemConfig = Union[ApplicationConfig, SystemConfig]
BackendConfig = Union[List[ExtendedApplicationConfig], List[SystemConfig]]
