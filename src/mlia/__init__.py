# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Init of MLIA."""
# Package setup must happen before importing even the lightweight public errors.
# ruff: noqa: E402

import logging
import os
import pkgutil
from importlib import import_module
from importlib.metadata import version
from typing import Any

# redirect warnings to logging
logging.captureWarnings(True)

# Allow mlia subpackages to be provided by multiple distributions.
__path__ = pkgutil.extend_path(__path__, __name__)

# Prevent "No handler" warnings without configuring global logging.
logging.getLogger("mlia").addHandler(logging.NullHandler())


# disable TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = version("mlia")

from mlia.core.errors import (
    ConfigurationError,
    FunctionalityNotSupportedError,
    InternalError,
    UnsupportedConfigurationError,
)

_LAZY_EXPORTS = {
    "backends": ("mlia.backends", None),
    "install_backends": ("mlia.api", "install_backends"),
    "list_backend_options": ("mlia.api", "list_backend_options"),
    "list_backends": ("mlia.api", "list_backends"),
    "list_target_profiles": ("mlia.api", "list_target_profiles"),
    "list_targets": ("mlia.api", "list_targets"),
    "run_advisor": ("mlia.api", "run_advisor"),
    "supported_backends": ("mlia.api", "supported_backends"),
    "target_profiles": ("mlia.target_profiles", None),
    "targets": ("mlia.targets", None),
    "uninstall_backends": ("mlia.api", "uninstall_backends"),
    "ValidationMode": ("mlia.api", "ValidationMode"),
}

__all__ = [
    "backends",
    "install_backends",
    "uninstall_backends",
    "list_backend_options",
    "list_backends",
    "list_target_profiles",
    "list_targets",
    "ConfigurationError",
    "FunctionalityNotSupportedError",
    "InternalError",
    "UnsupportedConfigurationError",
    "ValidationMode",
    "run_advisor",
    "supported_backends",
    "target_profiles",
    "targets",
]


def __getattr__(name: str) -> Any:
    """Load runtime-heavy public exports on first access."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as err:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from err

    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public exports through standard introspection."""
    return sorted(set(globals()) | set(__all__))
