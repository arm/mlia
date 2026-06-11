# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Init of MLIA."""
# Late imports are intentional in this file because environment and logging
# setup must happen before importing submodules.
# ruff: noqa: E402

# pylint: disable=wrong-import-position
import logging
import os
import pkgutil
from importlib.metadata import version

# redirect warnings to logging
logging.captureWarnings(True)

# Allow mlia subpackages to be provided by multiple distributions.
__path__ = pkgutil.extend_path(__path__, __name__)

# Prevent "No handler" warnings without configuring global logging.
logging.getLogger("mlia").addHandler(logging.NullHandler())


# disable TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = version("mlia")

from mlia import backends, target_profiles, targets
from mlia.api import (
    ValidationMode,
    install_backends,
    list_backend_options,
    list_backends,
    list_target_profiles,
    list_targets,
    run_advisor,
    supported_backends,
    uninstall_backends,
)
from mlia.core.errors import (
    ConfigurationError,
    FunctionalityNotSupportedError,
    InternalError,
    UnsupportedConfigurationError,
)

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
