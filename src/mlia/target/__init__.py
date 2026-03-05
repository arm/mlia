# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target module."""

import pkgutil

# Allow target subpackages to be provided by multiple distributions.
__path__ = pkgutil.extend_path(__path__, __name__)

# Make sure all targets are registered with the registry by importing the
# sub-modules
# flake8: noqa
from mlia.backend.registry import registry as backend_registry
from mlia.plugins.plugins import load_backend_plugins, load_target_plugins
from mlia.target.registry import registry as target_registry

load_backend_plugins(backend_registry)
load_target_plugins(target_registry)
