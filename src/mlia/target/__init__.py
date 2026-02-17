# SPDX-FileCopyrightText: Copyright 2022,2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target module."""

# Make sure all targets are registered with the registry by importing the
# sub-modules
# flake8: noqa
from mlia.plugins.plugins import load_target_plugins
from mlia.target.registry import registry as target_registry

load_target_plugins(target_registry)
