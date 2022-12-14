# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""
from mlia.backend.config import BackendConfiguration
from mlia.utils.registry import Registry

# All supported targets are required to be registered here.
registry = Registry[BackendConfiguration]()
