# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""

import pkgutil

# Allow backend subpackages to be provided by multiple distributions.
__path__ = pkgutil.extend_path(__path__, __name__)

from mlia.backend.registry import registry
from mlia.plugins.plugins import load_backend_plugins

load_backend_plugins(registry)
