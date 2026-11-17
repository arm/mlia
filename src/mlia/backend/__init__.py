# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""

import pkgutil

# Allow backend subpackages to be provided by multiple distributions.
__path__ = pkgutil.extend_path(__path__, __name__)
