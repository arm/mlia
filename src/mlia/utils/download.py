# SPDX-FileCopyrightText: Copyright 2023, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for classified download gateways."""

from mlia.utils.boundaries.network import DownloadConfig, download_progress
from mlia.utils.boundaries.network import download_with_notice as download

__all__ = ["DownloadConfig", "download", "download_progress"]
