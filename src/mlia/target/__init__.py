# SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Target module."""
# Make sure all targets are registered with the registry by importing the
# sub-modules
# flake8: noqa
from mlia.target import cortex_a
from mlia.target import ethos_u
from mlia.target import neural_technology
from mlia.target import tosa
