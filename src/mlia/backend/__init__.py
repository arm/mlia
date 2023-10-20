# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Backend module."""
# Make sure all targets are registered with the registry by importing the
# sub-modules
# flake8: noqa
from mlia.backend import argo
from mlia.backend import armnn_tflite_delegate
from mlia.backend import corstone
from mlia.backend import ngp_graph_compiler
from mlia.backend import tosa_checker
from mlia.backend import vela
from mlia.backend import vulkan_model_converter
