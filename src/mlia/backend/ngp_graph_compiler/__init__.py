# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""NGP Graph Compiler backend module."""
import logging

from mlia.backend.config import BackendConfiguration
from mlia.backend.config import BackendType
from mlia.backend.config import System
from mlia.backend.ngp_graph_compiler.install import get_ngp_graph_compiler_installation
from mlia.backend.registry import registry
from mlia.core.common import AdviceCategory

logger = logging.getLogger(__name__)

registry.register(
    "ngp-graph-compiler",
    BackendConfiguration(
        supported_advice=[AdviceCategory.PERFORMANCE],
        supported_systems=[System.LINUX_AMD64],
        backend_type=BackendType.CUSTOM,
        installation=get_ngp_graph_compiler_installation(),
    ),
    pretty_name="NGP Graph Compiler",
)
