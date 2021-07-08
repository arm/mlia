# Copyright 2021, Arm Ltd.
"""Operators module."""
from typing import List

from mlia.config import EthosUConfiguration
from mlia.config import IPConfiguration
from mlia.config import ModelConfiguration
from mlia.config import TFLiteModel
from mlia.metadata import Operation
from mlia.tools import vela_wrapper


def supported_operators(
    model: ModelConfiguration, device: IPConfiguration
) -> List[Operation]:
    """Return list of model's operations."""
    if not isinstance(model, TFLiteModel):
        raise Exception("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise Exception("Unsupported ip configuration")

    return vela_wrapper.supported_operators(model, device)
