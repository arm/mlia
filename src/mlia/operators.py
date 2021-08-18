# Copyright 2021, Arm Ltd.
"""Operators module."""
import logging

from mlia.config import EthosUConfiguration
from mlia.config import IPConfiguration
from mlia.config import ModelConfiguration
from mlia.config import TFLiteModel
from mlia.metadata import Operators
from mlia.tools import vela_wrapper


LOGGER = logging.getLogger("mlia.operators")


def supported_operators(
    model: ModelConfiguration, device: IPConfiguration
) -> Operators:
    """Return list of model's operators."""
    LOGGER.info("Checking operator compatibility ...")

    if not isinstance(model, TFLiteModel):
        raise Exception("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise Exception("Unsupported ip configuration")

    ops = vela_wrapper.supported_operators(model, device)
    LOGGER.info("Done")

    return ops


def generate_supported_operators_report() -> None:
    """Generate supported operators report."""
    vela_wrapper.generate_supported_operators_report()
