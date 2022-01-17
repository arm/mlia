# Copyright 2021, Arm Ltd.
"""Operators module."""
import logging
from pathlib import Path

from mlia.devices.ethosu.config import EthosUConfiguration
from mlia.devices.ethosu.config import IPConfiguration
from mlia.nn.tensorflow.config import ModelConfiguration
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.tools import vela_wrapper

logger = logging.getLogger(__name__)


def supported_operators(
    model: ModelConfiguration, device: IPConfiguration
) -> vela_wrapper.Operators:
    """Return list of model's operators."""
    logger.info("Checking operator compatibility ...")

    if not isinstance(model, TFLiteModel):
        raise Exception("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise Exception("Unsupported ip configuration")

    return vela_wrapper.supported_operators(
        Path(model.model_path), device.compiler_options
    )


def generate_supported_operators_report() -> None:
    """Generate supported operators report."""
    vela_wrapper.generate_supported_operators_report()
