# Copyright 2021, Arm Ltd.
"""Performance estimation."""
from mlia.config import EthosUConfiguration
from mlia.config import IPConfiguration
from mlia.config import ModelConfiguration
from mlia.config import TFLiteModel
from mlia.exceptions import ConfigurationError
from mlia.metrics import PerformanceMetrics
from mlia.tools.vela_wrapper import estimate_performance


def collect_performance_metrics(
    model: ModelConfiguration, device: IPConfiguration
) -> PerformanceMetrics:
    """Collect performance metrics."""
    if not isinstance(model, TFLiteModel):
        raise ConfigurationError("Unsupported model configuration")

    if not isinstance(device, EthosUConfiguration):
        raise ConfigurationError("Unsupported device configuration")

    return estimate_performance(model, device)
