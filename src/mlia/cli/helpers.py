# Copyright 2022, Arm Ltd.
"""Module for various helper classes."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from mlia.cli.options import get_target_opts
from mlia.core.helpers import ActionResolver
from mlia.nn.tensorflow.optimizations.select import OptimizationSettings
from mlia.nn.tensorflow.utils import is_keras_model
from mlia.utils.types import is_list_of


class CLIActionResolver(ActionResolver):
    """Helper class for generating cli commands."""

    def __init__(self, args: Dict[str, Any]) -> None:
        """Init action resolver."""
        self.args = args

    @staticmethod
    def _general_optimization_command(model_path: Optional[str]) -> List[str]:
        """Return general optimization command description."""
        keras_note = []
        if model_path is None or not is_keras_model(model_path):
            model_path = "/path/to/keras_model"
            keras_note = ["Note: you will need a Keras/TF.saved_model input for that."]

        return [
            *keras_note,
            "For example: mlia optimization --optimization-type "
            f"pruning,clustering --optimization-target 0.5,32 {model_path}",
            "For more info: mlia optimization --help",
        ]

    @staticmethod
    def _specific_optimization_command(
        model_path: str,
        device_opts: str,
        opt_settings: List[OptimizationSettings],
    ) -> List[str]:
        """Return specific optimization command description."""
        opt_types = ",".join(opt.optimization_type for opt in opt_settings)
        opt_targs = ",".join(str(opt.optimization_target) for opt in opt_settings)

        return [
            "For more info: mlia optimization --help",
            "Optimization command: "
            f"mlia optimization --optimization-type {opt_types} "
            f"--optimization-target {opt_targs}{device_opts} {model_path}",
        ]

    def apply_optimizations(self, **kwargs: Any) -> List[str]:
        """Return command details for applying optimizations."""
        model_path, device_opts = self._get_model_and_device_opts()

        if (opt_settings := kwargs.pop("opt_settings", None)) is None:
            return self._general_optimization_command(model_path)

        if is_list_of(opt_settings, OptimizationSettings) and model_path:
            return self._specific_optimization_command(
                model_path, device_opts, opt_settings
            )

        return []

    def supported_operators_info(self) -> List[str]:
        """Return command details for generating supported ops report."""
        return [
            "For guidance on supported operators, run: mlia operators "
            "--supported-ops-report",
        ]

    def check_performance(self) -> List[str]:
        """Return command details for checking performance."""
        model_path, device_opts = self._get_model_and_device_opts()
        if not model_path:
            return []

        return [
            "Check the estimated performance by running the following command: ",
            f"mlia performance{device_opts} {model_path}",
        ]

    def check_operator_compatibility(self) -> List[str]:
        """Return command details for op compatibility."""
        model_path, device_opts = self._get_model_and_device_opts()
        if not model_path:
            return []

        return [
            "Try running the following command to verify that:",
            f"mlia operators{device_opts} {model_path}",
        ]

    def operator_compatibility_details(self) -> List[str]:
        """Return command details for op compatibility."""
        return ["For more details, run: mlia operators --help"]

    def optimization_details(self) -> List[str]:
        """Return command details for optimization."""
        return ["For more info, see: mlia optimization --help"]

    def _get_model_and_device_opts(
        self, separate_device_opts: bool = True
    ) -> Tuple[Optional[str], str]:
        """Get model and device options."""
        device_opts = " ".join(get_target_opts(self.args))
        if separate_device_opts and device_opts:
            device_opts = f" {device_opts}"

        model_path = self.args.get("model")
        return model_path, device_opts
