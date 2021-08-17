# Copyright 2021, Arm Ltd.
"""Model and IP configuration."""
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf
from typing_extensions import Literal


class ModelConfiguration:
    """Base class for model configuration."""

    model_path: str


class KerasModel(ModelConfiguration):
    """Keras model congiguration."""

    def __init__(self, model_path: str):
        """Init Keras model configuration."""
        self.model_path = model_path

    def get_keras_model(self) -> tf.keras.Model:
        """Return associated keras model."""
        return tf.keras.models.load_model(self.model_path)


class TFLiteModel(ModelConfiguration):
    """TFLite model configuration."""

    def __init__(self, model_path: Union[Path, str]):
        """Init TFLite model configuration."""
        self.model_path: str = str(model_path)

    def input_details(self) -> List[Dict]:
        """Get model's input details."""
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        return cast(List[Dict], interpreter.get_input_details())

    def get_tflite_model(self) -> tf.lite.Interpreter:
        """Return associated tflite model."""
        tf.lite.Interpreter(model_path=self.model_path)


class CompilerOptions:
    """Compiler options."""

    def __init__(
        self,
        config_files: Optional[Union[str, List[str]]] = None,
        system_config: str = "internal-default",
        memory_mode: str = "internal-default",
        accelerator_config: Literal[
            "ethos-u55-32",
            "ethos-u55-64",
            "ethos-u55-128",
            "ethos-u55-256",
            "ethos-u65-256",
            "ethos-u65-512",
        ] = "ethos-u55-256",
        max_block_dependency: int = 3,
        arena_cache_size: Optional[int] = None,
        tensor_allocator: Literal["LinearAlloc", "Greedy", "HillClimb"] = "HillClimb",
        cpu_tensor_alignment: int = 16,
        optimization_strategy: Literal["Performance", "Size"] = "Performance",
        output_dir: Optional[str] = None,
    ):
        """Init compiler options."""
        self.config_files = config_files
        self.system_config = system_config
        self.memory_mode = memory_mode
        self.accelerator_config = accelerator_config
        self.max_block_dependency = max_block_dependency
        self.arena_cache_size = arena_cache_size
        self.tensor_allocator = tensor_allocator
        self.cpu_tensor_alignment = cpu_tensor_alignment
        self.optimization_strategy = optimization_strategy
        self.output_dir = output_dir

    def __str__(self) -> str:
        """Return string representation."""
        params = ", ".join(
            f"{param}: {param_value}" for param, param_value in self.__dict__.items()
        )
        return f"Compiler options {params}"


class IPConfiguration:
    """Base class for IP configuration."""


class EthosUConfiguration(IPConfiguration):
    """EthosU configuration."""

    def __init__(
        self,
        ip_class: Literal["ethos-u55", "ethos-u65"],
        mac: int,
        compiler_options: CompilerOptions,
    ):
        """Init EthosU configuration."""
        self.ip_class = ip_class
        self.mac = mac
        self.compiler_options = compiler_options

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"EthosU ip_class={self.ip_class} "
            f"mac={self.mac} "
            f"compiler_options= {self.compiler_options}"
        )


class EthosU55(EthosUConfiguration):
    """EthosU55 configuration."""

    def __init__(self, mac: Literal[32, 64, 128, 256] = 256, **kwargs: Any) -> None:
        """Init EthosU55 configuration."""
        if not mac or mac not in (32, 64, 128, 256):
            raise Exception("Wrong or empty MAC value")

        super().__init__(
            ip_class="ethos-u55",
            mac=mac,
            compiler_options=CompilerOptions(
                accelerator_config=f"ethos-u55-{mac}", **kwargs  # type: ignore
            ),
        )


class EthosU65(EthosUConfiguration):
    """EthosU65 configuration."""

    def __init__(self, mac: Literal[256, 512] = 256, **kwargs: Any) -> None:
        """Init EthosU65 configuration."""
        if not mac or mac not in (256, 512):
            raise Exception("Wrong or empty MAC value")

        super().__init__(
            ip_class="ethos-u65",
            mac=mac,
            compiler_options=CompilerOptions(
                accelerator_config=f"ethos-u65-{mac}", **kwargs  # type: ignore
            ),
        )
