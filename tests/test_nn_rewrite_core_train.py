# SPDX-FileCopyrightText: Copyright 2023-2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.core.train."""
# pylint: disable=too-many-arguments
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
from keras.api._v2 import keras  # Temporary workaround for now: MLIA-1107

from mlia.nn.rewrite.core.rewrite import FullyConnectedRewrite
from mlia.nn.rewrite.core.rewrite import QATRewrite
from mlia.nn.rewrite.core.train import augment_fn_twins
from mlia.nn.rewrite.core.train import AUGMENTATION_PRESETS
from mlia.nn.rewrite.core.train import LearningRateSchedule
from mlia.nn.rewrite.core.train import mixup
from mlia.nn.rewrite.core.train import train
from mlia.nn.rewrite.core.train import TrainingParameters
from mlia.nn.rewrite.library.fc_layer import get_keras_model as fc_rewrite
from tests.utils.rewrite import MockTrainingParameters


def replace_fully_connected_with_conv(
    input_shape: Any, output_shape: Any
) -> keras.Model:
    """Get a replacement model for the fully connected layer."""
    for name, shape in {
        "Input": input_shape,
        "Output": output_shape,
    }.items():
        if len(shape) != 1:
            raise RuntimeError(f"{name}: shape (N,) expected, but it is {input_shape}.")

    model = keras.Sequential(name="RewriteModel")
    model.add(keras.Input(input_shape))
    model.add(keras.layers.Reshape((1, 1, input_shape[0])))
    model.add(keras.layers.Conv2D(filters=output_shape[0], kernel_size=(1, 1)))
    model.add(keras.layers.Reshape(output_shape))

    return model


def check_train(
    tflite_model: Path,
    tfrecord: Path,
    train_params: TrainingParameters = MockTrainingParameters(),
    use_unmodified_model: bool = False,
    quantized: bool = False,
) -> None:
    """Test the train() function."""
    with TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir, "out.tflite")
        mock_rewrite = FullyConnectedRewrite(
            name="replace",
            rewrite_fn=fc_rewrite,
        )
        is_qat = isinstance(mock_rewrite, QATRewrite)
        result = train(
            source_model=str(tflite_model),
            unmodified_model=str(tflite_model) if use_unmodified_model else None,
            output_model=str(output_file),
            input_tfrec=str(tfrecord),
            rewrite=mock_rewrite,
            input_tensors=["sequential/flatten/Reshape"],
            output_tensors=["StatefulPartitionedCall:0"],
            is_qat=is_qat,
            train_params=train_params,
        )

        assert len(result[0][0]) == 2
        assert all(
            res >= 0.0 for res in result[0][0]
        ), f"Results out of bound: {result}"
        assert output_file.is_file()

        if quantized:
            interpreter = tf.lite.Interpreter(model_path=str(output_file))
            interpreter.allocate_tensors()
            # Check that the quantization parameters are non-zero
            assert all(interpreter.get_output_details()[0]["quantization"])
            assert all(interpreter.get_input_details()[0]["quantization"])
            dtypes = []
            for tensor_detail in interpreter.get_tensor_details():
                dtypes.append(tensor_detail["dtype"])
            assert all(np.issubdtype(dtype, np.integer) for dtype in dtypes)


@pytest.mark.parametrize(
    (
        "batch_size",
        "show_progress",
        "augmentation_preset",
        "lr_schedule",
        "use_unmodified_model",
        "num_procs",
    ),
    (
        (1, False, AUGMENTATION_PRESETS["none"], "cosine", False, 2),
        (32, True, AUGMENTATION_PRESETS["gaussian"], "late", True, 1),
        (2, False, AUGMENTATION_PRESETS["mixup"], "constant", True, 0),
        (
            1,
            False,
            AUGMENTATION_PRESETS["mix_gaussian_large"],
            "cosine",
            False,
            2,
        ),
    ),
)
def test_train_fp32(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
    batch_size: int,
    show_progress: bool,
    augmentation_preset: tuple[float | None, float | None],
    lr_schedule: LearningRateSchedule,
    use_unmodified_model: bool,
    num_procs: int,
) -> None:
    """Test the train() function with valid parameters."""
    check_train(
        tflite_model=test_tflite_model_fp32,
        tfrecord=test_tfrecord_fp32,
        train_params=MockTrainingParameters(
            batch_size=batch_size,
            show_progress=show_progress,
            augmentations=augmentation_preset,
            learning_rate_schedule=lr_schedule,
            num_procs=num_procs,
        ),
        use_unmodified_model=use_unmodified_model,
    )


@pytest.mark.parametrize(
    (
        "batch_size",
        "show_progress",
        "augmentation_preset",
        "lr_schedule",
        "use_unmodified_model",
        "num_procs",
    ),
    (
        (1, False, AUGMENTATION_PRESETS["none"], "cosine", False, 2),
        (32, True, AUGMENTATION_PRESETS["gaussian"], "late", True, 1),
        (2, False, AUGMENTATION_PRESETS["mixup"], "constant", True, 0),
        (
            1,
            False,
            AUGMENTATION_PRESETS["mix_gaussian_large"],
            "cosine",
            False,
            2,
        ),
    ),
)
def test_train_int8(
    test_tflite_model: Path,
    test_tfrecord: Path,
    batch_size: int,
    show_progress: bool,
    augmentation_preset: tuple[float | None, float | None],
    lr_schedule: LearningRateSchedule,
    use_unmodified_model: bool,
    num_procs: int,
) -> None:
    """Test the train() function with valid parameters."""
    check_train(
        tflite_model=test_tflite_model,
        tfrecord=test_tfrecord,
        train_params=MockTrainingParameters(
            batch_size=batch_size,
            show_progress=show_progress,
            augmentations=augmentation_preset,
            learning_rate_schedule=lr_schedule,
            num_procs=num_procs,
        ),
        use_unmodified_model=use_unmodified_model,
        quantized=True,
    )


def test_train_invalid_schedule(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
) -> None:
    """Test the train() function with an invalid schedule."""
    with pytest.raises(AssertionError):
        check_train(
            tflite_model=test_tflite_model_fp32,
            tfrecord=test_tfrecord_fp32,
            train_params=MockTrainingParameters(
                learning_rate_schedule="unknown_schedule",
            ),
        )


def test_train_invalid_augmentation(
    test_tflite_model_fp32: Path,
    test_tfrecord_fp32: Path,
) -> None:
    """Test the train() function with an invalid augmentation."""
    with pytest.raises(AssertionError):
        check_train(
            tflite_model=test_tflite_model_fp32,
            tfrecord=test_tfrecord_fp32,
            train_params=MockTrainingParameters(
                augmentations=(1.0, 2.0, 3.0),
            ),
        )


def test_mixup() -> None:
    """Test the mixup() function."""
    src = np.array((1, 2, 3))
    dst = mixup(rng=np.random.default_rng(123), batch=src)
    assert src.shape == dst.shape
    assert np.all(dst >= 0.0)
    assert np.all(dst <= 3.0)


@pytest.mark.parametrize(
    "augmentations, expected_error",
    [
        (AUGMENTATION_PRESETS["none"], does_not_raise()),
        (AUGMENTATION_PRESETS["mix_gaussian_large"], does_not_raise()),
        ((None,) * 3, pytest.raises(AssertionError)),
    ],
)
def test_augment_fn_twins(augmentations: tuple, expected_error: Any) -> None:
    """Test function augment_fn()."""
    dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2, 3], "b": [4, 5, 6]})
    with expected_error:
        fn_twins = augment_fn_twins(dataset, augmentations)  # type: ignore
        assert len(fn_twins) == 2


def test_train_checkpoint(
    test_tflite_model: Path,
    test_tfrecord: Path,
) -> None:
    """Test the train() function with valid checkpoint parameters."""
    check_train(
        tflite_model=test_tflite_model,
        tfrecord=test_tfrecord,
        train_params=MockTrainingParameters(steps=64, checkpoint_at=[24, 32]),
        use_unmodified_model=False,
        quantized=True,
    )
