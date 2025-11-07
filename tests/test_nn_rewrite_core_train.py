# SPDX-FileCopyrightText: Copyright 2023-2025, Arm Limited and/or its affiliates.
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

from mlia.nn.rewrite.core.rewrite import GenericRewrite
from mlia.nn.rewrite.core.rewrite import SparsityRewrite
from mlia.nn.rewrite.core.train import augment_fn
from mlia.nn.rewrite.core.train import augment_fn_twins
from mlia.nn.rewrite.core.train import AUGMENTATION_PRESETS
from mlia.nn.rewrite.core.train import detect_activation_from_rewrite_function
from mlia.nn.rewrite.core.train import LearningRateSchedule
from mlia.nn.rewrite.core.train import mixup
from mlia.nn.rewrite.core.train import train
from mlia.nn.rewrite.core.train import TrainingParameters
from tests.utils.rewrite import MockTrainingParameters


def replace_fully_connected_with_conv(
    input_shape: Any,
    output_shape: Any,
    **kwargs: Any,
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
    if act := kwargs.get("activation"):
        model.add(
            keras.layers.Conv2D(
                filters=output_shape[0], kernel_size=(1, 1), activation=act
            )
        )
    else:
        model.add(keras.layers.Conv2D(filters=output_shape[0], kernel_size=(1, 1)))
    model.add(keras.layers.Reshape(output_shape))

    return model


def check_train(
    tflite_model: Path,
    tfrecord: Path | None,
    train_params: TrainingParameters = MockTrainingParameters(),
    is_qat: bool = False,
    use_unmodified_model: bool = False,
    quantized: bool = False,
    rewrite_specific_params: dict | None = None,
    detect_activation_function: bool = False,
) -> None:
    """Test the train() function."""
    with TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir, "out.tflite")
        mock_rewrite = (
            SparsityRewrite("replace", replace_fully_connected_with_conv)
            if is_qat
            else GenericRewrite("replace", replace_fully_connected_with_conv)
        )
        result = train(
            source_model=str(tflite_model),
            unmodified_model=str(tflite_model) if use_unmodified_model else None,
            output_model=str(output_file),
            input_tfrec=str(tfrecord) if tfrecord else None,
            rewrite=mock_rewrite,
            is_qat=is_qat,
            input_tensors=["sequential/flatten/Reshape"],
            output_tensors=["StatefulPartitionedCall:0"],
            train_params=train_params,
            rewrite_specific_params=rewrite_specific_params,
            detect_activation_function=detect_activation_function,
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


@pytest.mark.slow
@pytest.mark.parametrize(
    (
        "batch_size",
        "show_progress",
        "augmentation_preset",
        "lr_schedule",
        "use_unmodified_model",
        "num_procs",
        "rewrite_specific_params",
        "detect_activation_function",
    ),
    (
        (1, False, AUGMENTATION_PRESETS["none"], "cosine", False, 2, {}, False),
        (32, True, AUGMENTATION_PRESETS["gaussian"], "late", True, 1, {}, True),
        (2, False, AUGMENTATION_PRESETS["mixup"], "constant", True, 0, {}, False),
        (
            1,
            False,
            AUGMENTATION_PRESETS["mix_gaussian_large"],
            "cosine",
            False,
            2,
            {"some_param": "some_value"},
            True,
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
    rewrite_specific_params: dict,
    detect_activation_function: bool,
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
        is_qat=False,
        use_unmodified_model=use_unmodified_model,
        rewrite_specific_params=rewrite_specific_params,
        detect_activation_function=detect_activation_function,
    )


@pytest.mark.slow
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
            steps=64,
            checkpoint_at=[32],
        ),
        is_qat=True,
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
def test_augment_fn_twins(augmentations: Any, expected_error: Any) -> None:
    """Test function augment_fn_twins()."""
    dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2, 3], "b": [4, 5, 6]})
    with expected_error:
        fn_twins = augment_fn_twins(dataset, augmentations)
        assert len(fn_twins) == 2


@pytest.mark.parametrize(
    "augmentations",
    (
        AUGMENTATION_PRESETS["none"],
        AUGMENTATION_PRESETS["gaussian"],
        AUGMENTATION_PRESETS["mix_gaussian_large"],
    ),
)
def test_augment_fn(augmentations: Any) -> None:
    """Test function augment_fn()."""
    seed = np.random.randint(2**32 - 1)
    rng = np.random.default_rng(seed)
    dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2, 3], "b": [4, 5, 6]})
    augment_function = augment_fn(dataset, augmentations, rng)
    assert (
        augment_function({"a": tf.constant([1, 2, 3]), "b": tf.constant([4, 5, 6])})
        is not None
    )


@pytest.mark.slow
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


def test_detect_activation_from_rewrite_function_no_activation(
    caplog: pytest.LogCaptureFixture, test_tflite_no_act_model: Path
) -> None:
    """
    Test function detect_activation_from_rewrite_function()
    with a model with no activation functions.
    """
    caplog.set_level(level=20)
    activation = detect_activation_from_rewrite_function(
        test_tflite_no_act_model.as_posix()
    )
    log_records = caplog.get_records(when="call")
    logging_messages = [x.message for x in log_records if x.levelno == 20]
    assert activation == "relu"
    assert (
        "No activation function specified, setting activation function to ReLU"
        in logging_messages
    )


def test_detect_activation_from_rewrite_function_relu_activation(
    caplog: pytest.LogCaptureFixture, test_tflite_model: Path
) -> None:
    """
    Test function detect_activation_from_rewrite_function()
    with a model with ReLU activation functions.
    """
    caplog.set_level(level=20)
    activation = detect_activation_from_rewrite_function(test_tflite_model.as_posix())
    log_records = caplog.get_records(when="call")
    logging_messages = [x.message for x in log_records if x.levelno == 20]
    assert activation == "relu"
    assert (
        "No activation function specified, setting activation function "
        "to most common activation detected in rewrite graph: relu" in logging_messages
    )


@pytest.mark.slow
def test_train_none_tf_record(
    test_tflite_model: Path,
) -> None:
    """Test the train() function with valid parameters and no dataset."""
    check_train(
        tflite_model=test_tflite_model,
        tfrecord=None,
        train_params=MockTrainingParameters(),
        quantized=True,
    )
