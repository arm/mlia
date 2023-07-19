# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module mlia.nn.rewrite.train."""
# pylint: disable=too-many-arguments
from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from mlia.nn.rewrite.core.train import augment_fn_twins
from mlia.nn.rewrite.core.train import AUGMENTATION_PRESETS
from mlia.nn.rewrite.core.train import LearningRateSchedule
from mlia.nn.rewrite.core.train import mixup
from mlia.nn.rewrite.core.train import train
from mlia.nn.rewrite.core.train import TrainingParameters
from tests.utils.rewrite import TestTrainingParameters


def replace_fully_connected_with_conv(
    input_shape: Any, output_shape: Any
) -> tf.keras.Model:
    """Get a replacement model for the fully connected layer."""
    for name, shape in {
        "Input": input_shape,
        "Output": output_shape,
    }.items():
        if len(shape) != 1:
            raise RuntimeError(f"{name}: shape (N,) expected, but it is {input_shape}.")

    model = tf.keras.Sequential(name="RewriteModel")
    model.add(tf.keras.Input(input_shape))
    model.add(tf.keras.layers.Reshape((1, 1, input_shape[0])))
    model.add(tf.keras.layers.Conv2D(filters=output_shape[0], kernel_size=(1, 1)))
    model.add(tf.keras.layers.Reshape(output_shape))

    return model


def check_train(
    tflite_model: Path,
    tfrecord: Path,
    train_params: TrainingParameters = TestTrainingParameters(),
    use_unmodified_model: bool = False,
) -> None:
    """Test the train() function."""
    with TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir, "out.tfrecord")
        result = train(
            source_model=str(tflite_model),
            unmodified_model=str(tflite_model) if use_unmodified_model else None,
            output_model=str(output_file),
            input_tfrec=str(tfrecord),
            replace_fn=replace_fully_connected_with_conv,
            input_tensors=["sequential/flatten/Reshape"],
            output_tensors=["StatefulPartitionedCall:0"],
            train_params=train_params,
        )
        assert len(result) == 2
        assert all(res >= 0.0 for res in result), f"Results out of bound: {result}"
        assert output_file.is_file()


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
def test_train(
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
        train_params=TestTrainingParameters(
            batch_size=batch_size,
            show_progress=show_progress,
            augmentations=augmentation_preset,
            learning_rate_schedule=lr_schedule,
            num_procs=num_procs,
        ),
        use_unmodified_model=use_unmodified_model,
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
            train_params=TestTrainingParameters(
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
            train_params=TestTrainingParameters(
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
