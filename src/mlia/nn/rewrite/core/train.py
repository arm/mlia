# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Sequential trainer."""
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
from __future__ import annotations

import logging
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Generator as GeneratorType
from typing import get_args
from typing import Literal

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from numpy.random import Generator

from mlia.nn.rewrite.core.extract import extract
from mlia.nn.rewrite.core.extract import ExtractPaths
from mlia.nn.rewrite.core.graph_edit.diff import diff_stats
from mlia.nn.rewrite.core.graph_edit.join import join_models
from mlia.nn.rewrite.core.graph_edit.record import record_model
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_count
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel
from mlia.nn.tensorflow.config import TFLiteModel
from mlia.nn.tensorflow.tflite_convert import convert_to_tflite
from mlia.nn.tensorflow.tflite_graph import load_fb
from mlia.nn.tensorflow.tflite_graph import save_fb
from mlia.utils.logging import log_action


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logger = logging.getLogger(__name__)

AUGMENTATION_PRESETS = {
    "none": (None, None),
    "gaussian": (None, 1.0),
    "mixup": (1.0, None),
    "mixout": (1.6, None),
    "mix_gaussian_large": (2.0, 1.0),
    "mix_gaussian_small": (1.6, 0.3),
}

LearningRateSchedule = Literal["cosine", "late", "constant"]
LEARNING_RATE_SCHEDULES = get_args(LearningRateSchedule)


@dataclass
class TrainingParameters:
    """Define default parameters for the training."""

    augmentations: tuple[float | None, float | None] = AUGMENTATION_PRESETS["gaussian"]
    batch_size: int = 32
    steps: int = 48000
    learning_rate: float = 1e-3
    learning_rate_schedule: LearningRateSchedule = "cosine"
    num_procs: int = 1
    num_threads: int = 0
    show_progress: bool = True
    checkpoint_at: list | None = None


def train(
    source_model: str,
    unmodified_model: Any,
    output_model: str,
    input_tfrec: str,
    replace_fn: Callable,
    input_tensors: list,
    output_tensors: list,
    train_params: TrainingParameters = TrainingParameters(),
) -> Any:
    """Extract and train a model, and return the results."""
    if unmodified_model:
        unmodified_model_dir = (
            tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        )
        unmodified_model_dir_path = unmodified_model_dir.name
        extract(
            unmodified_model_dir_path,
            source_model,
            input_tfrec,
            input_tensors,
            output_tensors,
            dequantize_output=True,
        )
    else:
        unmodified_model_dir = None
        unmodified_model_dir_path = None

    results = []
    with tempfile.TemporaryDirectory() as train_dir:
        extract(
            train_dir,
            source_model,
            input_tfrec,
            input_tensors,
            output_tensors,
            num_procs=train_params.num_procs,
            num_threads=train_params.num_threads,
            dequantize_output=True,
        )

        tflite_filenames = train_in_dir(
            train_dir=train_dir,
            baseline_dir=unmodified_model_dir_path,
            output_filename=Path(train_dir, "new.tflite"),
            replace_fn=replace_fn,
            train_params=train_params,
        )

        for i, filename in enumerate(tflite_filenames):
            results.append(
                eval_in_dir(
                    train_dir,
                    filename,
                    train_params.num_procs,
                    train_params.num_threads,
                )
            )

            if output_model:
                if i + 1 < len(tflite_filenames):
                    # Append the same _@STEPS.tflite postfix used by intermediate
                    # checkpoints for all but the last output
                    postfix = filename.split("_@")[-1]
                    output_filename = output_model.split(".tflite")[0] + postfix
                else:
                    output_filename = output_model
                join_in_dir(train_dir, filename, output_filename)

        # Assess the output diff between the parts after the rewrite subgraph
        # in original and optimized model
        optimized_end_path = Path(train_dir, "optimized_end.tfrec")
        end_path = Path(train_dir, "end.tfrec")

        record_model(
            str(input_tfrec),
            output_filename,
            optimized_end_path,
            num_procs=train_params.num_procs,
            num_threads=train_params.num_threads,
        )
        mae, nrmse = diff_stats(end_path, str(optimized_end_path))

    if unmodified_model_dir:
        cast(tempfile.TemporaryDirectory, unmodified_model_dir).cleanup()

    return (results if train_params.checkpoint_at else results[0]), [
        mae,
        nrmse,
    ]  # only return a list if multiple checkpoints are asked for


def eval_in_dir(
    target_dir: str,
    new_part: str,
    num_procs: int = 1,
    num_threads: int = 0,
) -> tuple:
    """Evaluate a model in a given directory."""
    model_input_path = Path(target_dir, "input_orig.tfrec")
    model_output_path = Path(target_dir, "output_orig.tfrec")
    model_input = (
        model_input_path
        if model_input_path.exists()
        else ExtractPaths.tfrec.input(target_dir, False)
    )
    output = (
        model_output_path
        if model_output_path.exists()
        else ExtractPaths.tfrec.output(target_dir, False)
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        predict = Path(tmp_dir, "predict.tfrec")
        record_model(
            str(model_input),
            new_part,
            str(predict),
            num_procs=num_procs,
            num_threads=num_threads,
        )
        mae, nrmse = diff_stats(str(output), str(predict))

    return mae, nrmse


def join_in_dir(model_dir: str, new_part: str, output_model: str) -> None:
    """Join two models in a given directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        new_end = Path(tmp_dir, "new_end.tflite")
        join_models(new_part, ExtractPaths.tflite.end(model_dir), new_end)
        join_models(ExtractPaths.tflite.start(model_dir), new_end, output_model)


def _get_io_tensors(model: TFLiteModel) -> tuple[str, str]:
    assert (
        len(model.input_tensors()) == 1
    ), f"Can only train replacements with a single input tensor right now, \
        found {model.input_tensors()}"

    assert (
        len(model.output_tensors()) == 1
    ), f"Can only train replacements with a single output tensor right now, \
        found {model.output_tensors()}"

    input_name = model.input_tensors()[0]
    output_name = model.output_tensors()[0]
    return (input_name, output_name)


def _check_model_compatibility(teacher: TFLiteModel, replace: TFLiteModel) -> None:
    """Assert that teacher and replaced sub-graph are compatible."""
    assert len(teacher.shape_from_name) == len(
        replace.shape_from_name
    ), f"Baseline and train models must have the same number of inputs and outputs. \
        Teacher: {teacher.shape_from_name}\nTrain dir: {replace.shape_from_name}"

    assert all(
        tn == rn and (ts[1:] == rs[1:]).all()
        for (tn, ts), (rn, rs) in zip(
            teacher.shape_from_name.items(), replace.shape_from_name.items()
        )
    ), "Baseline and train models must have the same input and output shapes for the \
        subgraph being replaced. Teacher: {teacher.shape_from_name}\n \
        Train dir: {replace.shape_from_name}"


def set_up_data_pipeline(
    teacher: TFLiteModel,
    replace: TFLiteModel,
    train_dir: str,
    augmentations: tuple[float | None, float | None],
    steps: int,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Create a data pipeline for training of the replacement model."""
    _check_model_compatibility(teacher, replace)

    input_name, output_name = _get_io_tensors(teacher)

    model_is_quantized = replace.is_tensor_quantized(name=input_name)

    input_filename = ExtractPaths.tfrec.input(train_dir, model_is_quantized)
    total = numpytf_count(str(input_filename))
    dict_inputs = numpytf_read(str(input_filename))

    inputs = dict_inputs.map(lambda d: tf.squeeze(d[input_name], axis=0))

    steps_per_epoch = math.ceil(total / batch_size)
    epochs = int(math.ceil(steps / steps_per_epoch))
    logger.info(
        "Training on %d items for %d steps (%d epochs with batch size %d)",
        total,
        epochs * steps_per_epoch,
        epochs,
        batch_size,
    )

    teacher_dir = Path(teacher.model_path).parent
    if any(augmentations):
        # Map the teacher inputs here because the augmentation stage passes these
        # through a TFLite model to get the outputs
        teacher_outputs = numpytf_read(
            str(ExtractPaths.tfrec.input(teacher_dir, model_is_quantized))
        ).map(lambda d: tf.squeeze(d[input_name], axis=0))
    else:
        teacher_outputs = numpytf_read(
            str(ExtractPaths.tfrec.output(teacher_dir, model_is_quantized))
        ).map(lambda d: tf.squeeze(d[output_name], axis=0))

    dataset = tf.data.Dataset.zip((inputs, teacher_outputs))
    if epochs > 1:
        dataset = dataset.cache()
    dataset = dataset.shuffle(total).repeat().batch(batch_size)

    if any(augmentations):
        augment_train, augment_teacher = augment_fn_twins(dict_inputs, augmentations)

        def get_augment_results(
            train: Any, teach: Any  # pylint: disable=redefined-outer-name
        ) -> tuple:
            """Return results of train and teach based on augmentations."""
            augmented_train = augment_train({input_name: train})[input_name]
            # If augmentation of the input is enabled, we need to re-generate
            # the reference output by running the augmented data through the
            # teacher model.
            if model_is_quantized:
                # If the input model is quantized we have to additionally
                # - quantize the augmented data before running it through the
                #   (quantized) teacher model
                # - de-quantize the output for the training.
                augmented_teach = teacher.dequantize_outputs(
                    teacher(
                        teacher.quantize_inputs(augment_teacher({input_name: teach}))
                    )
                )[output_name]
            else:
                augmented_teach = teacher(augment_teacher({input_name: teach}))[
                    output_name
                ]
            return (augmented_train, augmented_teach)

        dataset = dataset.map(
            lambda augment_train, augment_teach: tf.py_function(
                get_augment_results,
                inp=[augment_train, augment_teach],
                Tout=[tf.float32, tf.float32],
            )
        )

    # Restore data shapes of the dataset as they are set to unknown per default
    # and get lost during augmentation with tf.py_function.
    shape_in, shape_out = (
        teacher.shape_from_name[name].tolist() for name in (input_name, output_name)
    )
    for shape in (shape_in, shape_out):
        shape[0] = None  # set dynamic batch size

    def restore_shapes(input_: Any, output: Any) -> tuple[Any, Any]:
        input_.set_shape(shape_in)
        output.set_shape(shape_out)
        return input_, output

    dataset = dataset.map(restore_shapes)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_in_dir(
    train_dir: str,
    baseline_dir: Any,
    output_filename: Path,
    replace_fn: Callable,
    train_params: TrainingParameters = TrainingParameters(),
) -> list[str]:
    """Train a replacement for replace.tflite using the input.tfrec \
        and output.tfrec in train_dir.

    If baseline_dir is provided, train the replacement to match baseline
    outputs for train_dir inputs. Result saved as new.tflite in train_dir.
    """
    teacher_dir = baseline_dir if baseline_dir else train_dir
    teacher = ParallelTFLiteModel(
        ExtractPaths.tflite.replace(teacher_dir),
        train_params.num_procs,
        train_params.num_threads,
        batch_size=train_params.batch_size,
    )
    replace = TFLiteModel(ExtractPaths.tflite.replace(train_dir))

    input_name, output_name = _get_io_tensors(teacher)

    model_is_quantized = replace.is_tensor_quantized(name=input_name)

    if model_is_quantized:
        replace.check_datatypes(np.int8)

    dataset = set_up_data_pipeline(
        teacher,
        replace,
        train_dir,
        augmentations=train_params.augmentations,
        steps=train_params.steps,
        batch_size=train_params.batch_size,
    )

    input_shape = teacher.shape_from_name[input_name][1:]
    output_shape = teacher.shape_from_name[output_name][1:]

    model = replace_fn(input_shape, output_shape)

    optimizer = tf.keras.optimizers.Nadam(learning_rate=train_params.learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    if model_is_quantized:
        model = tfmot.quantization.keras.quantize_model(model)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["mae"])

    logger.info(model.summary())

    steps_so_far = 0

    def cosine_decay(
        epoch_step: int, logs: Any  # pylint: disable=unused-argument
    ) -> None:
        """Cosine decay from learning rate at start of the run to zero at the end."""
        current_step = epoch_step + steps_so_far
        cd_learning_rate = (
            train_params.learning_rate
            * (math.cos(math.pi * current_step / train_params.steps) + 1)
            / 2.0
        )
        tf.keras.backend.set_value(optimizer.learning_rate, cd_learning_rate)

    def late_decay(
        epoch_step: int, logs: Any  # pylint: disable=unused-argument
    ) -> None:
        """Constant until the last 20% of the run, then linear decay to zero."""
        current_step = epoch_step + steps_so_far
        steps_remaining = train_params.steps - current_step
        decay_length = train_params.steps // 5
        decay_fraction = min(steps_remaining, decay_length) / decay_length
        ld_learning_rate = train_params.learning_rate * decay_fraction
        tf.keras.backend.set_value(optimizer.learning_rate, ld_learning_rate)

    assert train_params.learning_rate_schedule in LEARNING_RATE_SCHEDULES, (
        f'Learning rate schedule "{train_params.learning_rate_schedule}" '
        f"not implemented - expected one of {LEARNING_RATE_SCHEDULES}."
    )
    if train_params.learning_rate_schedule == "cosine":
        callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_begin=cosine_decay)]
    elif train_params.learning_rate_schedule == "late":
        callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_begin=late_decay)]
    elif train_params.learning_rate_schedule == "constant":
        callbacks = []

    output_filenames = []
    checkpoints = (train_params.checkpoint_at if train_params.checkpoint_at else []) + [
        train_params.steps
    ]

    while steps_so_far < train_params.steps:
        steps_to_train = checkpoints.pop(0) - steps_so_far
        lr_start = optimizer.learning_rate.numpy()
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=steps_to_train,
            callbacks=callbacks,
            verbose=train_params.show_progress,
        )
        steps_so_far += steps_to_train
        logger.info(
            "lr decayed from %f to %f over %d steps",
            lr_start,
            optimizer.learning_rate.numpy(),
            steps_to_train,
        )

        if steps_so_far < train_params.steps:
            filename, ext = Path(output_filename).parts[1:]
            checkpoint_filename = filename + (f"_@{steps_so_far}") + ext
        else:
            checkpoint_filename = str(output_filename)
        with log_action(
            f"{steps_so_far}/{train_params.steps}: Saved as {checkpoint_filename}"
        ):
            save_as_tflite(
                model,
                checkpoint_filename,
                input_name,
                replace.shape_from_name[input_name],
                output_name,
                replace.shape_from_name[output_name],
                model_is_quantized,
            )
            output_filenames.append(checkpoint_filename)

    teacher.close()
    return output_filenames


def save_as_tflite(
    keras_model: tf.keras.Model,
    filename: str,
    input_name: str,
    input_shape: list,
    output_name: str,
    output_shape: list,
    model_is_quantized: bool = False,
) -> None:
    """Save Keras model as TFLite file."""

    @contextmanager
    def fixed_input(keras_model: tf.keras.Model, tmp_shape: list) -> GeneratorType:
        """Fix the input shape of the Keras model temporarily.

        This avoids artifacts during conversion to TensorFlow Lite.
        """
        orig_shape = keras_model.input.shape
        keras_model.input.set_shape(tf.TensorShape(tmp_shape))
        try:
            yield keras_model
        finally:
            # Restore original shape to not interfere with further training
            keras_model.input.set_shape(orig_shape)

    with fixed_input(keras_model, input_shape) as fixed_model:
        convert_to_tflite(fixed_model, model_is_quantized, Path(filename))

    # Now fix the shapes and names to match those we expect
    flatbuffer = load_fb(filename)
    i = flatbuffer.subgraphs[0].inputs[0]
    flatbuffer.subgraphs[0].tensors[i].shape = np.array(input_shape, dtype=np.int32)
    flatbuffer.subgraphs[0].tensors[i].name = input_name.encode("utf-8")
    output = flatbuffer.subgraphs[0].outputs[0]
    flatbuffer.subgraphs[0].tensors[output].shape = np.array(
        output_shape, dtype=np.int32
    )
    flatbuffer.subgraphs[0].tensors[output].name = output_name.encode("utf-8")
    save_fb(flatbuffer, filename)


def augment_fn_twins(
    inputs: tf.data.Dataset, augmentations: tuple[float | None, float | None]
) -> Any:
    """Return a pair of twinned augmentation functions with the same sequence \
        of random numbers."""
    seed = np.random.randint(2**32 - 1)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    return augment_fn(inputs, augmentations, rng1), augment_fn(
        inputs, augmentations, rng2
    )


def augment_fn(
    inputs: Any, augmentations: tuple[float | None, float | None], rng: Generator
) -> Any:
    """Augmentation module."""
    assert len(augmentations) == 2, (
        f"Unexpected number of augmentation parameters: {len(augmentations)} "
        "(must be 2)"
    )

    mixup_strength, gaussian_strength = augmentations

    augments = []

    if mixup_strength:
        mixup_range = (0.5 - mixup_strength / 2, 0.5 + mixup_strength / 2)

        def mixup_augment(augment_dict: dict) -> dict:
            return {
                k: mixup(rng, v.numpy(), mixup_range) for k, v in augment_dict.items()
            }

        augments.append(mixup_augment)

    if gaussian_strength:
        values = defaultdict(list)
        for numpy_dict in inputs.as_numpy_iterator():
            for key, value in numpy_dict.items():
                values[key].append(value)
        noise_scale = {
            k: np.std(v, axis=0).astype(np.float32) for k, v in values.items()
        }

        def gaussian_strength_augment(augment_dict: dict) -> dict:
            return {
                k: v
                + rng.standard_normal(v.shape).astype(np.float32)
                * gaussian_strength
                * noise_scale[k]
                for k, v in augment_dict.items()
            }

        augments.append(gaussian_strength_augment)

    if len(augments) == 0:
        return lambda x: x
    if len(augments) == 1:
        return augments[0]
    if len(augments) == 2:
        return lambda x: augments[1](augments[0](x))

    raise RuntimeError(
        "Unexpected number of augmentation functions (must be <=2): " f"{len(augments)}"
    )


def mixup(rng: Generator, batch: Any, beta_range: tuple = (0.0, 1.0)) -> Any:
    """Each tensor in the batch becomes a linear combination of it \
        and one other tensor."""
    batch_a = batch
    batch_b = np.array(batch)
    rng.shuffle(batch_b)  # randomly pair up tensors in the batch
    # random mixing coefficient for each pair
    beta = rng.uniform(
        low=beta_range[0], high=beta_range[1], size=batch.shape[0]
    ).astype(np.float32)
    return (batch_a.T * beta).T + (
        batch_b.T * (1.0 - beta)
    ).T  # return linear combinations
