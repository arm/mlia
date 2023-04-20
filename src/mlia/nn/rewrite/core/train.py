# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Sequential trainer."""
# pylint: disable=too-many-arguments, too-many-instance-attributes,
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
from __future__ import annotations

import logging
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import get_args
from typing import Literal

import numpy as np
import tensorflow as tf
from numpy.random import Generator

from mlia.nn.rewrite.core.extract import extract
from mlia.nn.rewrite.core.graph_edit.diff import diff_stats
from mlia.nn.rewrite.core.graph_edit.join import join_models
from mlia.nn.rewrite.core.graph_edit.record import record_model
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_count
from mlia.nn.rewrite.core.utils.numpy_tfrecord import numpytf_read
from mlia.nn.rewrite.core.utils.numpy_tfrecord import TFLiteModel
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel
from mlia.nn.rewrite.core.utils.utils import load
from mlia.nn.rewrite.core.utils.utils import save
from mlia.utils.logging import log_action


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logger = logging.getLogger(__name__)

augmentation_presets = {
    "none": (None, None),
    "gaussian": (None, 1.0),
    "mixup": (1.0, None),
    "mixout": (1.6, None),
    "mix_gaussian_large": (2.0, 1.0),
    "mix_gaussian_small": (1.6, 0.3),
}

LearningRateSchedule = Literal["cosine", "late", "constant"]
LEARNING_RATE_SCHEDULES = get_args(LearningRateSchedule)


def train(
    source_model: str,
    unmodified_model: Any,
    output_model: str,
    input_tfrec: str,
    replace_fn: Callable,
    input_tensors: list,
    output_tensors: list,
    augment: tuple[float | None, float | None],
    steps: int,
    learning_rate: float,
    batch_size: int,
    verbose: bool,
    show_progress: bool,
    learning_rate_schedule: LearningRateSchedule = "cosine",
    checkpoint_at: list | None = None,
    num_procs: int = 1,
    num_threads: int = 0,
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
            num_procs=num_procs,
            num_threads=num_threads,
        )

        tflite_filenames = train_in_dir(
            train_dir,
            unmodified_model_dir_path,
            Path(train_dir, "new.tflite"),
            replace_fn,
            augment,
            steps,
            learning_rate,
            batch_size,
            checkpoint_at=checkpoint_at,
            verbose=verbose,
            show_progress=show_progress,
            num_procs=num_procs,
            num_threads=num_threads,
            schedule=learning_rate_schedule,
        )

        for i, filename in enumerate(tflite_filenames):
            results.append(eval_in_dir(train_dir, filename, num_procs, num_threads))

            if output_model:
                if i + 1 < len(tflite_filenames):
                    # Append the same _@STEPS.tflite postfix used by intermediate
                    # checkpoints for all but the last output
                    postfix = filename.split("_@")[-1]
                    output_filename = output_model.split(".tflite")[0] + postfix
                else:
                    output_filename = output_model
                join_in_dir(train_dir, filename, output_filename)

    if unmodified_model_dir:
        cast(tempfile.TemporaryDirectory, unmodified_model_dir).cleanup()

    return (
        results if checkpoint_at else results[0]
    )  # only return a list if multiple checkpoints are asked for


def eval_in_dir(
    target_dir: str, new_part: str, num_procs: int = 1, num_threads: int = 0
) -> tuple:
    """Evaluate a model in a given directory."""
    model_input_path = Path(target_dir, "input_orig.tfrec")
    model_output_path = Path(target_dir, "output_orig.tfrec")
    model_input = (
        model_input_path
        if model_input_path.exists()
        else Path(target_dir, "input.tfrec")
    )
    output = (
        model_output_path
        if model_output_path.exists()
        else Path(target_dir, "output.tfrec")
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
        join_models(new_part, Path(model_dir, "end.tflite"), new_end)
        join_models(Path(model_dir, "start.tflite"), new_end, output_model)


def train_in_dir(
    train_dir: str,
    baseline_dir: Any,
    output_filename: Path,
    replace_fn: Callable,
    augmentations: tuple[float | None, float | None],
    steps: int,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    checkpoint_at: list | None = None,
    schedule: str = "cosine",
    verbose: bool = False,
    show_progress: bool = False,
    num_procs: int = 0,
    num_threads: int = 1,
) -> list:
    """Train a replacement for replace.tflite using the input.tfrec \
        and output.tfrec in train_dir.

    If baseline_dir is provided, train the replacement to match baseline
    outputs for train_dir inputs. Result saved as new.tflite in train_dir.
    """
    teacher_dir = baseline_dir if baseline_dir else train_dir
    teacher = ParallelTFLiteModel(
        f"{teacher_dir}/replace.tflite", num_procs, num_threads, batch_size=batch_size
    )
    replace = TFLiteModel(f"{train_dir}/replace.tflite")
    assert (
        len(teacher.input_tensors()) == 1
    ), f"Can only train replacements with a single input tensor right now, \
        found {teacher.input_tensors()}"

    assert (
        len(teacher.output_tensors()) == 1
    ), f"Can only train replacements with a single output tensor right now, \
        found {teacher.output_tensors()}"

    input_name = teacher.input_tensors()[0]
    output_name = teacher.output_tensors()[0]

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

    input_filename = Path(train_dir, "input.tfrec")
    total = numpytf_count(str(input_filename))
    dict_inputs = numpytf_read(str(input_filename))
    inputs = dict_inputs.map(lambda d: tf.squeeze(d[input_name], axis=0))
    if any(augmentations):
        # Map the teacher inputs here because the augmentation stage passes these
        # through a TFLite model to get the outputs
        teacher_outputs = numpytf_read(str(Path(teacher_dir, "input.tfrec"))).map(
            lambda d: tf.squeeze(d[input_name], axis=0)
        )
    else:
        teacher_outputs = numpytf_read(str(Path(teacher_dir, "output.tfrec"))).map(
            lambda d: tf.squeeze(d[output_name], axis=0)
        )

    steps_per_epoch = math.ceil(total / batch_size)
    epochs = int(math.ceil(steps / steps_per_epoch))
    if verbose:
        logger.info(
            "Training on %d items for %d steps (%d epochs with batch size %d)",
            total,
            epochs * steps_per_epoch,
            epochs,
            batch_size,
        )

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
            return (
                augment_train({input_name: train})[input_name],
                teacher(augment_teacher({input_name: teach}))[output_name],
            )

        dataset = dataset.map(
            lambda augment_train, augment_teach: tf.py_function(
                get_augment_results,
                inp=[augment_train, augment_teach],
                Tout=[tf.float32, tf.float32],
            )
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    input_shape = teacher.shape_from_name[input_name][1:]
    output_shape = teacher.shape_from_name[output_name][1:]
    model = replace_fn(input_shape, output_shape)

    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)

    if verbose:
        model.summary()

    steps_so_far = 0

    def cosine_decay(
        epoch_step: int, logs: Any  # pylint: disable=unused-argument
    ) -> None:
        """Cosine decay from learning rate at start of the run to zero at the end."""
        current_step = epoch_step + steps_so_far
        cd_learning_rate = (
            learning_rate * (math.cos(math.pi * current_step / steps) + 1) / 2.0
        )
        tf.keras.backend.set_value(optimizer.learning_rate, cd_learning_rate)

    def late_decay(
        epoch_step: int, logs: Any  # pylint: disable=unused-argument
    ) -> None:
        """Constant until the last 20% of the run, then linear decay to zero."""
        current_step = epoch_step + steps_so_far
        steps_remaining = steps - current_step
        decay_length = steps // 5
        decay_fraction = min(steps_remaining, decay_length) / decay_length
        ld_learning_rate = learning_rate * decay_fraction
        tf.keras.backend.set_value(optimizer.learning_rate, ld_learning_rate)

    if schedule == "cosine":
        callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_begin=cosine_decay)]
    elif schedule == "late":
        callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_begin=late_decay)]
    elif schedule == "constant":
        callbacks = []
    else:
        assert schedule not in LEARNING_RATE_SCHEDULES
        raise ValueError(
            f'Learning rate schedule "{schedule}" not implemented - '
            f"expected one of {LEARNING_RATE_SCHEDULES}."
        )

    output_filenames = []
    checkpoints = (checkpoint_at if checkpoint_at else []) + [steps]
    while steps_so_far < steps:
        steps_to_train = checkpoints.pop(0) - steps_so_far
        lr_start = optimizer.learning_rate.numpy()
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=steps_to_train,
            callbacks=callbacks,
            verbose=show_progress,
        )
        steps_so_far += steps_to_train
        logger.info(
            "lr decayed from %f to %f over %d steps",
            lr_start,
            optimizer.learning_rate.numpy(),
            steps_to_train,
        )

        if steps_so_far < steps:
            filename, ext = Path(output_filename).parts[1:]
            checkpoint_filename = filename + (f"_@{steps_so_far}") + ext
        else:
            checkpoint_filename = str(output_filename)
        with log_action(f"{steps_so_far}/{steps}: Saved as {checkpoint_filename}"):
            save_as_tflite(
                model,
                checkpoint_filename,
                input_name,
                replace.shape_from_name[input_name],
                output_name,
                replace.shape_from_name[output_name],
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
) -> None:
    """Save Keras model as TFLite file."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    with open(filename, "wb") as file:
        file.write(tflite_model)

    # Now fix the shapes and names to match those we expect
    flatbuffer = load(filename)
    i = flatbuffer.subgraphs[0].inputs[0]
    flatbuffer.subgraphs[0].tensors[i].shape = np.array(input_shape, dtype=np.int32)
    flatbuffer.subgraphs[0].tensors[i].name = input_name.encode("utf-8")
    output = flatbuffer.subgraphs[0].outputs[0]
    flatbuffer.subgraphs[0].tensors[output].shape = np.array(
        output_shape, dtype=np.int32
    )
    flatbuffer.subgraphs[0].tensors[output].name = output_name.encode("utf-8")
    save(flatbuffer, filename)


def augment_fn_twins(
    inputs: dict, augmentations: tuple[float | None, float | None]
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

    if len(augments) == 0:  # pylint: disable=no-else-return
        return lambda x: x
    elif len(augments) == 1:
        return augments[0]
    elif len(augments) == 2:
        return lambda x: augments[1](augments[0](x))
    else:
        assert (
            False
        ), f"Unexpected number of augmentation \
        functions ({len(augments)})"


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
