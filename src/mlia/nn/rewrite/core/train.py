# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import math
import os
import tempfile
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    from tensorflow.keras.optimizers.schedules import CosineDecay
except ImportError:
    # In TF 2.4 CosineDecay was still experimental
    from tensorflow.keras.experimental import CosineDecay

import numpy as np
from mlia.nn.rewrite.core.utils.numpy_tfrecord import (
    NumpyTFReader,
    NumpyTFWriter,
    TFLiteModel,
    numpytf_count,
)
from mlia.nn.rewrite.core.utils.parallel import ParallelTFLiteModel
from mlia.nn.rewrite.core.graph_edit.record import record_model
from mlia.nn.rewrite.core.utils.utils import load, save
from mlia.nn.rewrite.core.extract import extract
from mlia.nn.rewrite.core.graph_edit.join import join_models
from mlia.nn.rewrite.core.graph_edit.diff import diff_stats


augmentation_presets = {
    "none": (None, None),
    "gaussian": (None, 1.0),
    "mixup": (1.0, None),
    "mixout": (1.6, None),
    "mix_gaussian_large": (2.0, 1.0),
    "mix_gaussian_small": (1.6, 0.3),
}

learning_rate_schedules = {"cosine", "late", "constant"}


def train(
    source_model,
    unmodified_model,
    output_model,
    input_tfrec,
    replace_fn,
    input_tensors,
    output_tensors,
    augment,
    steps,
    lr,
    batch_size,
    verbose,
    show_progress,
    learning_rate_schedule="cosine",
    checkpoint_at=None,
    checkpoint_decay_steps=0,
    num_procs=1,
    num_threads=0,
):
    if unmodified_model:
        unmodified_model_dir = tempfile.TemporaryDirectory()
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
        p = lambda file: os.path.join(train_dir, file)

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
            p("new.tflite"),
            replace_fn,
            augment,
            steps,
            lr,
            batch_size,
            checkpoint_at=checkpoint_at,
            checkpoint_decay_steps=checkpoint_decay_steps,
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
                    # Append the same _@STEPS.tflite postfix used by intermediate checkpoints for all but the last output
                    postfix = filename.split("_@")[-1]
                    output_filename = output_model.split(".tflite")[0] + postfix
                else:
                    output_filename = output_model
                join_in_dir(train_dir, filename, output_filename)

    if unmodified_model_dir:
        unmodified_model_dir.cleanup()

    return (
        results if checkpoint_at else results[0]
    )  # only return a list if multiple checkpoints are asked for


def eval_in_dir(dir, new_part, num_procs=1, num_threads=0):
    p = lambda file: os.path.join(dir, file)
    input = (
        p("input_orig.tfrec")
        if os.path.exists(p("input_orig.tfrec"))
        else p("input.tfrec")
    )
    output = (
        p("output_orig.tfrec")
        if os.path.exists(p("output_orig.tfrec"))
        else p("output.tfrec")
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        predict = os.path.join(tmp_dir, "predict.tfrec")
        record_model(
            input, new_part, predict, num_procs=num_procs, num_threads=num_threads
        )
        mae, nrmse = diff_stats(output, predict)

    return mae, nrmse


def join_in_dir(dir, new_part, output_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        d = lambda file: os.path.join(dir, file)
        new_end = os.path.join(tmp_dir, "new_end.tflite")
        join_models(new_part, d("end.tflite"), new_end)
        join_models(d("start.tflite"), new_end, output_model)


def train_in_dir(
    train_dir,
    baseline_dir,
    output_filename,
    replace_fn,
    augmentations,
    steps,
    lr=1e-3,
    batch_size=32,
    checkpoint_at=None,
    checkpoint_decay_steps=0,
    schedule="cosine",
    verbose=False,
    show_progress=False,
    num_procs=None,
    num_threads=1,
):
    """Train a replacement for replace.tflite using the input.tfrec and output.tfrec in train_dir.
    If baseline_dir is provided, train the replacement to match baseline outputs for train_dir inputs.
    Result saved as new.tflite in train_dir.
    """
    teacher_dir = baseline_dir if baseline_dir else train_dir
    teacher = ParallelTFLiteModel(
        "%s/replace.tflite" % teacher_dir, num_procs, num_threads, batch_size=batch_size
    )
    replace = TFLiteModel("%s/replace.tflite" % train_dir)
    assert len(teacher.input_tensors()) == 1, (
        "Can only train replacements with a single input tensor right now, found %s"
        % teacher.input_tensors()
    )
    assert len(teacher.output_tensors()) == 1, (
        "Can only train replacements with a single output tensor right now, found %s"
        % teacher.output_tensors()
    )
    input_name = teacher.input_tensors()[0]
    output_name = teacher.output_tensors()[0]

    assert len(teacher.shape_from_name) == len(
        replace.shape_from_name
    ), "Baseline and train models must have the same number of inputs and outputs. Teacher: {}\nTrain dir: {}".format(
        teacher.shape_from_name, replace.shape_from_name
    )
    assert all(
        tn == rn and (ts[1:] == rs[1:]).all()
        for (tn, ts), (rn, rs) in zip(
            teacher.shape_from_name.items(), replace.shape_from_name.items()
        )
    ), "Baseline and train models must have the same input and output shapes for the subgraph being replaced. Teacher: {}\nTrain dir: {}".format(
        teacher.shape_from_name, replace.shape_from_name
    )

    input_filename = os.path.join(train_dir, "input.tfrec")
    total = numpytf_count(input_filename)
    dict_inputs = NumpyTFReader(input_filename)
    inputs = dict_inputs.map(lambda d: tf.squeeze(d[input_name], axis=0))
    if any(augmentations):
        # Map the teacher inputs here because the augmentation stage passes these through a TFLite model to get the outputs
        teacher_outputs = NumpyTFReader(os.path.join(teacher_dir, "input.tfrec")).map(
            lambda d: tf.squeeze(d[input_name], axis=0)
        )
    else:
        teacher_outputs = NumpyTFReader(os.path.join(teacher_dir, "output.tfrec")).map(
            lambda d: tf.squeeze(d[output_name], axis=0)
        )

    steps_per_epoch = math.ceil(total / batch_size)
    epochs = int(math.ceil(steps / steps_per_epoch))
    if verbose:
        print(
            "Training on %d items for %d steps (%d epochs with batch size %d)"
            % (total, epochs * steps_per_epoch, epochs, batch_size)
        )

    dataset = tf.data.Dataset.zip((inputs, teacher_outputs))
    if epochs > 1:
        dataset = dataset.cache()
    dataset = dataset.shuffle(total).repeat().batch(batch_size)

    if any(augmentations):
        augment_train, augment_teacher = augment_fn_twins(dict_inputs, augmentations)
        augment_fn = lambda train, teach: (
            augment_train({input_name: train})[input_name],
            teacher(augment_teacher({input_name: teach}))[output_name],
        )
        dataset = dataset.map(
            lambda train, teach: tf.py_function(
                augment_fn, inp=[train, teach], Tout=[tf.float32, tf.float32]
            )
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    input_shape = teacher.shape_from_name[input_name][1:]
    output_shape = teacher.shape_from_name[output_name][1:]
    model = replace_fn(input_shape, output_shape)

    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)

    if verbose:
        model.summary()

    steps_so_far = 0

    def cosine_decay(epoch_step, logs):
        """Cosine decay from lr at start of the run to zero at the end"""
        current_step = epoch_step + steps_so_far
        learning_rate = lr * (math.cos(math.pi * current_step / steps) + 1) / 2.0
        tf.keras.backend.set_value(optimizer.learning_rate, learning_rate)

    def late_decay(epoch_step, logs):
        """Constant until the last 20% of the run, then linear decay to zero"""
        current_step = epoch_step + steps_so_far
        steps_remaining = steps - current_step
        decay_length = steps // 5
        decay_fraction = min(steps_remaining, decay_length) / decay_length
        learning_rate = lr * decay_fraction
        tf.keras.backend.set_value(optimizer.learning_rate, learning_rate)

    if schedule == "cosine":
        callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_begin=cosine_decay)]
    elif schedule == "late":
        callbacks = [tf.keras.callbacks.LambdaCallback(on_batch_begin=late_decay)]
    elif schedule == "constant":
        callbacks = []
    else:
        assert schedule not in learning_rate_schedules
        raise ValueError(
            f'LR schedule "{schedule}" not implemented - expected one of {learning_rate_schedules}.'
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
        print(
            "lr decayed from %f to %f over %d steps"
            % (lr_start, optimizer.learning_rate.numpy(), steps_to_train)
        )

        if steps_so_far < steps:
            filename, ext = os.path.splitext(output_filename)
            checkpoint_filename = filename + ("_@%d" % steps_so_far) + ext
        else:
            checkpoint_filename = output_filename
        print("%d/%d: Saved as %s" % (steps_so_far, steps, checkpoint_filename))
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
    keras_model, filename, input_name, input_shape, output_name, output_shape
):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    with open(filename, "wb") as f:
        f.write(tflite_model)

    # Now fix the shapes and names to match those we expect
    fb = load(filename)
    i = fb.subgraphs[0].inputs[0]
    fb.subgraphs[0].tensors[i].shape = np.array(input_shape, dtype=np.int32)
    fb.subgraphs[0].tensors[i].name = input_name.encode("utf-8")
    o = fb.subgraphs[0].outputs[0]
    fb.subgraphs[0].tensors[o].shape = np.array(output_shape, dtype=np.int32)
    fb.subgraphs[0].tensors[o].name = output_name.encode("utf-8")
    save(fb, filename)


def augment_fn_twins(inputs, augmentations):
    """Return a pair of twinned augmentation functions with the same sequence of random numbers"""
    seed = np.random.randint(2**32 - 1)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    return augment_fn(inputs, augmentations, rng1), augment_fn(
        inputs, augmentations, rng2
    )


def augment_fn(inputs, augmentations, rng):
    mixup_strength, gaussian_strength = augmentations

    augments = []

    if mixup_strength:
        mixup_range = (0.5 - mixup_strength / 2, 0.5 + mixup_strength / 2)
        augment = lambda d: {
            k: mixup(rng, v.numpy(), mixup_range) for k, v in d.items()
        }
        augments.append(augment)

    if gaussian_strength:
        values = defaultdict(list)
        for d in inputs.as_numpy_iterator():
            for k, v in d.items():
                values[k].append(v)
        noise_scale = {
            k: np.std(v, axis=0).astype(np.float32) for k, v in values.items()
        }
        augment = lambda d: {
            k: v
            + rng.standard_normal(v.shape).astype(np.float32)
            * gaussian_strength
            * noise_scale[k]
            for k, v in d.items()
        }
        augments.append(augment)

    if len(augments) == 0:
        return lambda x: x
    elif len(augments) == 1:
        return augments[0]
    elif len(augments) == 2:
        return lambda x: augments[1](augments[0](x))
    else:
        assert False, "Unexpected number of augmentation functions (%d)" % len(augments)


def mixup(rng, batch, beta_range=(0.0, 1.0)):
    """Each tensor in the batch becomes a linear combination of it and one other tensor"""
    a = batch
    b = np.array(batch)
    rng.shuffle(b)  # randomly pair up tensors in the batch
    # random mixing coefficient for each pair
    beta = rng.uniform(
        low=beta_range[0], high=beta_range[1], size=batch.shape[0]
    ).astype(np.float32)
    return (a.T * beta).T + (b.T * (1.0 - beta)).T  # return linear combinations
