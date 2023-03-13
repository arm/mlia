# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import random
import tempfile
from collections import defaultdict

import numpy as np

from mlia.nn.rewrite.core.utils.utils import load
from mlia.nn.rewrite.core.utils.utils import save

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.lite.python import interpreter as interpreter_wrapper


def make_decode_fn(filename):
    def decode_fn(record_bytes, type_map):
        parse_dict = {
            name: tf.io.FixedLenFeature([], tf.string) for name in type_map.keys()
        }
        example = tf.io.parse_single_example(record_bytes, parse_dict)
        features = {
            n: tf.io.parse_tensor(example[n], tf.as_dtype(t))
            for n, t in type_map.items()
        }
        return features

    meta_filename = filename + ".meta"
    with open(meta_filename) as f:
        type_map = json.load(f)["type_map"]
    return lambda record_bytes: decode_fn(record_bytes, type_map)


def NumpyTFReader(filename):
    decode_fn = make_decode_fn(filename)
    dataset = tf.data.TFRecordDataset(filename)
    return dataset.map(decode_fn)


def numpytf_count(filename):
    meta_filename = filename + ".meta"
    with open(meta_filename) as f:
        return json.load(f)["count"]


class NumpyTFWriter:
    def __init__(self, filename):
        self.filename = filename
        self.meta_filename = filename + ".meta"
        self.writer = tf.io.TFRecordWriter(filename)
        self.type_map = {}
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def write(self, array_dict):
        type_map = {n: str(a.dtype.name) for n, a in array_dict.items()}
        self.type_map.update(type_map)
        self.count += 1

        feature = {
            n: tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(a).numpy()])
            )
            for n, a in array_dict.items()
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

    def close(self):
        with open(self.meta_filename, "w") as f:
            meta = {"type_map": self.type_map, "count": self.count}
            json.dump(meta, f)
        self.writer.close()


class TFLiteModel:
    def __init__(self, filename, batch_size=None, num_threads=None):
        if num_threads == 0:
            num_threads = None
        if batch_size == None:
            self.interpreter = interpreter_wrapper.Interpreter(
                model_path=filename, num_threads=num_threads
            )
        else:  # if a batch size is specified, modify the TFLite model to use this size
            with tempfile.TemporaryDirectory() as tmp:
                fb = load(filename)
                for sg in fb.subgraphs:
                    for t in list(sg.inputs) + list(sg.outputs):
                        sg.tensors[t].shape = np.array(
                            [batch_size] + list(sg.tensors[t].shape[1:]), dtype=np.int32
                        )
                tempname = os.path.join(tmp, "rewrite_tmp.tflite")
                save(fb, tempname)
                self.interpreter = interpreter_wrapper.Interpreter(
                    model_path=tempname, num_threads=num_threads
                )

        try:
            self.interpreter.allocate_tensors()
        except RuntimeError:
            self.interpreter = interpreter_wrapper.Interpreter(
                model_path=filename, num_threads=num_threads
            )
            self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        details = list(self.input_details) + list(self.output_details)
        self.handle_from_name = {d["name"]: d["index"] for d in details}
        self.shape_from_name = {d["name"]: d["shape"] for d in details}
        self.batch_size = next(iter(self.shape_from_name.values()))[0]

    def __call__(self, named_input):
        """Execute the model on one or a batch of named inputs (a dict of name: numpy array)"""
        input_len = next(iter(named_input.values())).shape[0]
        full_steps = input_len // self.batch_size
        remainder = input_len % self.batch_size

        named_ys = defaultdict(list)
        for i in range(full_steps):
            for name, x_batch in named_input.items():
                x = x_batch[i : i + self.batch_size]
                self.interpreter.set_tensor(self.handle_from_name[name], x)
            self.interpreter.invoke()
            for d in self.output_details:
                named_ys[d["name"]].append(self.interpreter.get_tensor(d["index"]))
        if remainder:
            for name, x_batch in named_input.items():
                x = np.zeros(self.shape_from_name[name]).astype(x_batch.dtype)
                x[:remainder] = x_batch[-remainder:]
                self.interpreter.set_tensor(self.handle_from_name[name], x)
            self.interpreter.invoke()
            for d in self.output_details:
                named_ys[d["name"]].append(
                    self.interpreter.get_tensor(d["index"])[:remainder]
                )
        return {k: np.concatenate(v) for k, v in named_ys.items()}

    def input_tensors(self):
        return [d["name"] for d in self.input_details]

    def output_tensors(self):
        return [d["name"] for d in self.output_details]


def sample_tfrec(input_file, k, output_file):
    total = numpytf_count(input_file)
    next = sorted(random.sample(range(total), k=k), reverse=True)

    reader = NumpyTFReader(input_file)
    with NumpyTFWriter(output_file) as writer:
        for i, data in enumerate(reader):
            if i == next[-1]:
                next.pop()
                writer.write(data)
                if not next:
                    break
