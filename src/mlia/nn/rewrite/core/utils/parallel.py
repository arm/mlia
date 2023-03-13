# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from psutil import cpu_count

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mlia.nn.rewrite.core.utils.numpy_tfrecord import TFLiteModel


class ParallelTFLiteModel(TFLiteModel):
    def __init__(self, filename, num_procs=1, num_threads=0, batch_size=None):
        """num_procs: 0 => detect real cores on system
        num_threads: 0 => TFLite impl. specific setting, usually 3
        batch_size: None => automatic (num_procs or file-determined)
        """
        self.pool = None
        self.filename = filename
        if not num_procs:
            self.num_procs = cpu_count(logical=False)
        else:
            self.num_procs = int(num_procs)

        self.num_threads = num_threads

        if self.num_procs > 1:
            if not batch_size:
                batch_size = self.num_procs  # default to min effective batch size
            local_batch_size = int(math.ceil(batch_size / self.num_procs))
            super().__init__(filename, batch_size=local_batch_size)
            del self.interpreter
            self.pool = Pool(
                processes=self.num_procs,
                initializer=_pool_create_worker,
                initargs=[filename, self.batch_size, self.num_threads],
            )
        else:  # fall back to serial implementation for max performance
            super().__init__(
                filename, batch_size=batch_size, num_threads=self.num_threads
            )

        self.total_batches = 0
        self.partial_batches = 0
        self.warned = False

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.terminate()

    def __del__(self):
        self.close()

    def __call__(self, named_input):
        if self.pool:
            global_batch_size = next(iter(named_input.values())).shape[0]
            # Note: self.batch_size comes from superclass and is local batch size
            chunks = int(math.ceil(global_batch_size / self.batch_size))
            self.total_batches += 1
            if chunks != self.num_procs:
                self.partial_batches += 1
            if (
                not self.warned
                and self.total_batches > 10
                and self.partial_batches / self.total_batches >= 0.5
            ):
                print(
                    "ParallelTFLiteModel(%s): warning - %.1f%% of batches do not use all %d processes, set batch size to a multiple of this"
                    % (
                        self.filename,
                        100 * self.partial_batches / self.total_batches,
                        self.num_procs,
                    )
                )
                self.warned = True

            local_batches = [
                {
                    key: values[i * self.batch_size : (i + 1) * self.batch_size]
                    for key, values in named_input.items()
                }
                for i in range(chunks)
            ]
            chunk_results = self.pool.map(_pool_run, local_batches)
            named_ys = defaultdict(list)
            for chunk in chunk_results:
                for k, v in chunk.items():
                    named_ys[k].append(v)
            return {k: np.concatenate(v) for k, v in named_ys.items()}
        else:
            return super().__call__(named_input)


_local_model = None


def _pool_create_worker(filename, local_batch_size=None, num_threads=None):
    global _local_model
    _local_model = TFLiteModel(
        filename, batch_size=local_batch_size, num_threads=num_threads
    )


def _pool_run(named_inputs):
    return _local_model(named_inputs)
