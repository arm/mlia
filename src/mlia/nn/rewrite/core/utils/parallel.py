# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Parallelize a TFLiteModel."""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from mlia.nn.rewrite.core.utils.numpy_tfrecord import TFLiteModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logger = logging.getLogger(__name__)


class ParallelTFLiteModel(TFLiteModel):
    """A parallel version of a TFLiteModel.

    num_procs: 0 => detect real cores on system
    num_threads: 0 => TFLite impl. specific setting, usually 3
    batch_size: None => automatic (num_procs or file-determined)
    """

    def __init__(
        self,
        filename: str | Path,
        num_procs: int = 1,
        num_threads: int = 0,
        batch_size: int | None = None,
    ) -> None:
        """Initiate a Parallel TFLite Model."""
        self.pool = None
        filename = str(filename)
        self.filename = filename
        if not num_procs:
            self.num_procs = cpu_count()
        else:
            self.num_procs = int(num_procs)

        self.num_threads = num_threads

        if self.num_procs > 1:
            if not batch_size:
                batch_size = self.num_procs  # default to min effective batch size
            local_batch_size = int(math.ceil(batch_size / self.num_procs))
            super().__init__(filename, batch_size=local_batch_size)
            del self.interpreter
            self.pool = Pool(  # pylint: disable=consider-using-with
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

    def close(self) -> None:
        """Close and terminate pool."""
        if self.pool:
            self.pool.close()
            self.pool.terminate()

    def __del__(self) -> None:
        """Close instance."""
        self.close()

    def __call__(self, named_input: dict) -> Any:
        """Call instance."""
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
                logger.warning(
                    "ParallelTFLiteModel(%s): warning - %.1f of batches "
                    "do not use all %d processes, set batch size to "
                    "a multiple of this.",
                    self.filename,
                    100 * self.partial_batches / self.total_batches,
                    self.num_procs,
                )
                self.warned = True

            local_batches = [
                {
                    key: values[
                        i * self.batch_size : (i + 1) * self.batch_size  # noqa: E203
                    ]
                    for key, values in named_input.items()
                }
                for i in range(chunks)
            ]
            chunk_results = self.pool.map(_pool_run, local_batches)
            named_ys = defaultdict(list)
            for chunk in chunk_results:
                for key, value in chunk.items():
                    named_ys[key].append(value)
            return {key: np.concatenate(value) for key, value in named_ys.items()}

        return super().__call__(named_input)


_LOCAL_MODEL = None


def _pool_create_worker(
    filename: str, local_batch_size: int = 0, num_threads: int = 0
) -> None:
    global _LOCAL_MODEL  # pylint: disable=global-statement
    _LOCAL_MODEL = TFLiteModel(
        filename, batch_size=local_batch_size, num_threads=num_threads
    )


def _pool_run(named_inputs: dict) -> Any:
    if _LOCAL_MODEL:
        return _LOCAL_MODEL(named_inputs)
    raise ValueError("TFLiteModel is not initiated")
