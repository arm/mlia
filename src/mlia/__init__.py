# SPDX-FileCopyrightText: Copyright 2022, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Init of MLIA."""

import logging
import os
import pkgutil
from importlib.metadata import version

# Allow mlia subpackages to be provided by multiple distributions.
__path__ = pkgutil.extend_path(__path__, __name__)

# redirect warnings to logging
logging.captureWarnings(True)


# as TensorFlow tries to configure root logger
# it should be configured before importing TensorFlow
root_logger = logging.getLogger()
root_logger.addHandler(logging.NullHandler())


# disable TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = version("mlia")
