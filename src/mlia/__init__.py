# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Init of MLIA."""
import logging
import os

import pkg_resources

# redirect warnings to logging
logging.captureWarnings(True)


# as TensorFlow tries to configure root logger
# it should be configured before importing TensorFlow
root_logger = logging.getLogger()
root_logger.addHandler(logging.NullHandler())


# disable TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = pkg_resources.get_distribution("mlia").version
