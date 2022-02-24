# Copyright (C) 2021-2022, Arm Ltd.
"""Init of mlia."""
import logging
import os

import pkg_resources

# redirect warnings to logging
logging.captureWarnings(True)


# as tensorflow tries to configure root logger
# it should be configured before importing tensorflow
root_logger = logging.getLogger()
root_logger.addHandler(logging.NullHandler())


# disable tensorflow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = pkg_resources.get_distribution("mlia").version
