# Copyright 2021, Arm Ltd.
"""Init of mlia."""
import os

import pkg_resources

# disable tensorflow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = pkg_resources.get_distribution("mlia").version
