# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0
"""@todo: detailed usage information here."""
import argparse
import logging
import sys
from typing import List
from typing import Optional

from . import __version__


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the application."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="@todo: brief summary here",
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase verbosity"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="print the version and exit",
    )
    parser.add_argument("FILEs", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose > 1:
        logging.getLogger().setLevel(logging.DEBUG)

    for path in args.FILEs:
        sys.stdout.write("arg: %s\n" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
