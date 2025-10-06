# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pre-commit hook that checks the current year is in the Copyright header of a file.

If the header is out of date it will print a warning.
"""
import datetime
import os
import subprocess  # nosec


class CopyrightHeaderChecker:
    """Class that wraps the checker for the Copyright header."""

    def check_files_have_updated_header(self, filenames: list) -> None:
        """Check whether input files have the current year in the copyright string."""
        current_year = str(datetime.datetime.now().year)
        for filename in filenames:
            # Skip deleted or missing files (e.g. after git rm)
            if not os.path.exists(filename):
                continue

            with open(filename, encoding="utf-8") as file:
                first_line = file.readline()
                second_line = file.readline()

            # Handle Markdown vs others
            header_line = second_line if filename.endswith(".md") else first_line
            if current_year not in header_line:
                print(f"WARNING: The Copyright header of {filename} is out of date!")


if __name__ == "__main__":
    staged_files = (
        subprocess.check_output(["git", "diff", "--cached", "--name-only"])  # nosec
        .decode()
        .splitlines()
    )
    CopyrightHeaderChecker().check_files_have_updated_header(filenames=staged_files)
