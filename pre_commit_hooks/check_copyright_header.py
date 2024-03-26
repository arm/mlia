# SPDX-FileCopyrightText: Copyright 2024, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pre-commit hook that checks the current year is in the Copyright header of a file.

If the header is out of date it will print a warning.
"""
import datetime
import subprocess  # nosec


class CopyrightHeaderChecker:
    """Class that wraps the checker for the Copyright header."""

    def check_files_have_updated_header(self, filenames: list) -> None:
        """Check whether input files have the current year in the copyright string."""
        current_year = str(datetime.datetime.now().year)
        for filename in filenames:
            with open(filename, encoding="utf-8") as file:
                first_line = file.readline()
                second_line = file.readline()
            if filename.endswith(".md") and current_year not in second_line:
                print(f"WARNING: The Copyright header of {filename} is out of date!")

            if not filename.endswith(".md") and current_year not in first_line:
                print(f"WARNING: The Copyright header of {filename} is out of date!")


if __name__ == "__main__":
    staged_files = (
        subprocess.check_output(["git", "diff", "--cached", "--name-only"])  # nosec
        .decode()
        .splitlines()
    )
    CopyrightHeaderChecker().check_files_have_updated_header(filenames=staged_files)
