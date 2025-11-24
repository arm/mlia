# SPDX-FileCopyrightText: Copyright 2024-2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pre-commit hook that checks the current year is in the Copyright header of a file.

Checks both staged files and files modified in the last commit to catch cases
where files might have been committed with --no-verify and outdated headers.
If the header is out of date it will print a warning.
"""
import datetime
import os
import subprocess  # nosec
import sys


class CopyrightHeaderChecker:
    """Class that wraps the checker for the Copyright header."""

    def check_files_have_updated_header(self, filenames: list) -> bool:
        """Check whether input files have the current year in the copyright string.

        Args:
            filenames: List of file paths to check for updated copyright headers.

        Returns:
            True if all files have updated headers, False otherwise.
        """
        current_year = str(datetime.datetime.now().year)
        has_outdated_headers = False

        for filename in filenames:
            # Skip deleted or missing files (e.g. after git rm)
            if not os.path.exists(filename):
                continue

            # For JSON files, check for sidecar .license file
            if filename.endswith(".json"):
                license_file = filename + ".license"
                if os.path.exists(license_file):
                    filename = license_file
                else:
                    print(
                        f"ERROR: JSON file {filename} requires a sidecar "
                        f"{license_file} file with copyright header!"
                    )
                    has_outdated_headers = True
                    continue

            try:
                with open(filename, encoding="utf-8") as file:
                    first_line = file.readline()
                    second_line = file.readline()
            except (UnicodeDecodeError, PermissionError) as err:
                # Skip binary files and files without read permissions
                print(f"WARN: Cannot check {filename}: {err}! ")
                continue

            # Handle Markdown vs others
            header_line = second_line if filename.endswith(".md") else first_line
            if current_year not in header_line:
                print(f"ERROR: The Copyright header of {filename} is out of date!")
                has_outdated_headers = True

        return not has_outdated_headers


if __name__ == "__main__":
    # Check staged files
    staged_files = (
        subprocess.check_output(["git", "diff", "--cached", "--name-only"])  # nosec
        .decode()
        .splitlines()
    )

    # Also check files modified in the last commit to catch files that might have
    # been committed with --no-verify and outdated copyright headers
    try:
        recently_modified_files = (
            subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
            )  # nosec
            .decode()
            .splitlines()
        )
    except subprocess.CalledProcessError:
        # Handle case where there's no previous commit (initial commit)
        recently_modified_files = []

    # Combine and deduplicate files to check
    all_files_to_check = list(set(staged_files + recently_modified_files))

    checker = CopyrightHeaderChecker()
    # pylint: disable-next=invalid-name
    headers_are_valid = checker.check_files_have_updated_header(
        filenames=all_files_to_check
    )

    if not headers_are_valid:
        sys.exit(1)  # Exit with error code to fail the pre-commit hook
