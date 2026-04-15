# SPDX-FileCopyrightText: Copyright 2024-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pre-commit hook that checks the current year is in the Copyright header of a file.

Checks both staged files and files modified in the last commit to catch cases
where files might have been committed with --no-verify and outdated headers.
If the header is out of date it will print a warning.
"""

import datetime
import os
import re
import subprocess  # nosec
import sys
from pathlib import Path

BUILD_FILES = [
    ".python-version",
]


class UnreadableFileError(Exception):
    """Raised when a file cannot be read for header validation."""

    def __init__(self, filename: str, error: Exception) -> None:
        """Initialise the unreadable file error."""
        super().__init__(f"WARN: Cannot check {filename}: {error}! ")


def requires_sidecar(filename: str) -> bool:
    """Return whether the file type requires a sidecar license file."""
    return filename.endswith(".json") or Path(filename).name in BUILD_FILES


def resolve_header_source(filename: str) -> tuple[str, str] | tuple[None, str]:
    """Return the file to inspect for a copyright header."""
    if filename.endswith(".license"):
        return filename, ""

    license_file = f"{filename}.license"
    if os.path.exists(license_file):
        return license_file, ""

    if requires_sidecar(filename):
        return (
            None,
            f"ERROR: {filename} requires a sidecar {license_file} file with "
            "copyright header!",
        )

    return filename, ""


def normalise_copyright_line(line: str) -> str:
    """Return a canonical SPDX copyright line for comparison."""
    payload = line.partition("SPDX-FileCopyrightText:")[2].strip()
    payload = re.sub(r"\s*(?:-->|\*/)\s*$", "", payload)
    return f"SPDX-FileCopyrightText: {payload}"


def extract_copyright_lines(filename: str) -> list[str]:
    """Return SPDX copyright lines from a file."""
    try:
        with open(filename, encoding="utf-8") as file:
            return [
                normalise_copyright_line(line)
                for line in file
                if "SPDX-FileCopyrightText:" in line
            ]
    except (PermissionError, UnicodeDecodeError) as error:
        raise UnreadableFileError(filename, error) from error


def read_header_lines(filename: str) -> tuple[str, str]:
    """Return the first two lines of a header source file."""
    try:
        with open(filename, encoding="utf-8") as file:
            return file.readline(), file.readline()
    except (PermissionError, UnicodeDecodeError) as error:
        raise UnreadableFileError(filename, error) from error


class CopyrightHeaderChecker:
    """Class that wraps the checker for the Copyright header."""

    def _check_for_conflicting_headers(
        self, filename: str, header_filename: str
    ) -> tuple[bool, str | None]:
        """Check whether file and sidecar contain conflicting copyright headers."""
        if header_filename == filename or requires_sidecar(filename):
            return True, None

        file_copyright_lines = extract_copyright_lines(filename)
        if not file_copyright_lines:
            return True, None

        sidecar_copyright_lines = extract_copyright_lines(header_filename)
        if set(file_copyright_lines) == set(sidecar_copyright_lines):
            return True, None

        return (
            False,
            f"ERROR: {filename} contains copyright headers that conflict with "
            f"{header_filename}!",
        )

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

            header_filename, error = resolve_header_source(filename)
            if header_filename is None:
                print(error)
                has_outdated_headers = True
                continue

            try:
                headers_are_consistent, consistency_error = (
                    self._check_for_conflicting_headers(filename, header_filename)
                )
                first_line, second_line = read_header_lines(header_filename)
            except UnreadableFileError as error:
                print(error)
                continue

            if not headers_are_consistent:
                print(consistency_error)
                has_outdated_headers = True
                continue

            # Handle Markdown vs others
            header_line = second_line if header_filename.endswith(".md") else first_line
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
            subprocess.check_output(["git", "diff", "--name-only", "HEAD~1", "HEAD"])  # nosec
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
