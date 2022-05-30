# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for Corstone based FVPs."""
import logging
import platform
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional

import mlia.tools.aiet_wrapper as aiet
from mlia.tools.metadata.common import DownloadAndInstall
from mlia.tools.metadata.common import Installation
from mlia.tools.metadata.common import InstallationType
from mlia.tools.metadata.common import InstallFromPath
from mlia.utils.download import DownloadArtifact
from mlia.utils.filesystem import all_files_exist
from mlia.utils.filesystem import all_paths_valid
from mlia.utils.filesystem import copy_all
from mlia.utils.filesystem import get_mlia_resources
from mlia.utils.filesystem import temp_directory
from mlia.utils.proc import working_directory

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Backend information."""

    backend_path: Path
    copy_source: bool = True
    system_config: Optional[str] = None


PathChecker = Callable[[Path], Optional[BackendInfo]]
BackendInstaller = Callable[[bool, Path], Path]


class AIETMetadata:
    """AIET installation metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        system_config: str,
        apps_resources: List[str],
        fvp_dir_name: str,
        download_artifact: Optional[DownloadArtifact],
        supported_platforms: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize AIETMetaData.

        Members expected_systems and expected_apps are filled automatically.
        """
        self.name = name
        self.description = description
        self.system_config = system_config
        self.apps_resources = apps_resources
        self.fvp_dir_name = fvp_dir_name
        self.download_artifact = download_artifact
        self.supported_platforms = supported_platforms

        self.expected_systems = aiet.get_all_system_names(name)
        self.expected_apps = aiet.get_all_application_names(name)

    @property
    def expected_resources(self) -> Iterable[Path]:
        """Return list of expected resources."""
        resources = [self.system_config, *self.apps_resources]

        return (get_mlia_resources() / resource for resource in resources)

    @property
    def supported_platform(self) -> bool:
        """Return true if current platform supported."""
        if not self.supported_platforms:
            return True

        return platform.system() in self.supported_platforms


class AIETBasedInstallation(Installation):
    """Backend installation based on AIET functionality."""

    def __init__(
        self,
        aiet_runner: aiet.AIETRunner,
        metadata: AIETMetadata,
        path_checker: PathChecker,
        backend_installer: Optional[BackendInstaller],
    ) -> None:
        """Init the tool installation."""
        self.aiet_runner = aiet_runner
        self.metadata = metadata
        self.path_checker = path_checker
        self.backend_installer = backend_installer

    @property
    def name(self) -> str:
        """Return name of the tool."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Return description of the tool."""
        return self.metadata.description

    @property
    def already_installed(self) -> bool:
        """Return true if tool already installed."""
        return self.aiet_runner.all_installed(
            self.metadata.expected_systems, self.metadata.expected_apps
        )

    @property
    def could_be_installed(self) -> bool:
        """Return true if tool could be installed."""
        if not self.metadata.supported_platform:
            return False

        return all_paths_valid(self.metadata.expected_resources)

    def supports(self, install_type: InstallationType) -> bool:
        """Return true if tools supported type of the installation."""
        if isinstance(install_type, DownloadAndInstall):
            return self.metadata.download_artifact is not None

        if isinstance(install_type, InstallFromPath):
            return self.path_checker(install_type.backend_path) is not None

        return False  # type: ignore

    def install(self, install_type: InstallationType) -> None:
        """Install the tool."""
        if isinstance(install_type, DownloadAndInstall):
            download_artifact = self.metadata.download_artifact
            assert download_artifact is not None, "No artifact provided"

            self.download_and_install(download_artifact, install_type.eula_agreement)
        elif isinstance(install_type, InstallFromPath):
            backend_path = self.path_checker(install_type.backend_path)
            assert backend_path is not None, "Unable to resolve backend path"

            self.install_from(backend_path)
        else:
            raise Exception(f"Unable to install {install_type}")

    def install_from(self, backend_info: BackendInfo) -> None:
        """Install tool from the directory."""
        mlia_resources = get_mlia_resources()

        with temp_directory() as tmpdir:
            fvp_dist_dir = tmpdir / self.metadata.fvp_dir_name

            system_config = self.metadata.system_config
            if backend_info.system_config:
                system_config = backend_info.system_config

            resources_to_copy = [mlia_resources / system_config]
            if backend_info.copy_source:
                resources_to_copy.append(backend_info.backend_path)

            copy_all(*resources_to_copy, dest=fvp_dist_dir)

            self.aiet_runner.install_system(fvp_dist_dir)

        for app in self.metadata.apps_resources:
            self.aiet_runner.install_application(mlia_resources / app)

    def download_and_install(
        self, download_artifact: DownloadArtifact, eula_agrement: bool
    ) -> None:
        """Download and install the tool."""
        with temp_directory() as tmpdir:
            try:
                downloaded_to = download_artifact.download_to(tmpdir)
            except Exception as err:
                raise Exception("Unable to download backend artifact") from err

            with working_directory(tmpdir / "dist", create_dir=True) as dist_dir:
                with tarfile.open(downloaded_to) as archive:
                    archive.extractall(dist_dir)

                assert self.backend_installer, (
                    f"Backend '{self.metadata.name}' does not support "
                    "download and installation."
                )
                backend_path = self.backend_installer(eula_agrement, dist_dir)
                if self.path_checker(backend_path) is None:
                    raise Exception("Downloaded artifact has invalid structure")

                self.install(InstallFromPath(backend_path))


class PackagePathChecker:
    """Package path checker."""

    def __init__(
        self, expected_files: List[str], backend_subfolder: Optional[str] = None
    ) -> None:
        """Init the path checker."""
        self.expected_files = expected_files
        self.backend_subfolder = backend_subfolder

    def __call__(self, backend_path: Path) -> Optional[BackendInfo]:
        """Check if directory contains all expected files."""
        resolved_paths = (backend_path / file for file in self.expected_files)
        if not all_files_exist(resolved_paths):
            return None

        if self.backend_subfolder:
            subfolder = backend_path / self.backend_subfolder

            if not subfolder.is_dir():
                return None

            return BackendInfo(subfolder)

        return BackendInfo(backend_path)


class StaticPathChecker:
    """Static path checker."""

    def __init__(
        self,
        static_backend_path: Path,
        expected_files: List[str],
        copy_source: bool = False,
        system_config: Optional[str] = None,
    ) -> None:
        """Init static path checker."""
        self.static_backend_path = static_backend_path
        self.expected_files = expected_files
        self.copy_source = copy_source
        self.system_config = system_config

    def __call__(self, backend_path: Path) -> Optional[BackendInfo]:
        """Check if directory equals static backend path with all expected files."""
        if backend_path != self.static_backend_path:
            return None

        resolved_paths = (backend_path / file for file in self.expected_files)
        if not all_files_exist(resolved_paths):
            return None

        return BackendInfo(
            backend_path,
            copy_source=self.copy_source,
            system_config=self.system_config,
        )


class CompoundPathChecker:
    """Compound path checker."""

    def __init__(self, *path_checkers: PathChecker) -> None:
        """Init compound path checker."""
        self.path_checkers = path_checkers

    def __call__(self, backend_path: Path) -> Optional[BackendInfo]:
        """Iterate over checkers and return first non empty backend info."""
        first_resolved_backend_info = (
            backend_info
            for path_checker in self.path_checkers
            if (backend_info := path_checker(backend_path)) is not None
        )

        return next(first_resolved_backend_info, None)


class Corstone300Installer:
    """Helper class that wraps Corstone 300 installation logic."""

    def __call__(self, eula_agreement: bool, dist_dir: Path) -> Path:
        """Install Corstone-300 and return path to the models."""
        with working_directory(dist_dir):
            install_dir = "corstone-300"
            try:
                fvp_install_cmd = [
                    "./FVP_Corstone_SSE-300.sh",
                    "-q",
                    "-d",
                    install_dir,
                ]
                if not eula_agreement:
                    fvp_install_cmd += [
                        "--nointeractive",
                        "--i-agree-to-the-contained-eula",
                    ]

                subprocess.check_call(fvp_install_cmd)
            except subprocess.CalledProcessError as err:
                raise Exception(
                    "Error occurred during Corstone-300 installation"
                ) from err

            return dist_dir / install_dir


def get_corstone_300_installation() -> Installation:
    """Get Corstone-300 installation."""
    corstone_300 = AIETBasedInstallation(
        aiet_runner=aiet.get_aiet_runner(),
        # pylint: disable=line-too-long
        metadata=AIETMetadata(
            name="Corstone-300",
            description="Corstone-300 FVP",
            system_config="aiet/systems/corstone-300/aiet-config.json",
            apps_resources=[
                "aiet/applications/inference_runner-sse-300-22.05.01-ethos-U55-Shared_Sram-TA",
                "aiet/applications/inference_runner-sse-300-22.05.01-ethos-U55-Sram_Only-TA",
                "aiet/applications/inference_runner-sse-300-22.05.01-ethos-U65-Dedicated_Sram-TA",
            ],
            fvp_dir_name="corstone_300",
            download_artifact=DownloadArtifact(
                name="Corstone-300 FVP",
                url="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.16_26.tgz",
                filename="FVP_Corstone_SSE-300_11.16_26.tgz",
                version="11.16_26",
                sha256_hash="e26139be756b5003a30d978c629de638aed1934d597dc24a17043d4708e934d7",
            ),
            supported_platforms=["Linux"],
        ),
        # pylint: enable=line-too-long
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=[
                    "models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55",
                    "models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U65",
                ],
                backend_subfolder="models/Linux64_GCC-6.4",
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=[
                    "VHT_Corstone_SSE-300_Ethos-U55",
                    "VHT_Corstone_SSE-300_Ethos-U65",
                ],
                copy_source=False,
                system_config="aiet/systems/corstone-300-vht/aiet-config.json",
            ),
        ),
        backend_installer=Corstone300Installer(),
    )

    return corstone_300


def get_corstone_310_installation() -> Installation:
    """Get Corstone-310 installation."""
    corstone_310 = AIETBasedInstallation(
        aiet_runner=aiet.get_aiet_runner(),
        # pylint: disable=line-too-long
        metadata=AIETMetadata(
            name="Corstone-310",
            description="Corstone-310 FVP",
            system_config="aiet/systems/corstone-310/aiet-config.json",
            apps_resources=[
                "aiet/applications/inference_runner-sse-310-22.05.01-ethos-U55-Shared_Sram-TA",
                "aiet/applications/inference_runner-sse-310-22.05.01-ethos-U55-Sram_Only-TA",
            ],
            fvp_dir_name="corstone_310",
            download_artifact=None,
            supported_platforms=["Linux"],
        ),
        # pylint: enable=line-too-long
        path_checker=CompoundPathChecker(
            PackagePathChecker(
                expected_files=[
                    "models/Linux64_GCC-9.3/FVP_Corstone_SSE-310",
                ],
                backend_subfolder="models/Linux64_GCC-9.3",
            ),
            StaticPathChecker(
                static_backend_path=Path("/opt/VHT"),
                expected_files=[
                    "VHT_Corstone_SSE-310",
                ],
                copy_source=False,
                system_config="aiet/systems/corstone-310-vht/aiet-config.json",
            ),
        ),
        backend_installer=None,
    )

    return corstone_310


def get_corstone_installations() -> List[Installation]:
    """Get Corstone installations."""
    return [
        get_corstone_300_installation(),
        get_corstone_310_installation(),
    ]
