# SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: LicenseRef-LICENSE
"""Module for the installation of Argo via Docker."""
from __future__ import annotations

from rich.status import Status

import docker
from mlia.backend.install import artifactory_credentials_from_env
from mlia.backend.install import DownloadAndInstall
from mlia.backend.install import Installation
from mlia.backend.install import InstallationType

DOCKER_IMAGE_NAME = "argo-app"


class DockerInstallation(Installation):
    """Define installation process for docker images."""

    def __init__(
        self,
        name: str,
        description: str,
        image_name: str,
        registry: str | None,
    ) -> None:
        """Init the installation."""
        super().__init__(name, description)
        self._image_name = image_name
        self._registry = registry
        self._repository = (
            f"{self._registry}/{self._image_name}"
            if self._registry
            else self._image_name
        )

    @property
    def image_name(self) -> str:
        """Get the image name."""
        return self._image_name

    @property
    def registry(self) -> str | None:
        """Get the registry."""
        return self._registry

    @property
    def repository(self) -> str:
        """Get the repository."""
        return self._repository

    @property
    def could_be_installed(self) -> bool:
        """Check if backend could be installed in current environment."""
        return True

    @property
    def already_installed(self) -> bool:
        """Check if docker container is already installed."""
        client = docker.from_env()  # type: ignore
        return bool(client.images.list(name=self.image_name))

    def supports(self, install_type: InstallationType) -> bool:
        """Check if installation supports requested installation type."""
        return isinstance(install_type, DownloadAndInstall)

    def install(self, install_type: InstallationType) -> None:
        """Pull the docker image for the backend from the registry."""
        if not self.supports(install_type):
            raise TypeError(f"Unsupported installation type '{install_type}'.")

        client = docker.from_env()  # type: ignore

        if self.registry:
            self._login(client)

        with Status(status="Installing argo backend...", spinner="dots") as status:
            for line in client.api.pull(
                repository=self.repository, stream=True, decode=True
            ):
                if line.get("status"):
                    status.update(line["status"])
                else:
                    status.update("Installing argo backend...")

        if self.registry:
            # Tag the image without the registry URL as prefix, e.g. tag image
            # 'my-registry.com/my-image' as 'my-image', just as it would be named
            # if built locally.
            if not client.api.tag(self.repository, self.image_name):
                raise RuntimeError(
                    f"Failed to tag docker image {self.repository} as "
                    f"{self.image_name}."
                )

    def uninstall(self) -> None:
        """Remove the docker image by id (also removed tagged versions)."""
        client = docker.from_env()  # type: ignore
        for image in client.images.list(self.repository):
            client.images.remove(image.id, force=True)

    def _login(self, client: docker.DockerClient) -> None:  # type: ignore
        """Log in to the registry using credentials from environment variables."""
        try:
            username, password = artifactory_credentials_from_env()
        except RuntimeError as ex:
            raise RuntimeError(
                f"Failed to retrieve credentials for registry '{self.registry}'."
            ) from ex

        client.login(
            username=username,
            password=password,
            registry=self.registry,
        )


def get_argo_installation() -> Installation:
    """Get all information to install Argo."""
    return DockerInstallation(
        name="argo",
        description="Argo performance backend",
        image_name=DOCKER_IMAGE_NAME,
        registry="ml-tooling--docker-local.artifactory.geo.arm.com",
    )
