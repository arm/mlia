"""IP configuration module."""


class IPConfiguration:  # pylint: disable=too-few-public-methods
    """Base class for IP configuration."""

    def __init__(self, ip_class: str) -> None:
        """Init IP configuration instance."""
        self.ip_class = ip_class
