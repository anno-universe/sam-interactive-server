"""SDK-specific exception classes."""


class AnnoSDKError(Exception):
    """Base exception for all SDK errors."""


class AnnoAPIError(AnnoSDKError):
    """Raised when the Anno backend returns an HTTP 4xx or 5xx response.

    Attributes:
        status_code: The HTTP status code.
        detail: The response body decoded as a string.
    """

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AnnoConnectionError(AnnoSDKError):
    """Raised on network / timeout errors when contacting the Anno backend."""
