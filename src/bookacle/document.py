"""This module defines the document structure used throughout the package."""

from typing import Any, NotRequired, TypedDict


class Document(TypedDict):
    """A [TypedDict][typing.TypedDict] that represents a page in a PDF file.

    Attributes:
        page_content: The text content of the page.
        metadata: Additional metadata about the page.
    """

    page_content: str
    metadata: NotRequired[dict[str, Any]]
