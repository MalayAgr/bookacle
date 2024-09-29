"""This module defines functions for loading PDF documents and some utilities to manage loaders."""

from __future__ import annotations

import itertools
from collections import UserDict
from enum import Enum
from typing import Callable, Protocol

import pymupdf
import pymupdf4llm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class LoaderLike(Protocol):
    """A protocol that all document loaders should follow."""

    def __call__(
        self, file_path: str, start_page: int = 0, end_page: int | None = None
    ) -> list[Document]:
        """
        Load a PDF document.

        Args:
            file_path: The path to the PDF file.
            start_page: The starting (0-based) page number in the PDF to begin reading from.
            end_page: The ending (0-based) page number to stop reading at (non-inclusive).
                      When `None`, all pages in the PDF are read.

        Returns:
            Pages in the file.
        """
        ...


class LoaderManager(UserDict[str, LoaderLike]):
    """Manager to maintain registry of all document loaders.

    It behaves like a dictionary, where each document loader is registered to a name.

    Example:
        ```python
        from bookacle.loaders import LoaderManager, register_loader
        from langchain_core.documents import Document

        manager = LoaderManager()

        @register_loader(name="custom_loader", manager=manager)
        def doc_loader(file_path: str, start_page: int = 0, end_page: int | None = None) -> list[Document]:
            ...

        manager["custom_loader"] is doc_loader
        ```
    """

    @property
    def enum(self) -> Enum:
        """Obtain the names of the document loaders as an Enum."""
        return Enum("LoaderChoices", {name.upper(): name for name in self.keys()})


LOADER_MANAGER = LoaderManager()
"""Default loader manager."""


def register_loader(
    name: str, manager: LoaderManager | None = None
) -> Callable[[LoaderLike], LoaderLike]:
    """A decorator that registers a loader function with the loader manager.

    Args:
        name: The name to map the loader function to.
        manager: The manager to register the function with.
                 If `None`, [`LOADER_MANAGER`][bookacle.loaders.LOADER_MANAGER] is used.
    """
    if manager is None:
        manager = LOADER_MANAGER

    def decorator(func: LoaderLike) -> LoaderLike:
        manager[name] = func
        return func

    return decorator


@register_loader("pymupdf4llm")
def pymupdf4llm_loader(
    file_path: str, start_page: int = 0, end_page: int | None = None
) -> list[Document]:
    """Document loader which uses `pymupdf4llm` to load the PDF as Markdown.

    Can be accessed using the name `'pymupdf4llm'` via the default loader manager.
    """
    with pymupdf.open(file_path) as doc:
        if end_page is None:
            end_page = doc.page_count

        assert isinstance(end_page, int)

        pages = pymupdf4llm.to_markdown(
            doc, page_chunks=True, pages=list(range(start_page, end_page))
        )

    return [Document(page_content=page["text"], **page) for page in pages]  # type: ignore


@register_loader("pymupdf")
def pymupdf_loader(
    file_path: str, start_page: int = 0, end_page: int | None = None
) -> list[Document]:
    """Document loader which uses `pymupdf` to load the PDF as text.

    Can be accessed using the name `'pymupdf'` via the default loader manager.
    """
    loader = PyMuPDFLoader(file_path=file_path)

    if start_page == 0 and end_page is None:
        return loader.load()

    pages = enumerate(loader.lazy_load())
    remaining_pages = itertools.dropwhile(lambda page: page[0] < start_page, pages)

    if end_page is not None:
        return [
            page
            for _, page in itertools.takewhile(
                lambda page: page[0] < end_page, remaining_pages
            )
        ]

    return [page for _, page in remaining_pages]
