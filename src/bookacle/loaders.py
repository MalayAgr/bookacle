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
    def __call__(
        self, file_path: str, start_page: int = 0, end_page: int | None = None
    ) -> list[Document]: ...


class _LoaderManager(UserDict[str, LoaderLike]):
    @property
    def to_enum(self) -> Enum:
        return Enum("LoaderChoices", {name.upper(): name for name in self.keys()})


LOADER_MANAGER = _LoaderManager()


def register_loader(name: str) -> Callable[[LoaderLike], LoaderLike]:
    """A decorator that registers a loader function with the loader manager.

    Args:
        name (str): The name to map the loader function to.
    """

    def decorator(func: LoaderLike) -> LoaderLike:
        LOADER_MANAGER[name] = func
        return func

    return decorator


@register_loader("pymupdf4llm")
def pymupdf4llm_loader(
    file_path: str, start_page: int = 0, end_page: int | None = None
) -> list[Document]:
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
