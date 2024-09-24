from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from bookacle.loaders import LOADER_MANAGER
from langchain_core.documents import Document

app = typer.Typer()


def load_data(
    file_path: str,
    loader: Enum,
    start_page: int = 0,
    end_page: int | None = 0,
) -> list[Document]:
    pdf_loader = LOADER_MANAGER.get(loader.name.lower())

    if pdf_loader is None:
        raise ValueError(f"Loader {pdf_loader} is not supported.")

    return pdf_loader(file_path, start_page=start_page, end_page=end_page)


@app.command()
def chat(
    file_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            show_default=False,
            file_okay=True,
            dir_okay=False,
            help="Path to the PDF file.",
        ),
    ],
    loader: Annotated[
        LOADER_MANAGER.to_enum,  # type: ignore
        typer.Option(case_sensitive=False, help="Loader to use."),
    ] = "pymupdf4llm",
    start_page: Annotated[
        int,
        typer.Option(
            help="The page (0-based) in the PDF file to start reading from. "
            "If not provided, defaults to 0, reading from the beginning.",
            show_default=False,
        ),
    ] = 0,
    end_page: Annotated[
        int | None,
        typer.Option(
            help="The page (0-based) in the PDF file to stop reading at (not inclusive). "
            "If not provided, the document will be read till the end.",
            show_default=False,
        ),
    ] = None,
) -> None:
    documents = load_data(
        file_path=str(file_path),
        loader=loader,
        start_page=start_page,
        end_page=end_page,
    )
    print(len(documents))


if __name__ == "__main__":
    typer.run(chat)
