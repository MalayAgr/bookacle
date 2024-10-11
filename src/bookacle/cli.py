from __future__ import annotations

import importlib
import re
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from bookacle.chat import Chat
from bookacle.conf import settings
from bookacle.document import Document
from bookacle.loaders import LOADER_MANAGER
from bookacle.tree.builder import TreeBuilderLike
from rich.console import Console

# Register custom document loaders
if settings.CUSTOM_LOADERS_DIR:
    import sys

    custom_loader_dir = settings.CUSTOM_LOADERS_DIR

    sys.path.append(custom_loader_dir)

    pattern = re.compile(r"^(?!.*__init__\.py$).*.py$")
    for module in (
        path for path in Path(custom_loader_dir).rglob("*.py") if pattern.match(str(path))
    ):
        _ = importlib.import_module(module.stem)

    sys.path.remove(custom_loader_dir)


def version_callback(value: bool) -> None:
    if value:
        import bookacle

        print(f"v{bookacle.__version__}")
        raise typer.Exit()


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


app = typer.Typer()


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
    loader: Annotated[  # type: ignore
        LOADER_MANAGER.enum,  # type: ignore
        typer.Option("--loader", "-l", case_sensitive=False, help="Loader to use."),
    ] = "pymupdf4llm",
    start_page: Annotated[
        int,
        typer.Option(
            "--start-page",
            "-s",
            help="The page (0-based) in the PDF file to start reading from. "
            "If not provided, defaults to 0, reading from the beginning.",
            show_default=False,
        ),
    ] = 0,
    end_page: Annotated[
        int | None,
        typer.Option(
            "--end-page",
            "-e",
            help="The page (0-based) in the PDF file to stop reading at (not inclusive). "
            "If not provided, the document will be read till the end.",
            show_default=False,
        ),
    ] = None,
    user_avatar: Annotated[
        str,
        typer.Option(
            "--user-avatar",
            "-a",
            help="Avatar that should be used for the user during chat.",
        ),
    ] = "👤",
    history_file: Annotated[
        str,
        typer.Option(
            "--history_file", "-h", help="File where chat history should be stored."
        ),
    ] = str(Path.home() / ".bookacle-chat-history.txt"),
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config-file",
            "-c",
            exists=True,
            file_okay=True,
            dir_okay=False,
            show_default=False,
            help="Custom configuration file. If not provided, the default settings are used.",
        ),
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option(
            "--prompt-file",
            "-p",
            exists=True,
            file_okay=True,
            dir_okay=False,
            show_default=False,
            help="Custom prompts file. If not provided, the default prompts are used.",
        ),
    ] = None,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", "-v", callback=version_callback, help="Print version and exit."
        ),
    ] = None,
) -> None:
    if config_file is not None:
        settings.load_file(config_file)

    if prompt_file is not None:
        settings.load_file(prompt_file)

    documents = load_data(
        file_path=str(file_path),
        loader=loader,
        start_page=start_page,
        end_page=end_page,
    )

    console = Console()

    settings.validators.validate()

    tree_builder: TreeBuilderLike = settings.TREE_BUILDER

    tree = tree_builder.build_from_documents(
        documents=documents,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chat = Chat(
        retriever=settings.RETRIEVER,
        qa_model=settings.QA_MODEL,
        console=console,
        history_file=history_file,
        user_avatar=user_avatar,
    )

    chat.run(
        tree=tree,
        stream=settings.STREAM_OUTPUT,
        system_prompt=settings.QA_MODEL_SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    typer.run(chat)
