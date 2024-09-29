"""This module implements document splitters for document chunking."""

import itertools
import re
from typing import Protocol

from bookacle.tokenizer import TokenizerLike
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import PreTrainedTokenizerBase


class DocumentSplitterLike(Protocol):
    """A protocol that defines the methods that a document splitter should implement."""

    def __call__(
        self, documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0
    ) -> list[Document]:
        """
        Split a list of documents into smaller chunks.

        Args:
            documents: The list of documents to be split.
            chunk_size: The size of each chunk.
            chunk_overlap: The overlap between consecutive chunks.

        Returns:
            The list of documents after splitting into chunks.
        """
        ...


class HuggingFaceTextSplitter:
    """Text-based document splitter which uses a HuggingFace tokenizer to calculate
    length when splitting.

    It uses Langchain's
    [RecursiveCharacterTextSplitter][langchain_text_splitters.RecursiveCharacterTextSplitter]
    and expects the list of documents to be in plain text.

    It implements the [DocumentSplitterLike][bookacle.splitters.DocumentSplitterLike]
    protocol.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, separators: list[str] | None = None
    ) -> None:
        """
        Initialize the splitter.

        Args:
            tokenizer: The HuggingFace tokenizer to use for calculating length.
            separators: The list of separators to use.
                        When `None`, the default separators are used: `["\\n\\n", "\\n", ".", "!", "?"]`.
        """
        self.tokenizer = tokenizer

        if separators is None:
            separators = ["\n\n", "\n", ".", "!", "?"]

        self.separators = separators

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tokenizer={self.tokenizer}, separators={self.separators})"

    def __call__(
        self, documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            strip_whitespace=True,
        )

        return splitter.split_documents(documents=documents)


class HuggingFaceMarkdownSplitter:
    """Markdown-based document splitter which uses a HuggingFace tokenizer to calculate
    length of chunks when splitting.

    It uses Langchain's
    [MarkdownTextSplitter][langchain_text_splitters.MarkdownTextSplitter]
    and expects the list of documents to be in Markdown.

    It implements the [DocumentSplitterLike][bookacle.splitters.DocumentSplitterLike]
    protocol.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialize the splitter.

        Args:
            tokenizer: The HuggingFace tokenizer to use to calculate length.
        """
        self.tokenizer = tokenizer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tokenizer={self.tokenizer})"

    def __call__(
        self, documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0
    ) -> list[Document]:
        splitter = MarkdownTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=True,
        )

        return splitter.split_documents(documents=documents)


class RaptorSplitter:
    """Document splitter which implements the chunking technique
    as defined in the RAPTOR paper.

    It expects a tokenizer which implements the
    [TokenizerLike][bookacle.tokenizer.TokenizerLike]
    protocol to calculate the length of chunks.

    For more details, see:
    https://github.com/parthsarthi03/raptor/blob/7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767/raptor/utils.py#L22.

    It implements the [DocumentSplitterLike][bookacle.splitters.DocumentSplitterLike]
    protocol.
    """

    def __init__(
        self, tokenizer: TokenizerLike, *, delimiters: list[str] | None = None
    ) -> None:
        """Initialize the splitter.

        Args:
            tokenizer: Tokenizer to use for calculating chunk lengths.
        """
        self.tokenizer = tokenizer

        if delimiters is None:
            delimiters = [".", "!", "?", "\n"]

        self.delimiters = delimiters

    def split_single_document(
        self, document: Document, chunk_size: int, chunk_overlap: int
    ) -> list[Document]:
        """Split a single document into chunks.

        Args:
            document: [Document][langchain_core.documents.Document] to split into chunks.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between each chunk.
        """
        text = document.page_content
        tokenizer = self.tokenizer

        regex_pattern = "|".join(map(re.escape, self.delimiters))
        sentences = re.split(regex_pattern, text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        def add_chunk(chunk: list[str]) -> tuple[list[str], int]:
            """Helper function to add chunk and apply overlap if necessary."""
            if chunk:
                chunks.append(" ".join(chunk))
                chunk = chunk[-chunk_overlap:] if chunk_overlap > 0 else []
                length = sum(len(tokenizer.encode(" " + sentence)) for sentence in chunk)
                return chunk, length

            return [], 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            token_count = len(tokenizer.encode(" " + sentence))

            if token_count > chunk_size:
                sub_sentences = re.split(r"[,;:]", sentence)

                sub_chunk: list[str] = []
                sub_length = 0

                for sub_sentence in sub_sentences:
                    if not sub_sentence.strip():
                        continue

                    sub_token_count = len(tokenizer.encode(" " + sub_sentence))

                    if sub_length + sub_token_count > chunk_size:
                        sub_chunk, sub_length = add_chunk(sub_chunk)

                    sub_chunk.append(sub_sentence)
                    sub_length += sub_token_count

                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))

                continue

            if current_length + token_count > chunk_size:
                current_chunk, current_length = add_chunk(current_chunk)

            current_chunk.append(sentence)
            current_length += token_count

        doc_args = document.dict()
        doc_args.pop("page_content")

        return [Document(page_content=chunk, **doc_args) for chunk in chunks]

    def __call__(
        self, documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0
    ) -> list[Document]:
        per_doc_chunks = [
            self.split_single_document(doc, chunk_size, chunk_overlap)
            for doc in documents
        ]

        return list(itertools.chain.from_iterable(per_doc_chunks))


if __name__ == "__main__":

    from bookacle.conf import settings
    from bookacle.loaders import pymupdf4llm_loader

    settings.validators.validate()

    tokenizer = settings.EMBEDDING_MODEL.tokenizer

    documents = pymupdf4llm_loader(file_path="data/c-language.pdf")

    delimiters = [
        r"\n",  # Newlines (e.g., paragraphs)
        r"#{1,6}\s",  # Headings (#, ##, ###, etc.)
        r"-\s|\*\s|\+\s",  # List markers (-, *, +)
        r"`{3,}[^`]*`{3,}",  # Code blocks
        r"!\[.*?\]\(.*?\)",  # Images
        r"\[.*?\]\(.*?\)",  # Links
        r"```[\s\S]*?```",  # Triple backticks for code
    ]

    splitter = RaptorSplitter(tokenizer=tokenizer, delimiters=delimiters)

    chunks = splitter(documents=documents, chunk_size=100, chunk_overlap=10)

    print(chunks)
