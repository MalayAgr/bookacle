from typing import Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizerBase


class DocumentSplitterLike(Protocol):
    def __call__(
        self, documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0
    ) -> list[Document]: ...


class HuggingFaceDocumentSplitter:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, separators: list[str] | None = None
    ) -> None:
        self.tokenizer = tokenizer

        if separators is None:
            separators = ["\n\n", "\n", ".", "!", "?"]

        self.separators = separators

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
