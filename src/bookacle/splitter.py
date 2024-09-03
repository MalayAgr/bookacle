from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizerBase


class DocumentSplitterLike(Protocol):
    def __call__(
        self, documents: list[Document], max_tokens: int = 100, overlap: int = 0
    ) -> list[Document]: ...


class HuggingFaceDocumentSplitter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self, documents: list[Document], max_tokens: int = 100, overlap: int = 0
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=max_tokens,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
            ],
            keep_separator="end",
        )

        return splitter.split_documents(documents=documents)
