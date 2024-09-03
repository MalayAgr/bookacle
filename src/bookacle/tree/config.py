from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from bookacle.models import EmbeddingModelLike, SummarizationModelLike
from bookacle.splitter import DocumentSplitterLike


class SelectionMode(Enum):
    TOP_K = auto()
    THRESHOLD = auto()


@dataclass
class RaptorTreeConfig:
    embedding_model: EmbeddingModelLike
    summarization_model: SummarizationModelLike
    document_splitter: DocumentSplitterLike
    max_num_layers: int = 5
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: SelectionMode = SelectionMode.TOP_K
    max_workers: int = 4

    @property
    def embedding_tokenizer(self) -> Any:
        return self.embedding_model.tokenizer

    @property
    def summarization_tokenizer(self) -> Any:
        return self.summarization_model.tokenizer
