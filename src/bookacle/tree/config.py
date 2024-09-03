from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, cached_property

from bookacle.models import (
    EmbeddingModelLike,
    HuggingFaceEmbeddingModel,
    HuggingFaceSummarizationModel,
    SummarizationModelLike,
)
from bookacle.splitter import DocumentSplitterLike
from pyexpat import model
from transformers import PreTrainedTokenizerBase


class SelectionMode(Enum):
    TOP_K = auto()
    THRESHOLD = auto()


class RaptorTreeConfig:
    def __init__(
        self,
        embedding_model: EmbeddingModelLike,
        summarization_model: SummarizationModelLike,
        document_splitter: DocumentSplitterLike,
        max_num_layers: int = 5,
        threshold: float = 0.5,
        top_k: int = 5,
        selection_mode: SelectionMode = SelectionMode.TOP_K,
        max_workers: int = 4,
    ):
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.document_splitter = document_splitter
        self.max_num_layers = max_num_layers
        self.threshold = threshold
        self.top_k = top_k
        self.selection_mode = selection_mode
        self.max_workers = max_workers

    @property
    def embedding_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.embedding_model.tokenizer

    @property
    def summarization_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.summarization_model.tokenizer
