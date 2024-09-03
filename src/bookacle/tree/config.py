from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, cached_property

from bookacle.models import (
    EmbeddingModel,
    EmbeddingModelLike,
    SummarizationModel,
    SummarizationModelLike,
)
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
        max_tokens: int = 100,
        max_num_layers: int = 5,
        threshold: float = 0.5,
        top_k: int = 5,
        selection_mode: SelectionMode = SelectionMode.TOP_K,
        summarization_length: int = 100,
        use_gpu: bool = False,
        max_workers: int = 4,
    ):
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.max_tokens = max_tokens
        self.max_num_layers = max_num_layers
        self.threshold = threshold
        self.top_k = top_k
        self.selection_mode = selection_mode
        self.summarization_length = summarization_length
        self.max_workers = max_workers

    @property
    def embedding_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.embedding_model.tokenizer

    @property
    def summarization_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.summarization_model.tokenizer
