from dataclasses import dataclass
from enum import Enum, auto
from functools import cache, cached_property

from bookacle.models import EmbeddingModel, SummarizationModel
from pyexpat import model
from transformers import PreTrainedTokenizerBase


class SelectionMode(Enum):
    TOP_K = auto()
    THRESHOLD = auto()


class RaptorTreeConfig:
    def __init__(
        self,
        embedding_model_name: str,
        summarization_model_name: str,
        max_tokens: int = 100,
        num_layers: int = 5,
        threshold: float = 0.5,
        top_k: int = 5,
        selection_mode: SelectionMode = SelectionMode.TOP_K,
        summarization_length: int = 100,
        use_gpu: bool = False,
        max_workers: int = 4,
    ):
        self.embedding_model_name = embedding_model_name
        self.summarization_model_name = summarization_model_name
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name, use_gpu=use_gpu
        )
        self.summarization_model = SummarizationModel(
            model_name=summarization_model_name,
            max_tokens=summarization_length,
            use_gpu=use_gpu,
        )
        self.max_tokens = max_tokens
        self.num_layers = num_layers
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
