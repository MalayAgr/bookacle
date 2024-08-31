from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

from bookacle.models import EmbeddingModel, SummarizationModel
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)


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
    ):
        self.embedding_model_name = embedding_model_name
        self.summarization_model_name = summarization_model_name
        self.max_tokens = max_tokens
        self.num_layers = num_layers
        self.threshold = threshold
        self.top_k = top_k
        self.selection_mode = selection_mode
        self.summarization_length = summarization_length

    @cached_property
    def embedding_model(self) -> EmbeddingModel:
        return EmbeddingModel(model_name=self.embedding_model_name)

    @cached_property
    def summarization_model(self) -> SummarizationModel:
        return SummarizationModel(
            model_name=self.summarization_model_name,
            max_tokens=self.summarization_length,
        )
