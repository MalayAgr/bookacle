from dataclasses import dataclass
from enum import Enum

from bookacle.models.embedding import EmbeddingModelLike
from bookacle.models.summarization import SummarizationModelLike
from bookacle.splitters import DocumentSplitterLike
from bookacle.tokenizer import TokenizerLike
from bookacle.tree.clustering import (
    ClusteringBackendLike,
    ClusteringFunctionLike,
    raptor_clustering,
)


class SelectionMode(Enum):
    TOP_K = "top_k"
    THRESHOLD = "threshold"


@dataclass
class RaptorTreeConfig:
    embedding_model: EmbeddingModelLike
    summarization_model: SummarizationModelLike
    document_splitter: DocumentSplitterLike
    clustering_func: ClusteringFunctionLike = raptor_clustering
    clustering_backend: ClusteringBackendLike | None = None
    max_length_in_cluster: int = 3500
    max_num_layers: int = 5

    @property
    def embedding_tokenizer(self) -> TokenizerLike:
        return self.embedding_model.tokenizer

    @property
    def summarization_tokenizer(self) -> TokenizerLike:
        return self.summarization_model.tokenizer


@dataclass
class TreeRetrieverConfig:
    embedding_model: EmbeddingModelLike
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: SelectionMode = SelectionMode.TOP_K
    max_tokens: int = 3500

    @property
    def tokenizer(self) -> TokenizerLike:
        return self.embedding_model.tokenizer
