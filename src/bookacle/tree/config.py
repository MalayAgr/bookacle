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
    """Selection modes supported by the retriever."""

    TOP_K = "top_k"
    """Selection using Top K."""
    THRESHOLD = "threshold"
    """Selection using a threshold value on the distance."""


@dataclass
class ClusterTreeConfig:
    """Configuration for [ClusterTreeBuilder][bookacle.tree.builder.ClusterTreeBuilder].

    Parameters:
        embedding_model: The embedding model to use.
        summarization_model: The summarization model to use.
        document_splitter: The document splitter to use.
        clustering_func: The clustering function to use.
        clustering_backend: The clustering backend to use.
        max_length_in_cluster: The maximum length of a cluster.
        max_num_layers: The maximum number of layers
    """

    embedding_model: EmbeddingModelLike
    summarization_model: SummarizationModelLike
    document_splitter: DocumentSplitterLike
    clustering_func: ClusteringFunctionLike = raptor_clustering
    clustering_backend: ClusteringBackendLike | None = None
    max_length_in_cluster: int = 3500
    max_num_layers: int = 5

    @property
    def embedding_tokenizer(self) -> TokenizerLike:
        """
        Returns:
            The tokenizer of the embedding model.
        """
        return self.embedding_model.tokenizer

    @property
    def summarization_tokenizer(self) -> TokenizerLike:
        """
        Returns:
            The tokenizer of the summarization model.
        """
        return self.summarization_model.tokenizer


@dataclass
class TreeRetrieverConfig:
    """Configuration for [TreeRetriever][bookacle.tree.retriever.TreeRetriever].

    Parameters:
        embedding_model: The embedding model to use.
        threshold: The threshold value for selection when using threshold mode for selection.
        top_k: The number of top results to return when using top k mode for selection.
        selection_mode: The selection mode to use.
        max_tokens: The maximum number of tokens to retrieve.
    """

    embedding_model: EmbeddingModelLike
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: SelectionMode = SelectionMode.TOP_K
    max_tokens: int = 3500

    @property
    def tokenizer(self) -> TokenizerLike:
        """
        Returns:
            The tokenizer of the embedding model.
        """
        return self.embedding_model.tokenizer
