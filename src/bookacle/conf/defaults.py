from pathlib import Path

from bookacle.models.embedding import (
    EmbeddingModelLike,
    SentenceTransformerEmbeddingModel,
)
from bookacle.models.qa import OllamaQAModel, QAModelLike
from bookacle.models.summarization import (
    HuggingFaceSummarizationModel,
    SummarizationModelLike,
)
from bookacle.splitter import HuggingFaceMarkdownSplitter
from bookacle.tree.builder import ClusterTreeBuilder, TreeBuilderLike
from bookacle.tree.clustering import GMMClusteringBackend, raptor_clustering
from bookacle.tree.config import RaptorTreeConfig, SelectionMode, TreeRetrieverConfig
from bookacle.tree.retriever import RetrieverLike, TreeRetriever

## Default Embedding Model ##

EMBEDDING_MODEL: EmbeddingModelLike = SentenceTransformerEmbeddingModel(
    model_name="sentence-transformers/paraphrase-albert-small-v2",
    use_gpu=True,
)

## Default Summarization Model ##

SUMMARIZATION_MODEL: SummarizationModelLike = HuggingFaceSummarizationModel(
    model_name="facebook/bart-large-cnn",
    summarization_length=100,
    use_gpu=True,
)

## Default Retriever ##

RETRIEVER: RetrieverLike = TreeRetriever(
    config=TreeRetrieverConfig(
        embedding_model=EMBEDDING_MODEL,
        threshold=0.5,
        top_k=5,
        selection_mode=SelectionMode.TOP_K,
        max_tokens=3500,
    )
)

## Default Tree Builder ##

TREE_BUILDER: TreeBuilderLike = ClusterTreeBuilder(
    config=RaptorTreeConfig(
        embedding_model=EMBEDDING_MODEL,
        summarization_model=SUMMARIZATION_MODEL,
        document_splitter=HuggingFaceMarkdownSplitter(
            tokenizer=EMBEDDING_MODEL.tokenizer
        ),
        clustering_func=raptor_clustering,
        clustering_backend=GMMClusteringBackend(
            reduction_dim=10, umap_metric="cosine", umap_low_memory=False
        ),
        max_length_in_cluster=3500,
        max_num_layers=5,
    )
)

CHUNK_SIZE: int | None = None

CHUNK_OVERLAP: int | None = None

## Default QA Model ##

QA_MODEL: QAModelLike = OllamaQAModel(model_name="qwen2:0.5b")

STREAM_OUTPUT: bool = True

CUSTOM_LOADER_DIR: str = ""
