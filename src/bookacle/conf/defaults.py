from bookacle.models.embedding import (
    EmbeddingModelLike,
    SentenceTransformerEmbeddingModel,
)
from bookacle.models.qa import OllamaQAModel, QAModelLike
from bookacle.models.summarization import (
    HuggingFaceSummarizationModel,
    SummarizationModelLike,
)
from bookacle.splitter import DocumentSplitterLike, HuggingFaceMarkdownSplitter
from bookacle.tree.builder import ClusterTreeBuilder, TreeBuilderLike
from bookacle.tree.clustering import (
    ClusteringBackendLike,
    ClusteringFunctionLike,
    GMMClusteringBackend,
    raptor_clustering,
)
from bookacle.tree.config import RaptorTreeConfig, SelectionMode, TreeRetrieverConfig
from bookacle.tree.retriever import RetrieverLike, TreeRetriever

## UMAP Setting ##

UMAP_NEIGHBORS: int = 10

UMAP_METRIC: str = "cosine"

UMAP_LOW_MEMORY: bool = False

## Embedder Settings ##

EMBEDDER_USE_GPU: bool = True

EMBEDDING_MODEL: EmbeddingModelLike = SentenceTransformerEmbeddingModel(
    model_name="sentence-transformers/paraphrase-albert-small-v2",
    use_gpu=EMBEDDER_USE_GPU,
)

## Summarizer Settings ##

SUMMARIZER_USE_GPU: bool = True

SUMMARIZATION_LENGTH: int = 100

SUMMARIZATION_MODEL: SummarizationModelLike = HuggingFaceSummarizationModel(
    model_name="facebook/bart-large-cnn",
    summarization_length=SUMMARIZATION_LENGTH,
    use_gpu=SUMMARIZER_USE_GPU,
)

DOCUMENT_SPLITTER: DocumentSplitterLike = HuggingFaceMarkdownSplitter(
    tokenizer=EMBEDDING_MODEL.tokenizer
)

## Retriever Settings ##

RETRIEVER_EMBEDDING_MODEL = EMBEDDING_MODEL

RETRIEVER_EMBEDDER_USE_GPU: bool = True

RETRIEVER_THRESHOLD: float = 0.5

RETRIEVER_TOP_K: int = 5

RETRIEVER_SELECTION_MODE: SelectionMode = SelectionMode.TOP_K

RETRIEVER_MAX_TOKENS: int = 3500

RETRIEVER: RetrieverLike = TreeRetriever(
    config=TreeRetrieverConfig(
        embedding_model=RETRIEVER_EMBEDDING_MODEL,
        threshold=RETRIEVER_THRESHOLD,
        top_k=RETRIEVER_TOP_K,
        selection_mode=RETRIEVER_SELECTION_MODE,
        max_tokens=RETRIEVER_MAX_TOKENS,
    )
)

## Tree Builder Settings ##

REDUCTION_DIM: int = 10

CLUSTERING_FUNC: ClusteringFunctionLike = raptor_clustering

CLUSTERING_BACKEND: ClusteringBackendLike = GMMClusteringBackend(
    reduction_dim=REDUCTION_DIM
)

MAX_LENGTH_IN_CLUSTER: int = 3500

MAX_NUM_LAYERS: int = 5

TREE_BUILDER: TreeBuilderLike = ClusterTreeBuilder(
    config=RaptorTreeConfig(
        embedding_model=EMBEDDING_MODEL,
        summarization_model=SUMMARIZATION_MODEL,
        document_splitter=DOCUMENT_SPLITTER,
        clustering_func=CLUSTERING_FUNC,
        clustering_backend=CLUSTERING_BACKEND,
        max_length_in_cluster=MAX_LENGTH_IN_CLUSTER,
        max_num_layers=MAX_NUM_LAYERS,
    )
)

## QA Model Settings ##

QA_MODEL: QAModelLike = OllamaQAModel(model_name="qwen2:0.5b")
