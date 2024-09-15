from __future__ import annotations

from functools import cached_property
from typing import Any, Generic, Protocol, TypeVar

from bookacle.models import (
    EmbeddingModelLike,
    HuggingFaceSummarizationModel,
    QAModelLike,
    SentenceTransformerEmbeddingModel,
    SummarizationModelLike,
)
from bookacle.splitter import DocumentSplitterLike, HuggingFaceMarkdownSplitter
from bookacle.tree.builder import ClusterTreeBuilder, TreeBuilderLike
from bookacle.tree.config import RaptorTreeConfig, TreeRetrieverConfig
from bookacle.tree.retriever import RetrieverLike, TreeRetriever
from bookacle.tree.structures import Tree
from langchain_core.documents import Document

_T = TypeVar("_T")


class _SingletonMeta(type, Generic[_T]):

    _instances: dict[_SingletonMeta[_T], _T] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> _T:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Defaults(metaclass=_SingletonMeta):
    @cached_property
    def embedding_model(self) -> EmbeddingModelLike:
        return SentenceTransformerEmbeddingModel(
            model_name="sentence-transformers/paraphrase-albert-small-v2", use_gpu=True
        )

    @cached_property
    def summarization_model(self) -> SummarizationModelLike:
        return HuggingFaceSummarizationModel(
            model_name="facebook/bart-large-cnn", use_gpu=True
        )

    @property
    def document_splitter(self) -> DocumentSplitterLike:
        return HuggingFaceMarkdownSplitter(tokenizer=self.embedding_model.tokenizer)  # type: ignore

    @property
    def tree_builder(self) -> TreeBuilderLike:
        config = RaptorTreeConfig(
            embedding_model=self.embedding_model,
            summarization_model=self.summarization_model,
            document_splitter=self.document_splitter,
        )
        return ClusterTreeBuilder(config=config)

    @property
    def retriever(self) -> RetrieverLike:
        return TreeRetriever(
            config=TreeRetrieverConfig(embedding_model=self.embedding_model)
        )

    @property
    def qa_model(self) -> QAModelLike:
        raise NotImplementedError


class RAGLike(Protocol):
    tree_builder: TreeBuilderLike
    retriever: RetrieverLike
    qa_model: QAModelLike
    tree: Tree
    _tree: Tree | None = None

    def answer(self, question: str, *args, **kwargs) -> str: ...  # type: ignore


class RAG:
    def __init__(
        self,
        tree_builder: TreeBuilderLike | None = None,
        retriever: RetrieverLike | None = None,
        qa_model: QAModelLike | None = None,
    ) -> None:
        defaults = Defaults()

        self.tree_builder = tree_builder or defaults.tree_builder
        self.retriever = retriever or defaults.retriever
        self.qa_model = qa_model or defaults.qa_model
        self._tree: Tree | None = None

    @property
    def tree(self) -> Tree:
        if self._tree is None:
            raise ValueError("Tree not built yet")
        return self._tree

    @tree.setter
    def tree(self, value: Tree) -> None:
        self._tree = value

    def build_tree(  # type: ignore
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.tree = self.tree_builder.build_from_documents(
            documents,
            chunk_size,
            chunk_overlap,
            *args,
            **kwargs,
        )
