"""This module defines protocols and concrete implementations for embedding models used for text representation."""

from typing import Protocol, overload, runtime_checkable

import numpy as np
from bookacle.tokenizer import TokenizerLike
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase


@runtime_checkable
class EmbeddingModelLike(Protocol):
    """A protocol that defines the methods and attributes that an embedding model should implement."""

    @property
    def tokenizer(self) -> TokenizerLike:
        """
        Returns:
            The tokenizer used by the model.
        """
        ...

    @property
    def model_max_length(self) -> int:
        """
        Returns:
            The maximum length of the input that the model can accept.
        """
        ...

    @overload
    def embed(self, text: str) -> list[float]:
        """Embed a single input text.

        Args:
            text: The input text to embed.

        Returns:
            The embeddings of the input text.
        """
        ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]:
        """Embed a list of input texts.

        Args:
            text: The list of input texts to embed.

        Returns:
            The embeddings of the input texts.
        """
        ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed the input text or list of texts.

        Args:
            text: The input text or list of input texts to embed.

        Returns:
            The embeddings of the input text or list of texts.
        """
        ...


class SentenceTransformerEmbeddingModel:
    """An embedding model that uses the [SentenceTransformer](https://sbert.net/) library.

    It implements the [EmbeddingModelLike][bookacle.models.embedding.EmbeddingModelLike] protocol.

    Attributes:
        model_name (str): The name of the model to use.
        use_gpu (bool): Whether to use the GPU for inference.
        model (SentenceTransformer): The SentenceTransformer model.
    """

    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        """Initialize the embedding model.

        Args:
            model_name: The name of the model to use.
            use_gpu: Whether to use the GPU for inference.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = SentenceTransformer(
            model_name_or_path=model_name, device="cuda" if use_gpu is True else "cpu"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, use_gpu={self.use_gpu})"

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Returns:
            The tokenizer used by the underlying model.
        """
        return self.model.tokenizer

    @property
    def model_max_length(self) -> int:
        return self.model.max_seq_length

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        embeddings = self.model.encode(text, normalize_embeddings=True)

        assert isinstance(embeddings, np.ndarray)

        return embeddings.tolist()


if __name__ == "__main__":
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    embeddings = embedding_model.embed(["This is a test", "This is another test"])
    print(embeddings)
    print(len(embeddings))
