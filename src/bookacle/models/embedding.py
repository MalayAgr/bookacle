from typing import Protocol, overload

import numpy as np
from bookacle.tokenizer import TokenizerLike
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase


class EmbeddingModelLike(Protocol):
    @property
    def tokenizer(self) -> TokenizerLike: ...

    @property
    def model_max_length(self) -> int: ...

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]: ...


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name_or_path=model_name, device="cuda" if use_gpu is True else "cpu"
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
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