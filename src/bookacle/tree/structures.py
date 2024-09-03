from __future__ import annotations

from dataclasses import dataclass

from bookacle.models import EmbeddingModelLike, HuggingFaceEmbeddingModel


@dataclass
class Node:
    text: str
    index: int
    children: set[int]
    embeddings: list[float]
    metadata: dict[str, str] | None = None

    @classmethod
    def from_text(
        cls,
        index: int,
        text: str,
        embedding_model: EmbeddingModelLike,
        children_indices: set[int] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Node:
        if children_indices is None:
            children_indices = set()

        embeddings = embedding_model.embed(text=text)

        return cls(
            text=text,
            index=index,
            children=children_indices,
            embeddings=embeddings,
            metadata=metadata,
        )


@dataclass
class Tree:
    all_nodes: dict[int, Node]
    root_nodes: dict[int, Node]
    leaf_nodes: dict[int, Node]
    num_layers: int
    layer_to_nodes: dict[int, set[int]]
