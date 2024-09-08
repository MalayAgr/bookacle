from __future__ import annotations

from dataclasses import dataclass

from bookacle.models import EmbeddingModelLike, SummarizationModelLike


def concatenate_node_texts(nodes: list[Node]) -> str:
    return "\n\n".join(" ".join(node.text.splitlines()) for node in nodes) + "\n\n"


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

    @classmethod
    def from_children(
        cls,
        cluster: list[Node],
        embedding_model: EmbeddingModelLike,
        summarization_model: SummarizationModelLike,
        # new_level_nodes: dict[int, Node],
        next_node_index: int,
        summarization_length: int,
    ) -> Node:
        concatenated_text = concatenate_node_texts(nodes=cluster)

        summary = summarization_model.summarize(
            text=concatenated_text, max_tokens=summarization_length
        )

        return Node.from_text(
            index=next_node_index,
            text=summary,
            embedding_model=embedding_model,
            children_indices={node.index for node in cluster},
        )


@dataclass
class Tree:
    all_nodes: dict[int, Node]
    root_nodes: dict[int, Node]
    leaf_nodes: dict[int, Node]
    num_layers: int
    layer_to_nodes: dict[int, list[Node]]
