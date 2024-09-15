from __future__ import annotations

from dataclasses import dataclass, field

from bookacle.models import EmbeddingModelLike, SummarizationModelLike


def concatenate_node_texts(nodes: list[Node]) -> str:
    return "\n\n".join(node.text for node in nodes) + "\n\n"


@dataclass
class Node:
    text: str
    index: int
    children: set[int] = field(repr=False)
    embeddings: list[float] = field(repr=False)
    metadata: dict[str, str] | None = field(default=None, repr=False)
    layer: int = 0

    @property
    def num_children(self) -> int:
        return len(self.children)

    @classmethod
    def from_text(
        cls,
        index: int,
        text: str,
        embedding_model: EmbeddingModelLike,
        children_indices: set[int] | None = None,
        metadata: dict[str, str] | None = None,
        layer: int = 0,
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
            layer=layer,
        )

    @classmethod
    def from_children(
        cls,
        cluster: list[Node],
        embedding_model: EmbeddingModelLike,
        summarization_model: SummarizationModelLike,
        next_node_index: int,
        layer: int = 0,
    ) -> Node:
        concatenated_text = concatenate_node_texts(nodes=cluster)

        summary = summarization_model.summarize(text=concatenated_text)

        return Node.from_text(
            index=next_node_index,
            text=summary,
            embedding_model=embedding_model,
            children_indices={node.index for node in cluster},
            layer=layer,
        )


@dataclass
class Tree:
    all_nodes: dict[int, Node] = field(repr=False)
    root_nodes: dict[int, Node] = field(repr=False)
    leaf_nodes: dict[int, Node] = field(repr=False)
    num_layers: int
    layer_to_nodes: dict[int, list[Node]] = field(repr=False)

    @property
    def num_nodes(self) -> int:
        return len(self.all_nodes)

    @property
    def top_layer(self) -> int:
        return self.num_layers - 1

    def tolist(self) -> list[Node]:
        return list(self.all_nodes.values())

    def get_node(self, index: int) -> Node:
        return self.all_nodes[index]

    def fetch_layer(self, layer: int) -> list[Node]:
        return self.layer_to_nodes[layer]

    def fetch_node_layer(self, node_idx: int) -> int:
        return self.all_nodes[node_idx].layer
