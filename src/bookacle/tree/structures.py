from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bookacle.models.embedding import EmbeddingModelLike
from bookacle.models.summarization import SummarizationModelLike


def concatenate_node_texts(nodes: list[Node]) -> str:
    """Concatenate the texts of a list of nodes."""
    return "\n\n".join(node.text for node in nodes) + "\n\n"


@dataclass
class Node:
    """A node in the RAPTOR tree.

    Attributes:
        text: The text of the node.
        index: The global index of the node in the tree.
        children: The global indices of the children nodes.
        embeddings: Embeddings of the node's text.
        metadata: Metadata about the node's text.
        layer: Tree layer the node belongs to.
    """

    text: str
    index: int
    children: set[int] = field(repr=False)
    embeddings: list[float] = field(repr=False)
    metadata: dict[str, Any] | None = field(default=None, repr=False)
    layer: int = 0

    @property
    def num_children(self) -> int:
        """
        Returns:
            Number of children nodes.
        """
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
        """Create a node from text by embedding it using the given embedding model.

        Args:
            index: The global index of the node in the tree.
            text: The text of the node.
            embedding_model: The embedding model to use for embedding the text.
            children_indices: The global indices of the children nodes.
            metadata: Metadata about the node's text.
            layer: Tree layer the node belongs to.

        Returns:
            A node created from the text.
        """
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
        children: list[Node],
        embedding_model: EmbeddingModelLike,
        summarization_model: SummarizationModelLike,
        index: int,
        layer: int = 0,
    ) -> Node:
        """Create a node from a list of children nodes by summarizing their texts.

        The text of the children nodes is concatenated using
        [concatenate_node_texts()][bookacle.tree.structures.concatenate_node_texts] and passed to
        the summarization model to generate a summary.

        Args:
            children: A list of children nodes.
            embedding_model: The embedding model to use for embedding the summarized text.
            summarization_model: The summarization model to use for summarizing
                                 the text of the children nodes.
            index: The global index of the node in the tree.
            layer: Tree layer the node belongs to.

        Returns:
            A node created from the children nodes.
        """
        concatenated_text = concatenate_node_texts(nodes=children)

        summary = summarization_model.summarize(text=concatenated_text)

        return Node.from_text(
            index=index,
            text=summary,
            embedding_model=embedding_model,
            children_indices={node.index for node in children},
            layer=layer,
        )


@dataclass
class Tree:
    """A RAPTOR tree.

    Attributes:
        all_nodes: All nodes in the tree, mapped to their global indices.
        root_nodes: Root nodes in the tree, mapped to their global indices.
        leaf_nodes: Leaf nodes in the tree, mapped to their global indices.
        num_layers: Number of layers in the tree.
        layer_to_nodes: Nodes in a layer, mapped to their layer index.
    """

    all_nodes: dict[int, Node] = field(repr=False)
    root_nodes: dict[int, Node] = field(repr=False)
    leaf_nodes: dict[int, Node] = field(repr=False)
    num_layers: int
    layer_to_nodes: dict[int, list[Node]] = field(repr=False)

    @property
    def num_nodes(self) -> int:
        """
        Returns:
            The number of nodes in the tree.
        """
        return len(self.all_nodes)

    @property
    def top_layer(self) -> int:
        """
        Returns:
            Index of the root layer of the tree.
        """
        return self.num_layers - 1

    def tolist(self) -> list[Node]:
        """
        Returns:
            List of all nodes in the tree.
        """
        return list(self.all_nodes.values())

    def get_node(self, index: int) -> Node:
        """Fetch a node in the tree by its global index.

        Args:
            index: The global index of the node to fetch.

        Returns:
            The node with the given global index.
        """
        return self.all_nodes[index]

    def fetch_layer(self, layer: int) -> list[Node]:
        """Fetch all nodes in a layer of the tree.

        Args:
            layer: The layer index to fetch nodes from.

        Returns:
            List of nodes in the given layer.
        """
        return self.layer_to_nodes[layer]

    def fetch_node_layer(self, node_idx: int) -> int:
        """Fetch the index of the layer a node belongs to in the tree.

        Args:
            node_idx: The global index of the node.

        Returns:
            The index of the layer the node belongs to.
        """
        return self.all_nodes[node_idx].layer
