from typing import Protocol

import numpy as np
import numpy.typing as npt
from bookacle.tree.config import SelectionMode, TreeRetrieverConfig
from bookacle.tree.structures import Node, Tree, concatenate_node_texts
from sklearn.metrics.pairwise import cosine_similarity


class RetrieverLike(Protocol):
    """A protocol that defines the interface for a tree retriever."""

    def retrieve(  # type: ignore
        self, query: str, tree: Tree, *args, **kwargs
    ) -> tuple[list[Node], str]:
        """Retrieve relevant nodes from a tree given a query.

        Args:
            query: The query to retrieve nodes for.
            tree: The tree to retrieve nodes from.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The retrieved nodes.
            The concatenated text of the retrieved nodes.
        """
        ...


class TreeRetriever:
    """A tree retriever that retrieves relevant nodes from a tree given a query.

    It implements the [RetrieverLike][bookacle.tree.retriever.RetrieverLike] protocol.

    Attributes:
        config (TreeRetrieverConfig): The configuration for the retriever.
    """

    def __init__(self, config: TreeRetrieverConfig) -> None:
        """Initialize the tree retriever with the given configuration.

        Args:
            config: The configuration for the retriever.
        """
        self.config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def get_relevant_node_indices(
        self, target_embedding: list[float], candidate_nodes: list[Node]
    ) -> npt.NDArray[np.int64]:
        """Get the relevant node indices given a target embedding and candidate nodes.

        Nodes are selected in the following manner:
            - The cosine similarity between the target embedding and the candidate node embeddings is computed.
            - The cosine similarities are sorted in descending order.
            - If the selection mode is
              [SelectionMode.TOP_K][bookacle.tree.config.SelectionMode.TOP_K], the top `k` nodes are selected.
            - If the selection mode is
              [SelectionMode.THRESHOLD][bookacle.tree.config.SelectionMode.THRESHOLD],
              the nodes with cosine similarity greater than the threshold are selected.

        Args:
            target_embedding: The target embedding to compare against.
            candidate_nodes: The candidate nodes to compare against.

        Returns:
            The relevant node indices.
        """
        embeddings = [node.embeddings for node in candidate_nodes]
        distances = cosine_similarity([target_embedding], embeddings)  # type: ignore
        nearest_neighbors_indices = np.argsort(distances).reshape(-1)[::-1]

        if self.config.selection_mode == SelectionMode.THRESHOLD:
            return nearest_neighbors_indices[
                distances[nearest_neighbors_indices] > self.config.threshold
            ]

        return nearest_neighbors_indices[: self.config.top_k].reshape(-1)

    def get_nodes_within_context(self, candidate_nodes: list[Node]) -> list[Node]:
        """Filter candidate nodes to those that fit within the maximum token length.

        Args:
            candidate_nodes: The candidate nodes to filter.

        Returns:
            The filtered candidate nodes.
        """
        token_lengths = np.array(
            [len(self.config.tokenizer.encode(node.text)) for node in candidate_nodes]
        )
        cumulative_token_lengths = np.cumsum(token_lengths)

        valid_indices = np.where(cumulative_token_lengths <= self.config.max_tokens)[0]

        return [candidate_nodes[i] for i in valid_indices]

    def retrieve_collapse(
        self,
        query: str,
        tree: Tree,
    ) -> tuple[list[Node], str]:
        """Retrieve relevant nodes from a tree given a query using the collapsed tree strategy.

        For more details about the collapsed tree strategy, refer to the RAPTOR paper: https://arxiv.org/pdf/2401.18059.

        The retrieved nodes are selected in the following manner:
            - The query is embedded using the embedding model.
            - The candidate nodes are all the nodes in the tree.
            - Relevant nodes are retrieved using
              [get_relevant_node_indices()][bookacle.tree.retriever.TreeRetriever.get_relevant_node_indices].
            - The nodes are filtered to fit within the maximum token length using
              [get_nodes_within_context()][bookacle.tree.retriever.TreeRetriever.get_nodes_within_context].

        Args:
            query: The query to retrieve nodes for.
            tree: The tree to retrieve nodes from.

        Returns:
            The retrieved nodes.
            The concatenated text of the retrieved nodes.
        """
        query_embedding = self.config.embedding_model.embed(text=query)

        relevant_node_indices = self.get_relevant_node_indices(
            target_embedding=query_embedding, candidate_nodes=tree.tolist()
        )

        candidate_nodes = [tree.get_node(index) for index in relevant_node_indices]

        selected_nodes = self.get_nodes_within_context(candidate_nodes=candidate_nodes)

        return selected_nodes, concatenate_node_texts(nodes=selected_nodes)

    def retrieve_no_collapse(
        self, query: str, tree: Tree, start_layer: int, end_layer: int
    ) -> tuple[list[Node], str]:
        """Retrieve relevant nodes from a tree given a query using the tree traversal strategy.

        For more details about the collapsed tree strategy, refer to the RAPTOR paper: https://arxiv.org/pdf/2401.18059.

        The retrieved nodes are selected in the following manner:
            - The query is embedded using the embedding model.
            - The candidate nodes are all the nodes in the tree from the start layer to the end layer.
            - For each layer
                - Relevant nodes are retrieved using
                  [get_relevant_node_indices()][bookacle.tree.retriever.TreeRetriever.get_relevant_node_indices].
                - The process is repeated on the children of the relevant nodes until the end layer is reached
                  or no more children are available.
            - The nodes are filtered to fit within the maximum token length using

        Args:
            query: The query to retrieve nodes for.
            tree: The tree to retrieve nodes from.
            start_layer: The layer to start retrieving nodes from.
            end_layer: The layer to stop retrieving nodes at.

        Returns:
            The retrieved nodes.
            The concatenated text of the retrieved nodes.
        """
        query_embedding = self.config.embedding_model.embed(text=query)

        candidate_nodes: list[Node] = []

        current_layer = start_layer
        current_nodes = tree.fetch_layer(layer=start_layer)

        added_nodes: set[int] = set()

        while current_layer >= end_layer and current_nodes:
            relevant_node_indices = self.get_relevant_node_indices(
                target_embedding=query_embedding, candidate_nodes=current_nodes
            )
            relevant_nodes = [
                current_nodes[i]
                for i in relevant_node_indices
                if current_nodes[i].index not in added_nodes
            ]

            candidate_nodes.extend(relevant_nodes)
            added_nodes.update({node.index for node in relevant_nodes})

            next_level_nodes: set[int] = set()
            for node in relevant_nodes:
                next_level_nodes.update(node.children)

            if not next_level_nodes:
                break

            current_nodes = [tree.get_node(index) for index in next_level_nodes]
            current_layer -= 1

        selected_nodes = self.get_nodes_within_context(candidate_nodes=candidate_nodes)

        return selected_nodes, concatenate_node_texts(nodes=selected_nodes)

    def retrieve(
        self,
        query: str,
        tree: Tree,
        start_layer: int | None = None,
        end_layer: int | None = None,
        collapse_tree: bool = True,
    ) -> tuple[list[Node], str]:
        """Retrieve relevant nodes from a tree given a query.

        Args:
            query: The query to retrieve nodes for.
            tree: The tree to retrieve nodes from.
            start_layer: The layer to start retrieving nodes from. When `None`, the root layer is used.
            end_layer: The layer to stop retrieving nodes at. When `None`, the leaf layer is used.
            collapse_tree: Whether to use the collapsed tree strategy.

        Returns:
            The retrieved nodes.
            The concatenated text of the retrieved nodes
        """
        if collapse_tree is True:
            return self.retrieve_collapse(query=query, tree=tree)

        if start_layer is None:
            start_layer = tree.top_layer

        if end_layer is None:
            end_layer = 0

        return self.retrieve_no_collapse(
            query=query, tree=tree, start_layer=start_layer, end_layer=end_layer
        )
