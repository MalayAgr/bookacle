from typing import Protocol

import numpy as np
import numpy.typing as npt
from bookacle.tree.config import SelectionMode, TreeRetrieverConfig
from bookacle.tree.structures import Node, Tree, concatenate_node_texts
from sklearn.metrics.pairwise import cosine_similarity


class RetrieverLike(Protocol):
    def retrieve(self, query: str, *args, **kwargs) -> tuple[list[Node], str]: ...  # type: ignore


class TreeRetriever:
    def __init__(self, config: TreeRetrieverConfig) -> None:
        self.config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def get_relevant_node_indices(
        self, target_embedding: list[float], candidate_nodes: list[Node]
    ) -> npt.NDArray[np.int64]:
        embeddings = [node.embeddings for node in candidate_nodes]
        distances = cosine_similarity([target_embedding], embeddings)  # type: ignore
        nearest_neighbors_indices = np.argsort(distances).reshape(-1)[::-1]

        if self.config.selection_mode == SelectionMode.THRESHOLD:
            return nearest_neighbors_indices[
                distances[nearest_neighbors_indices] > self.config.threshold
            ]

        return nearest_neighbors_indices[: self.config.top_k].reshape(-1)

    def get_nodes_within_context(self, candidate_nodes: list[Node]) -> list[Node]:
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
        if collapse_tree is True:
            return self.retrieve_collapse(query=query, tree=tree)

        if start_layer is None:
            start_layer = tree.top_layer

        if end_layer is None:
            end_layer = 0

        return self.retrieve_no_collapse(
            query=query, tree=tree, start_layer=start_layer, end_layer=end_layer
        )
