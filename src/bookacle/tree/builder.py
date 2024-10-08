from __future__ import annotations

import copy
from typing import Any, Protocol

from bookacle.document import Document
from bookacle.tree.config import RaptorTreeConfig
from bookacle.tree.structures import Node, Tree, concatenate_node_texts


class TreeBuilderLike(Protocol):
    def build_from_documents(  # type: ignore
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *args,
        **kwargs,
    ) -> Tree: ...


class ClusterTreeBuilder:
    def __init__(self, config: RaptorTreeConfig):
        self.config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def create_leaf_nodes(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> dict[int, Node]:
        return {
            index: Node(
                text=chunk["page_content"],
                index=index,
                children=set(),
                embeddings=embeddings[index],
                layer=0,
                metadata=chunk.get("metadata"),
            )
            for index, chunk in enumerate(chunks)
        }

    def _create_next_tree_level(
        self, clusters: list[list[Node]], first_node_index: int, layer: int
    ) -> dict[int, Node]:
        cluster_texts = [concatenate_node_texts(cluster) for cluster in clusters]
        summaries = self.config.summarization_model.summarize(text=cluster_texts)
        embeddings = self.config.embedding_model.embed(text=summaries)

        return {
            first_node_index
            + index: Node(
                text=summary,
                index=index,
                children={node.index for node in cluster},
                embeddings=embeddings[index],
                layer=layer,
            )
            for index, (cluster, summary) in enumerate(zip(clusters, summaries))
        }

    def construct_tree(
        self,
        current_level_nodes: dict[int, Node],
        all_tree_nodes: dict[int, Node],
        layer_to_nodes: dict[int, list[Node]],
        reduction_dimension: int = 10,
    ) -> tuple[dict[int, Node], int]:
        num_layers = self.config.max_num_layers

        for layer in range(self.config.max_num_layers):
            if len(current_level_nodes) <= reduction_dimension + 1:
                num_layers = len(layer_to_nodes)
                break

            clusters = self.config.clustering_func(
                nodes=list(current_level_nodes.values()),
                tokenizer=self.config.embedding_tokenizer,
                clustering_backend=self.config.clustering_backend,
                max_length_in_cluster=self.config.max_length_in_cluster,
                reduction_dimension=reduction_dimension,
            )

            new_level_nodes = self._create_next_tree_level(
                clusters=clusters, first_node_index=len(all_tree_nodes), layer=layer + 1
            )

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return current_level_nodes, num_layers

    def build_from_documents(
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Tree:
        if chunk_size is None:
            chunk_size = self.config.embedding_model.model_max_length

        if chunk_overlap is None:
            chunk_overlap = int(chunk_size * 0.5)

        splitted_documents = self.config.document_splitter(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        embeddings = self.config.embedding_model.embed(
            text=[doc["page_content"] for doc in splitted_documents]
        )
        leaf_nodes = self.create_leaf_nodes(
            chunks=splitted_documents, embeddings=embeddings
        )

        layer_to_nodes = {0: list(leaf_nodes.values())}

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes, num_layers = self.construct_tree(
            current_level_nodes=leaf_nodes,
            all_tree_nodes=all_nodes,
            layer_to_nodes=layer_to_nodes,
        )

        return Tree(
            all_nodes=all_nodes,
            root_nodes=root_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=num_layers,
            layer_to_nodes=layer_to_nodes,
        )
