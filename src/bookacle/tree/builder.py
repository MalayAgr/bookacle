from __future__ import annotations

import copy
from typing import Any, Protocol

from bookacle.document import Document
from bookacle.tree.config import ClusterTreeConfig
from bookacle.tree.structures import Node, Tree, concatenate_node_texts


class TreeBuilderLike(Protocol):
    """A protocol that defines the interface for a RAPTOR tree builder."""

    def build_from_documents(  # type: ignore
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        *args,
        **kwargs,
    ) -> Tree:
        """Build a tree from a list of documents.

        Args:
            documents: A list of documents to build the tree from.
            chunk_size: The size of the chunks to split the documents into.
            chunk_overlap: The overlap between the chunks.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tree built from the documents.
        """
        ...


class ClusterTreeBuilder:
    """A RAPTOR tree builder that clusters nodes at each subsequent tree layer to build the tree.

    It implements the [TreeBuilderLike][bookacle.tree.builder.TreeBuilderLike] protocol.

    Attributes:
        config (RaptorTreeConfig): The configuration for the tree builder.
    """

    def __init__(self, config: ClusterTreeConfig):
        """Initialize the tree builder with the given configuration.

        Args:
            config: The configuration for the tree builder.
        """
        self.config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def create_leaf_nodes(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> dict[int, Node]:
        """Create leaf nodes from the given chunks.

        Args:
            chunks: The chunks to create the leaf nodes from.
            embeddings: The embeddings of the chunks.

        Returns:
            A mapping of the global index to the created leaf nodes.
        """
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

    def create_next_tree_level(
        self, clusters: list[list[Node]], first_node_index: int, layer: int
    ) -> dict[int, Node]:
        """Create the next tree level from the given clusters.

        For each cluster:
            - The texts of the nodes in the cluster are concatenated.
            - The concatenated text is summarized.
            - The summarized text is embedded.
            - A [Node][bookacle.tree.structures.Node] is created with the summarized text, embeddings, and the indices of the children nodes.

        Args:
            clusters: The clusters to create the next tree level from.
            first_node_index: The global index of the first node in the new layer.
            layer: The layer of the tree the clusters belong to.

        Returns:
            A mapping of the global indices to the created nodes.
        """
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
        chunks: list[Document],
        embeddings: list[list[float]],
        reduction_dimension: int = 10,
    ) -> Tree:
        """Construct a RAPTOR tree from the given chunks and embeddings.

        The tree is built in a bottom-up manner, starting from the leaf nodes and going up to the root nodes.

        To build the tree:
            - The leaf nodes are created from the chunks and embeddings.
            - The leaf nodes are clustered to create the next tree level using
            [create_next_tree_level()][bookacle.tree.builder.ClusterTreeBuilder.create_next_tree_level].
            - The process is repeated until the maximum number of layers is reached
            or the number of nodes in the next level is less than the reduction dimension.

        Args:
            chunks: The chunks to construct the tree from.
            embeddings: The embeddings of the chunks.
            reduction_dimension: The dimension to reduce the embeddings to before clustering.

        Returns:
            A RAPTOR tree constructed from the chunks and embeddings.
        """
        leaf_nodes = self.create_leaf_nodes(chunks=chunks, embeddings=embeddings)

        layer_to_nodes = {0: list(leaf_nodes.values())}

        current_level_nodes = leaf_nodes
        all_tree_nodes = leaf_nodes

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

            new_level_nodes = self.create_next_tree_level(
                clusters=clusters, first_node_index=len(all_tree_nodes), layer=layer + 1
            )

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

        return Tree(
            all_nodes=all_tree_nodes,
            root_nodes=current_level_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=num_layers,
            layer_to_nodes=layer_to_nodes,
        )

    def build_from_documents(
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Tree:
        """Build a RAPTOR tree from the given documents.

        Each document is split into chunks and each chunk is embedded.
        These are then passed to the [construct_tree()][bookacle.tree.builder.ClusterTreeBuilder.construct_tree] method to build the tree.

        Args:
            documents: The documents to build the tree from.
            chunk_size: The size of the chunks to split the documents into.
                        When `None`, it defaults to the maximum length supported by the embedding model.
            chunk_overlap: The overlap between the chunks. When `None`, it defaults to half the chunk size.

        Returns:
            A RAPTOR tree built from the documents.
        """
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

        return self.construct_tree(chunks=splitted_documents, embeddings=embeddings)
