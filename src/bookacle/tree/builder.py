from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from bookacle.tree.config import RaptorTreeConfig, SelectionMode
from bookacle.tree.structures import Node, Tree
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class RaptorTreeBuilder:
    def __init__(self, config: RaptorTreeConfig):
        self.config = config

    def get_relevant_nodes(
        self, current_node: Node, list_nodes: list[Node]
    ) -> list[Node]:
        embeddings = [node.embeddings for node in list_nodes]
        distances = cosine_similarity([current_node.embeddings], embeddings)  # type: ignore
        nearest_neighbors_indices = np.argsort(distances)

        if self.config.selection_mode == SelectionMode.THRESHOLD:
            return [
                list_nodes[i]
                for i in nearest_neighbors_indices
                if distances[i] > self.config.threshold
            ]

        return [list_nodes[i] for i in nearest_neighbors_indices[: self.config.top_k]]

    def create_leaf_nodes(self, chunks: list[str]) -> dict[int, Node]:
        with tqdm(total=len(chunks), desc="Creating leaf nodes", unit="node") as pbar:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                tasks = {
                    executor.submit(
                        Node.from_text, index, chunk, self.config.embedding_model
                    ): (index, chunk)
                    for index, chunk in enumerate(chunks)
                }

                leaf_nodes = {}
                for future in as_completed(tasks):
                    node = future.result()
                    leaf_nodes[node.index] = node
                    pbar.update(1)

        return leaf_nodes

    def build_from_documents(
        self,
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Tree:
        if chunk_size is None:
            chunk_size = self.config.embedding_model.model_max_length

        if chunk_overlap is None:
            chunk_overlap = int(chunk_size * 0.1)

        splitted_documents = self.config.document_splitter(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = [doc.page_content for doc in splitted_documents]
        leaf_nodes = self.create_leaf_nodes(chunks=chunks)
