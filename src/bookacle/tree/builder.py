from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
from bookacle.tree.config import RaptorTreeConfig, SelectionMode
from bookacle.tree.structures import Node, Tree
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


def split_documents(
    documents: list[Document],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 100,
    overlap: int = 0,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=[
            ".",
            "!",
            "?",
            "\n",
            "\n\n",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        keep_separator="end",
    )

    return splitter.split_documents(documents=documents)


def process_cluster(
    cluster: list[Node],
    new_level_nodes: dict[int, Node],
    next_node_index: int,
    lock: Lock,
):
    pass


class RaptorTreeBuilder:
    def __init__(self, config: RaptorTreeConfig):
        self.config = config

    def create_node(
        self, index: int, text: str, children_indices: set[int] | None = None
    ) -> tuple[int, Node]:
        if children_indices is None:
            children_indices = set()

        embeddings = self.config.embedding_model.embed(text)

        return index, Node(
            text=text,
            index=index,
            children=children_indices,
            embeddings=embeddings,
        )

    def summarize(self, text: str):
        return self.config.summarization_model.summarize(text=text)

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
                    executor.submit(self.create_node, index, chunk): (index, chunk)
                    for index, chunk in enumerate(chunks)
                }

                leaf_nodes = {}
                for future in as_completed(tasks):
                    index, node = future.result()
                    leaf_nodes[index] = node
                    pbar.update(1)

        return leaf_nodes

    def build_from_documents(self, documents: list[Document]) -> Tree:
        splitted_documents = split_documents(
            documents, tokenizer=self.config.embedding_tokenizer
        )
        chunks = [doc.page_content for doc in splitted_documents]
        leaf_nodes = self.create_leaf_nodes(chunks=chunks)
