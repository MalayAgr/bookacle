from threading import Lock

from bookacle.models import (
    EmbeddingModel,
    EmbeddingModelLike,
    SummarizationModel,
    SummarizationModelLike,
)
from bookacle.tree.structures import Node
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizerBase


def concatenate_node_texts(nodes: list[Node]) -> str:
    return "\n\n".join(" ".join(node.text.splitlines()) for node in nodes) + "\n\n"


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
            "\n\n",
            "\n",
            ".",
            "!",
            "?",
        ],
        keep_separator="end",
    )

    return splitter.split_documents(documents=documents)


def create_parent_node(
    cluster: list[Node],
    embedding_model: EmbeddingModelLike,
    summarization_model: SummarizationModelLike,
    # new_level_nodes: dict[int, Node],
    next_node_index: int,
    summarization_length: int,
    # lock: Lock,
) -> Node:
    concatenated_text = concatenate_node_texts(nodes=cluster)

    summary = summarization_model.summarize(
        text=concatenated_text, max_tokens=summarization_length
    )

    return Node.from_text(
        index=next_node_index,
        text=summary,  # type: ignore
        embedding_model=embedding_model,
        children_indices={node.index for node in cluster},
    )
