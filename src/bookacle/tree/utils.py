from threading import Lock

from bookacle.models import EmbeddingModelLike, SummarizationModelLike
from bookacle.tree.structures import Node


def concatenate_node_texts(nodes: list[Node]) -> str:
    return "\n\n".join(" ".join(node.text.splitlines()) for node in nodes) + "\n\n"


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
        text=summary,
        embedding_model=embedding_model,
        children_indices={node.index for node in cluster},
    )
