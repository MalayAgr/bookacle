from dataclasses import dataclass


@dataclass
class Node:
    text: str
    index: int
    children: set[int]
    embeddings: list[float]
    metadata: dict[str, str] | None = None


@dataclass
class Tree:
    all_nodes: dict[int, Node]
    root_nodes: dict[int, Node]
    leaf_nodes: dict[int, Node]
    num_layers: int
    layer_to_nodes: dict[int, set[int]]
