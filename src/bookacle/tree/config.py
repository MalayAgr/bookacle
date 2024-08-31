from dataclasses import dataclass
from enum import Enum, auto


class SelectionMode(Enum):
    TOP_K = auto()
    THRESHOLD = auto()


@dataclass
class RaptorTreeConfig:
    embedding_model: str
    summarization_model: str
    max_tokens: int = 100
    num_layers: int = 5
    threshold: float = 0.5
    top_k: int = 5
    selection_mode: SelectionMode = SelectionMode.TOP_K
    summarization_length: int = 100
    summarization_length: int = 100
