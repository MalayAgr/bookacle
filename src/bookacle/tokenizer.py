from typing import Protocol


class TokenizerLike(Protocol):
    """A protocol that defines the methods that a tokenizer should implement."""

    def encode(self, *args, **kwargs) -> list[int]:  # type: ignore
        """Encode the input into a list of integers."""
        ...
