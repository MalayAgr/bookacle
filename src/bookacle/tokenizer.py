from typing import Protocol


class TokenizerLike(Protocol):
    """A protocol that all tokenizers should follow."""

    def encode(self, *args, **kwargs) -> list[int]:  # type: ignore
        """Tokenize the input text into a list of integers.

        Returns:
            Tokenized input.
        """
        ...
