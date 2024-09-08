from typing import Protocol


class TokenizerLike(Protocol):
    def encode(self, *args, **kwargs) -> list[int]: ...  # type: ignore
