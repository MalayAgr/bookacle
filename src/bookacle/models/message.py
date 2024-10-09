"""This module defines data structures for representing messages exchanged in a conversation with a language model (LLM)."""

from typing import Literal, TypedDict


class Message(TypedDict):
    """A [TypedDict][typing.TypedDict] that represents a message in a conversation with an LLM.

    Attributes:
        role: The role of the message sender.
        content: The content of the message.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str
