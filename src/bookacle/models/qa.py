"""This module defines a protocol and an implementation for a Question-Answering (QA) model that processes questions
and returns answers with or without streaming capabilities."""

from collections.abc import Iterable, Iterator
from typing import Literal, Protocol, overload

import ollama
from bookacle.models.message import Message


class QAModelLike(Protocol):
    """A protocol that defines the methods that a QA model should implement."""

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[Message] | None = None,
        *args,
        stream: Literal[True] = True,
        **kwargs,
    ) -> Iterator[Message]:
        """Answer a question with streaming given a context and chat history."""
        ...

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[Message] | None = None,
        *args,
        stream: Literal[False] = False,
        **kwargs,
    ) -> Message:
        """Answer a question without streaming given a context and chat history."""
        ...

    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[Message] | None = None,
        *args,
        stream: bool = False,
        **kwargs,
    ) -> Message | Iterator[Message]:
        """Answer a question given a context and chat history with or without streaming.

        Args:
            question: The question to answer.
            context: The context for the question.
            history: The chat history.
            stream: Whether to stream the AI response.

        Returns:
            A single message from the QA model or a stream of messages.
        """
        ...


class OllamaQAModel:
    """A QA model that uses the [Ollama](https://ollama.com/) library.

    It implements the [QAModelLike][bookacle.models.qa.QAModelLike] protocol.

    Attributes:
        model_name (str): The name of the model to use.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the QA model.

        Args:
            model_name: The name of the model to use.
        """
        self.model_name = model_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[Message] | None = None,
        *args,
        stream: Literal[True] = True,
        **kwargs,
    ) -> Iterator[Message]: ...

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[Message] | None = None,
        *args,
        stream: Literal[False] = False,
        **kwargs,
    ) -> Message: ...

    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[Message] | None = None,
        *args,
        stream: bool = True,
        **kwargs,
    ) -> Message | Iterable[Message]:
        """Answer a question given a context and chat history with or without streaming.

        The question and the context are combined into a single message
        for the QA model, using the following template:

        ```html
        CONTEXT:
        <context>

        QUERY: <question>
        ```

        After combining the question and context, the message is appended to the history (if any)
        and sent to the QA model.

        A system prompt can be provided by adding it as the first message in the history.
        """
        user_message_with_context: Message = {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUERY: {question}",
        }

        messages = (
            [user_message_with_context]
            if history is None
            else history + [user_message_with_context]
        )

        if stream is True:
            chunks = ollama.chat(model=self.model_name, messages=messages, stream=True)  # type: ignore
            return (chunk["message"] for chunk in chunks)

        response = ollama.chat(model=self.model_name, messages=messages, stream=False)  # type: ignore
        return response["message"]


if __name__ == "__main__":
    qa_model: QAModelLike = OllamaQAModel(model_name="qwen2:0.5b")
    history: list[Message] = [
        {"role": "user", "content": "Tell me about the Eiffel Tower."},
        {
            "role": "assistant",
            "content": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        },
        {"role": "user", "content": "What is the height of the Eiffel Tower?"},
        {
            "role": "assistant",
            "content": "The height of the Eiffel Tower is approximately 330 meters.",
        },
    ]

    chunks = qa_model.answer(
        question="What is the capital of France?",
        context="France is a country in Europe.",
        history=history,
        stream=True,
    )

    for chunk in chunks:
        print(chunk, end="", flush=True)
