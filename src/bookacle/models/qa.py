from collections.abc import Iterable, Iterator
from typing import Literal, Protocol, TypedDict, overload

import ollama


class Message(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class QAModelLike(Protocol):
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
        stream: bool = False,
        **kwargs,
    ) -> Message | Iterator[Message]: ...


class OllamaQAModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"

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
