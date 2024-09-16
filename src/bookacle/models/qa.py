from collections.abc import Iterable, Iterator
from typing import Literal, Protocol, overload

import ollama


class QAModelLike(Protocol):
    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        *args,
        stream: Literal[True] = True,
        **kwargs,
    ) -> Iterator[str]: ...

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        *args,
        stream: Literal[False] = False,
        **kwargs,
    ) -> str: ...

    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        *args,
        stream: bool = False,
        **kwargs,
    ) -> str | Iterator[str]: ...


class OllamaQAModel:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[ollama.Message] | None = None,
        *args,
        stream: Literal[True] = True,
        **kwargs,
    ) -> Iterator[str]: ...

    @overload
    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[ollama.Message] | None = None,
        *args,
        stream: Literal[False] = False,
        **kwargs,
    ) -> str: ...

    def answer(  # type: ignore
        self,
        question: str,
        context: str,
        history: list[ollama.Message] | None = None,
        *args,
        stream: bool = True,
        **kwargs,
    ) -> str | Iterable[str]:
        user_message_with_context: ollama.Message = {
            "role": "user",
            "content": f"CONTEXT: {context}\n\n{question}",
        }

        messages = (
            [user_message_with_context]
            if history is None
            else history + [user_message_with_context]
        )

        if stream is True:
            chunks = ollama.chat(model=self.model_name, messages=messages, stream=True)
            return (chunk["message"]["content"] for chunk in chunks)

        response = ollama.chat(model=self.model_name, messages=messages, stream=False)
        return response["message"]["content"]


if __name__ == "__main__":
    qa_model: QAModelLike = OllamaQAModel(model_name="qwen2:0.5b")
    non_stream_answer = qa_model.answer(
        question="What is the capital of France?",
        context="France is a country in Europe.",
        stream=True,
    )

    print(non_stream_answer)