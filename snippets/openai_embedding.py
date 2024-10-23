from typing import overload

import tiktoken
from openai import OpenAI
from tiktoken.core import Encoding


class OpenAIEmbeddingModel:
    def __init__(
        self, model_name: str, tokenizer_name: str = "", dimensions: int = 256
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.tokenizer_name = tokenizer_name

        self._client = OpenAI()
        # If the tokenizer is not provided, automatically fetch it for the model
        self._tokenizer = (
            tiktoken.encoding_for_model(model_name)
            if not tokenizer_name
            else tiktoken.get_encoding(tokenizer_name)
        )
        # See: https://platform.openai.com/docs/guides/embeddings/embedding-models
        self._model_max_length = 8191

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, dimensions={self.dimensions!r}, tokenizer_name={self.tokenizer_name!r})"

    @property
    def tokenizer(self) -> Encoding:
        return self._tokenizer

    @property
    def model_max_length(self) -> int:
        return self._model_max_length

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        response = self._client.embeddings.create(
            input=[text] if isinstance(text, str) else text,
            model=self.model_name,
            dimensions=self.dimensions,
            encoding_format="float",
        )

        if isinstance(text, str):
            return response.data[0].embedding

        return [item.embedding for item in response.data]
