# Models

## Supported Models

`bookacle` supports three kinds of models:

- Embedding Models - These models are used to embed text into a vector.
- Summarization Models - These models are used to summarize text.
- Question Answering Models - These models are used for question-answering on PDF documents.

`bookacle` comes with built-in implementations for all of the models, making it easy to use. All implementations are local out of the box - you do not need an OpenAI key to use `bookacle` (just good hardware :wink:).

Custom models can be implemented easily by implementing the corresponding protocols. By implementing these protocols, `bookacle` can practically support any model available in the market.

## Working with Embedding Models

### The `EmbeddingModelLike` protocol

All embedding models in `bookacle` need to implement the [`EmbeddingModelLike`][bookacle.models.embedding.EmbeddingModelLike] protocol.

This is how the protocol is defined:

```python
class EmbeddingModelLike(Protocol):
    """A protocol that defines the methods and attributes that an embedding model should implement."""

    @property
    def tokenizer(self) -> TokenizerLike:
        """
        Returns:
            The tokenizer used by the model.
        """
        ...

    @property
    def model_max_length(self) -> int:
        """
        Returns:
            The maximum length of the input that the model can accept.
        """
        ...

    @overload
    def embed(self, text: str) -> list[float]:
        """Embed a single input text.

        Args:
            text: The input text to embed.

        Returns:
            The embeddings of the input text.
        """
        ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]:
        """Embed a list of input texts.

        Args:
            text: The list of input texts to embed.

        Returns:
            The embeddings of the input texts.
        """
        ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Embed the input text or list of texts.

        Args:
            text: The input text or list of input texts to embed.

        Returns:
            The embeddings of the input text or list of texts.
        """
        ...
```

### Use an embedding model from `sentence-transformers`

`bookacle` supports any model from the [`sentence-transformers`](https://sbert.net/) library via the [`SentenceTransformerEmbeddingModel`][bookacle.models.embedding.SentenceTransformerEmbeddingModel] class. Of course, it implements the [`EmbeddingLikeProtocol`][bookacle.models.embedding.EmbeddingModelLike].

You can embed a list of texts:

```python exec="true" source="material-block" result="python" session="embeddings"
from bookacle.models.embedding import SentenceTransformerEmbeddingModel

embedding_model = SentenceTransformerEmbeddingModel(model_name="all-MiniLM-L6-v2")
texts = ["This is a test", "This is another text"]
embeddings = embedding_model.embed(texts)
print(embeddings)
```

You can also embed a single text:

```python exec="true" source="material-block" result="python" session="embeddings"
text = "This is a test"
embeddings = embedding_model.embed(text)
print(embeddings)
```

To access the underlying tokenizer used by the model, you can use the `tokenizer` property:

```python exec="true" source="material-block" result="python" session="embeddings"
print(embedding_model.tokenizer)
```

You can also check the maximum length of a sequence supported by the model:

```python exec="true" source="material-block" result="python" session="embeddings"
print(embedding_model.model_max_length)
```

???+ note "GPU Inference"

    [`SentenceTransformerEmbeddingModel`][bookacle.models.embedding.SentenceTransformerEmbeddingModel] supports GPU inference. To use a GPU, set the `use_gpu` attribute to `True` when creating the embedding model.

### Implement a custom embedding model

Below is an example of how to implement a custom embedding model. We are going to implement an embedding model which uses [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) by creating a class called `OpenAIEmbeddingModel`. It will implement the [`EmbeddingModelLike`][bookacle.models.embedding.EmbeddingModelLike] protocol:

??? info "Dependencies"

    This example requires the following packages:

    - [`openai`](https://github.com/openai/openai-python)
    - [`tiktoken`](https://github.com/openai/tiktoken)

    === "Using `pip`"

        ```console
        $ python -m pip install openai tiktoken
        ```

    === "Using `uv`"

        ```console
        $ uv add openai tiktoken
        ```

```python
--8<-- "snippets/openai_embedding.py"
```

Example usage:

```python
embedding_model = OpenAIEmbeddingModel(model_name="text-embedding-3-small")
text = "This is a test"
embeddings = embedding_model.embed(text)
print(embeddings)
```

## Summarization Models

### Use a summarization model from HuggingFace

### Use an LLM from HuggingFace for summarization

### Implement a custom summarization model

## Question Answering Models

### Use a model from Ollama

### Implement a custom question-answering model
