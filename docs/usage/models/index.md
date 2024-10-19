# Models

## Supported Models

`bookacle` supports three kinds of models:

- Embedding Models - These models are used to embed text into a vector.
- Summarization Models - These models are used to summarize text.
- Question Answering Models - These models are used for question-answering on PDF documents.

`bookacle` comes with default implementations for all of the models, making it easy to use. All implementations are local out of the box - you do not need an OpenAI key to use `bookacle` (just good hardware :wink:).

Custom models can be implemented easily by implementing the corresponding protocols:

- [`EmbeddingModelLike`][bookacle.models.embedding.EmbeddingModelLike] - Protocol for all embedding models.
- [`SummarizationModelLike`][bookacle.models.summarization.SummarizationModelLike] - Protocol for all summarization models.
- [`QAModelLike`][bookacle.models.qa.QAModelLike] - Protocol for all question-answering models.

By implementing these protocols, `bookacle` can practically support any model available in the market.

## Quick Start

You can get started by using any of the default implementations.

### Use an embedding model from `sentence-transformers`

`bookacle` supports any model from the `sentence-transformers` library via the [`SentenceTransformerEmbeddingModel`][bookacle.models.embedding.SentenceTransformerEmbeddingModel] class.

You can embed a list of texts:

```python exec="true" source="material-block" result="python"
from bookacle.models.embedding import SentenceTransformerEmbeddingModel

embedding_model = SentenceTransformerEmbeddingModel(model_name="all-MiniLM-L6-v2")
texts = ["This is a test", "This is another text"]
embeddings = embedding_model.embed(texts)
print(embeddings)
```

You can also embed a single text:

```python exec="true" source="material-block" result="python"
from bookacle.models.embedding import SentenceTransformerEmbeddingModel

embedding_model = SentenceTransformerEmbeddingModel(model_name="all-MiniLM-L6-v2")
text = "This is a test"
embeddings = embedding_model.embed(text)
print(embeddings)
```

It is also possible to use a GPU for inference:

```python exec="true" source="material-block" result="python"
from bookacle.models.embedding import SentenceTransformerEmbeddingModel

embedding_model = SentenceTransformerEmbeddingModel(model_name="all-MiniLM-L6-v2", use_gpu=True)
text = "This is a test"
embeddings = embedding_model.embed(text)
print(embeddings)
```

## Use a summarization model from HuggingFace
