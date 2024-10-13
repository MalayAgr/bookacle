# bookacle

[![documentation](https://img.shields.io/badge/docs-bookacle-blue?style=flat
)](https://malayagr.github.io/bookacle)

Answer queries on complex PDF queries using RAPTOR-based RAG.

For more details on RAPTOR, refer to the paper: <https://arxiv.org/abs/2401.18059>.

## RAPTOR Overview

## Features

- Everything is a [Protocol](https://typing.readthedocs.io/en/latest/spec/protocol.html), allowing for convenient extensibility.
- Use custom embedding models, summarization models, question-answering models and many more easily - just implement the protocol.
- Sensible default implementations for all of the above:
    - [SentenceTransformerEmbeddingModel](reference/bookacle/models/embedding/#bookacle.models.embedding.SentenceTransformerEmbeddingModel) - Use any embedding model from the `sentence-transformers` library.
    - [HuggingFaceLLMSummarizationModel](reference/bookacle/models/summarization/#bookacle.models.summarization.HuggingFaceLLMSummarizationModel) - Use an LLM from HuggingFace for summarization.
