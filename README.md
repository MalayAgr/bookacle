# bookacle

[![documentation](https://img.shields.io/badge/docs-bookacle-blue?style=flat
)](https://malayagr.github.io/bookacle)

Answer queries on complex PDF documents using RAPTOR-based RAG.

For more details on RAPTOR, refer to the paper: <https://arxiv.org/abs/2401.18059>.

## RAPTOR Overview

## Features

- Everything is a [Protocol](https://typing.readthedocs.io/en/latest/spec/protocol.html), allowing for convenient extensibility.
- Completely local out of the box - no OpenAI key required.
- Sensible default implementations for embeddings models, summarization models and question-answering models. See [Models](https://malayagr.github.io/bookacle/usage/models/) for more details.
- Use custom embedding models, summarization models and question-answering easily - just implement the protocol. See [Models](https://malayagr.github.io/bookacle/usage/models/) for more details.
- Load your PDFs as text or markdown using the provided loaders or implement your own. See [Loaders](https://malayagr.github.io/bookacle/usage/loaders) for more details.
- Use _any_ tokenizer as long as it follows the `TokenizerLike` protocol. See [Tokenizers](https://malayagr.github.io/bookacle/usage/tokenizers/) for more details.
- Split your documents into chunks easily using the provided splitters or implement your own. See [Splitters](https://malayagr.github.io/bookacle/usage/splitters/) for more details.
- Customize the default RAPTOR-tree building methodology by implementing your own clustering logic. See [Clustering](https://malayagr.github.io/bookacle/usage/clustering/) for more details.
- Implement your own RAPTOR-tree building methodology by implementing the `TreeBuilderLike` protocol. See [Building RAPTOR Tree](https://malayagr.github.io/bookacle/usage/building_raptor_tree.md) for more details.
- Use a terminal-based chat to chat with your documents. See [Command-Line Interface](https://malayagr.github.io/bookacle/usage/cli/) for more details.
- Define configuration for `bookacle` using TOML files and use them throughout your application. See [Configuration](https://malayagr.github.io/bookacle/usage/config/) for more details.
