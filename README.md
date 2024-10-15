# bookacle

[![documentation](https://img.shields.io/badge/docs-bookacle-blue?style=flat
)](https://malayagr.github.io/bookacle)

Answer queries on complex PDF documents using RAPTOR-based RAG.

## RAPTOR Overview

RAPTOR (**R**ecursive **A**bstractive **P**rocessing for **T**ree-**O**rganized **R**etrieval) is a RAG technique designed to work with large documents in a limited context. From the abstract:

> _Retrieval-augmented language models can better adapt to changes in world state and incorporate long-tail knowledge. However, most existing methods retrieve only short contiguous chunks from a retrieval corpus, limiting holistic understanding of the overall document context. We introduce the novel approach of recursively embedding, clustering, and summarizing chunks of text, constructing a tree with differing levels of summarization from the bottom up. At inference time, our RAPTOR model retrieves from this tree, integrating information across lengthy documents at different levels of abstraction. Controlled experiments show that retrieval with recursive summaries offers significant improvements over traditional retrieval-augmented LMs on several tasks. On question-answering tasks that involve complex, multi-step reasoning, we show state-of-the-art results; for example, by coupling RAPTOR retrieval with the use of GPT-4, we can improve the best performance on the QuALITY benchmark by 20% in absolute accuracy._

It builds a hierarchial tree structure on the documents and queries the tree to retrieve relevant context. The idea is that the upper layers of the tree represent a more holistic understanding of the documents and as we go down the tree, the understanding becomes more granular until we reach the leaf nodes, where the actual text from the documents resides.

This holistic understanding is achieved as follows:

- The documents are split into small chunks.
- These chunks are embedded using an embedding model and become the leaf nodes of the tree structure.
- A clustering algorithm is used to cluster these leaf nodes.
- For each cluster:
    - The texts of the nodes in the cluster are concatenated.
    - A summary of the concatenated text is generated using a model.
    - The summary is embedded using the same embedding model and becomes a node in the next layer.
    - The children of this node are the nodes in the cluster.
- This process is repeated until no further layers can be made.

Since each subsequent layer is a summary of some nodes in the previous layer, it represents a holistic understanding of those nodes, helping build a holistic understanding of the overall documents as we go up the tree.

For more details on RAPTOR, refer to the paper: <https://arxiv.org/abs/2401.18059>.

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
