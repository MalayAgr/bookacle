# Usage

## Example RAG Application

### Full Code

??? Code

    ```python
    from bookacle.loaders import pymupdf_loader
    from bookacle.models.embedding import SentenceTransformerEmbeddingModel
    from bookacle.models.qa import OllamaQAModel
    from bookacle.models.summarization import HuggingFaceLLMSummarizationModel
    from bookacle.splitters import HuggingFaceTextSplitter
    from bookacle.tree.builder import ClusterTreeBuilder
    from bookacle.tree.config import ClusterTreeConfig, TreeRetrieverConfig
    from bookacle.tree.retriever import TreeRetriever
    from bookacle.models.message import Message

    documents = pymupdf_loader(file_path="data/the-godfather.pdf")

    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True
    )

    document_splitter = HuggingFaceTextSplitter(tokenizer=embedding_model.tokenizer)

    summarization_model = HuggingFaceLLMSummarizationModel(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        use_gpu=True,
        summarization_length=100,
    )

    config = ClusterTreeConfig(
        embedding_model=embedding_model,
        summarization_model=summarization_model,
        document_splitter=document_splitter,
    )

    tree_builder = ClusterTreeBuilder(config=config)

    tree = tree_builder.build_from_documents(documents=documents)

    retriever_config = TreeRetrieverConfig(embedding_model=embedding_model)
    retriever = TreeRetriever(config=retriever_config)

    qa_model = OllamaQAModel(model_name="qwen2.5:0.5b-instruct")

    query = "Who are the cast members of The Godfather?"

    _, context = retriever.retrieve(query=query, tree=tree)

    system_prompt = """You are a helpful assistant, designed to help users understand documents and answer questions on the documents.
    Use your knowledge and the context passed to you to answer user queries.
    The context will be text extracted from the document. It will be denoted by CONTEXT: in the prompt.
    The user's query will be denoted by QUERY: in the prompt.
    Do NOT explicitly state that you are referring to the context.
    """

    history = [Message(role="system", content=system_prompt)]

    answer = qa_model.answer(question=query, context=context, stream=False, history=history)

    print(f"Answer:\n{answer['content']}")
    ```

### Step-by-Step

Below, the code is shown step-by-step with intermediate outputs.

??? Imports

    ```python exec="true" source="material-block" session="rag"
    from bookacle.loaders import pymupdf_loader
    from bookacle.models.embedding import SentenceTransformerEmbeddingModel
    from bookacle.models.qa import OllamaQAModel
    from bookacle.models.summarization import HuggingFaceLLMSummarizationModel
    from bookacle.splitters import HuggingFaceTextSplitter
    from bookacle.tree.builder import ClusterTreeBuilder
    from bookacle.tree.config import ClusterTreeConfig, TreeRetrieverConfig
    from bookacle.tree.retriever import TreeRetriever
    from bookacle.models.message import Message
    ```

Load the data file. The example uses the first 2 pages (when exported in A3) of the Wikipedia entry on [The Godfather](https://en.wikipedia.org/wiki/The_Godfather):

```python exec="true" source="material-block" result="python" session="rag"
documents = pymupdf_loader(file_path="data/the-godfather.pdf")

print(f"Number of documents: {len(documents)}")
print(f"First document:\n{documents[0]}")
```

Create the embedding model, splitter to use and the summarization model. The example uses the following embedding and summarization models:

- [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the embedding model.
- [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the summarization model.

```python exec="true" source="material-block" result="python" session="rag"
embedding_model = SentenceTransformerEmbeddingModel(
    model_name="sentence-transformers/all-MiniLM-L6-v2", use_gpu=True
)

document_splitter = HuggingFaceTextSplitter(tokenizer=embedding_model.tokenizer)

summarization_model = HuggingFaceLLMSummarizationModel(
    model_name="Qwen/Qwen2-0.5B-Instruct",
    use_gpu=True,
    summarization_length=100,
)

print(f"Embedding Model: {embedding_model}")
print(f"Summarization Model: {summarization_model}")
```

Create the RAPTOR tree:

```python exec="true" source="material-block" result="python" session="rag"
config = ClusterTreeConfig(
    embedding_model=embedding_model,
    summarization_model=summarization_model,
    document_splitter=document_splitter,
)

tree_builder = ClusterTreeBuilder(config=config)

tree = tree_builder.build_from_documents(documents=documents)

print(f"Tree: {tree}")
```

Initialize the retriever and the question-answering model. The example uses the [`qwen2.5:0.5b-instruct`](https://ollama.com/library/qwen2.5:0.5b-instruct) model from Ollama for question-answering (GPU poor :sob:):

```python exec="true" source="material-block" result="python" session="rag"
retriever_config = TreeRetrieverConfig(embedding_model=embedding_model)
retriever = TreeRetriever(config=retriever_config)

qa_model = OllamaQAModel(model_name="qwen2.5:0.5b-instruct")

print(f"QA Model: {qa_model}")
```

Send a query to retriever to retrieve context:

```python exec="true" source="material-block" result="text" session="rag"
query = "Who are the cast members of The Godfather?"

_, context = retriever.retrieve(query=query, tree=tree)

print(f"Retrieved context:\n{context}")
```

Set the system prompt (optional) and get an answer from the question-answering model:

```python exec="true" source="material-block" result="text" session="rag"
system_prompt = """You are a helpful assistant, designed to help users understand documents and answer questions on the documents.
Use your knowledge and the context passed to you to answer user queries.
The context will be text extracted from the document. It will be denoted by CONTEXT: in the prompt.
The user's query will be denoted by QUERY: in the prompt.
Do NOT explicitly state that you are referring to the context.
"""

history = [Message(role="system", content=system_prompt)]

answer = qa_model.answer(question=query, context=context, stream=False, history=history)

print(f"Answer:\n{answer['content']}")
```
