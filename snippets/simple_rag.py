from bookacle.loaders import pymupdf_loader
from bookacle.models.embedding import SentenceTransformerEmbeddingModel
from bookacle.models.message import Message
from bookacle.models.qa import OllamaQAModel
from bookacle.models.summarization import HuggingFaceLLMSummarizationModel
from bookacle.splitters import HuggingFaceTextSplitter
from bookacle.tree.builder import ClusterTreeBuilder
from bookacle.tree.config import ClusterTreeConfig, TreeRetrieverConfig
from bookacle.tree.retriever import TreeRetriever

documents = pymupdf_loader(file_path="data/the-godfather.pdf")

embedding_model = SentenceTransformerEmbeddingModel(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

summarization_model = HuggingFaceLLMSummarizationModel(
    model_name="Qwen/Qwen2-0.5B-Instruct",
    summarization_length=100,
)

qa_model = OllamaQAModel(model_name="qwen2.5:0.5b-instruct")

document_splitter = HuggingFaceTextSplitter(tokenizer=embedding_model.tokenizer)

config = ClusterTreeConfig(
    embedding_model=embedding_model,
    summarization_model=summarization_model,
    document_splitter=document_splitter,
)

tree_builder = ClusterTreeBuilder(config=config)

tree = tree_builder.build_from_documents(documents=documents)

retriever_config = TreeRetrieverConfig(embedding_model=embedding_model)
retriever = TreeRetriever(config=retriever_config)

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
