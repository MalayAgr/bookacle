from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import PreTrainedTokenizerBase


class EmbeddingModelLike(Protocol):
    @property
    def tokenizer(self) -> Any: ...

    @property
    def model_max_length(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class SummarizationModelLike(Protocol):
    @property
    def tokenizer(self) -> Any: ...

    def summarize(self, text: str, max_tokens: int = 100) -> str: ...


class HuggingFaceEmbeddingModel:
    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            multi_process=True,
            model_kwargs={"device": "cpu" if use_gpu is False else "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.client.tokenizer

    @property
    def model_max_length(self) -> int:
        return self.model.client.tokenizer.model_max_length

    def embed(self, text: str) -> list[float]:
        return self.model.embed_query(text)


class HuggingFaceSummarizationModel:
    SYSTEM_PROMPT = """As a professional summarizer, create a concise and comprehensive summary of the provided text, while adhering to these guidelines:

        1. Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.

        2. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.

        3. Rely strictly on the provided text, without including external information.

        4. Format the summary in paragraph form for easy understanding.

        5. Use proper punctuation, grammar, and spelling to ensure a polished and professional summary.
    """

    CHAT_TEMPLATE = """
    {% if not add_generation_prompt is defined %}
        {% set add_generation_prompt = false %}
    {% endif %}

    {% for message in messages %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {% endfor %}
    {% if add_generation_prompt %}
        {{ '<|im_start|>assistant\n' }}
    {% endif %}
    """

    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.model = self._create_model(model_name=model_name, use_gpu=use_gpu)

    def _create_model(self, model_name: str, use_gpu: bool = False):
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="summarization",
            device=None if use_gpu is False else 0,
            pipeline_kwargs=dict(
                do_sample=False,
                repetition_penalty=1.03,
            ),
        )

        chat_model = ChatHuggingFace(llm=llm)
        chat_model.tokenizer.chat_template = self.CHAT_TEMPLATE

        return chat_model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    def summarize(self, text: str, max_tokens: int = 100) -> str:
        messages: list[BaseMessage] = [
            SystemMessage(content=f"{self.SYSTEM_PROMPT}"),
            HumanMessage(content=f"Summarize the text below:\n<text>\n{text}\n</text>"),
        ]

        ai_msg = self.model.invoke(messages, max_new_tokens=max_tokens)

        assert isinstance(ai_msg.content, str)

        return ai_msg.content


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    embedding_model = HuggingFaceEmbeddingModel(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    embeddings = embedding_model.embed("This is a test")
    print(embeddings)
    print(len(embeddings))

    text = """
    Hugging Face: Revolutionizing Natural Language Processing

    Introduction:

    In the rapidly evolving field of Natural Language Processing (NLP), Hugging Face has emerged as a prominent and innovative force. This article will explore the story and significance of Hugging Face, a company that has made remarkable contributions to NLP and AI as a whole. From its inception to its role in democratizing AI, Hugging Face has left an indelible mark on the industry.

    The Birth of Hugging Face:

    Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. The name "Hugging Face" was chosen to reflect the company's mission of making AI models more accessible and friendly to humans, much like a comforting hug. Initially, they began as a chatbot company but later shifted their focus to NLP, driven by their belief in the transformative potential of this technology.

    Transformative Innovations:
    Hugging Face is best known for its open-source contributions, particularly the "Transformers" library. This library has become the de facto standard for NLP and enables researchers, developers, and organizations to easily access and utilize state-of-the-art pre-trained language models, such as BERT, GPT-3, and more. These models have countless applications, from chatbots and virtual assistants to language translation and sentiment analysis.

    Key Contributions:
    1. **Transformers Library:** The Transformers library provides a unified interface for more than 50 pre-trained models, simplifying the development of NLP applications. It allows users to fine-tune these models for specific tasks, making it accessible to a wider audience.
    2. **Model Hub:** Hugging Face's Model Hub is a treasure trove of pre-trained models, making it simple for anyone to access, experiment with, and fine-tune models. Researchers and developers around the world can collaborate and share their models through this platform.
    """

    summary_model = HuggingFaceSummarizationModel(model_name="facebook/bart-large-cnn")
    result = summary_model.summarize(text, max_tokens=300)
    print(result)
