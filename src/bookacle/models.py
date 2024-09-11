from typing import Protocol, overload

from bookacle.tokenizer import TokenizerLike
from langchain import prompts
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
    def tokenizer(self) -> TokenizerLike: ...

    @property
    def model_max_length(self) -> int: ...

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]: ...


class SummarizationModelLike(Protocol):
    @property
    def tokenizer(self) -> TokenizerLike: ...

    @overload
    def summarize(self, text: list[str]) -> list[str]: ...

    @overload
    def summarize(self, text: str) -> str: ...

    def summarize(self, text: str | list[str]) -> str | list[str]: ...


class HuggingFaceEmbeddingModel:
    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            multi_process=True,
            model_kwargs={"device": "cpu" if use_gpu is False else "cuda"},
            encode_kwargs={"normalize_embeddings": True, "show_progress_bar": True},
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.client.tokenizer

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return self.model.embed_query(text)

        return self.model.embed_documents(texts=text)


class HuggingFaceSummarizationModel:
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

    def __init__(
        self,
        model_name: str,
        summarization_length: int = 100,
        *,
        task: str = "summarization",
        use_gpu: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model = self._create_model(
            model_name=model_name,
            summarization_length=summarization_length,
            task=task,
            use_gpu=use_gpu,
        )

    def _create_model(
        self,
        model_name: str,
        summarization_length: int = 100,
        task: str = "summarization",
        use_gpu: bool = False,
    ) -> ChatHuggingFace:
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task=task,
            device=None if use_gpu is False else 0,
            pipeline_kwargs={
                "do_sample": True,
                "repetition_penalty": 1.03,
                "max_new_tokens": summarization_length,
            },
        )

        chat_model = ChatHuggingFace(llm=llm)

        chat_model.tokenizer.chat_template = self.CHAT_TEMPLATE

        return chat_model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    @overload
    def summarize(self, text: list[str]) -> list[str]: ...

    @overload
    def summarize(self, text: str) -> str: ...

    def summarize(self, text: str | list[str]) -> str | list[str]:
        if isinstance(text, str):
            messages: list[BaseMessage] = [
                HumanMessage(
                    content=f"Summarize the text below:\n<text>\n{text}\n</text>"
                ),
            ]

            ai_messages = self.model.invoke(messages)

            assert isinstance(ai_messages.content, str)

            return ai_messages.content

        return self._batched_summarize(texts=text)

    def _batched_summarize(self, texts: list[str]) -> list[str]:
        prompt = prompts.ChatPromptTemplate.from_messages(
            [
                ("human", "Summarize the text below:\n<text>\n{text}\n</text>"),
            ]
        )

        chain = prompt | self.model

        ai_messages = chain.batch([{"text": text} for text in texts])

        return [message.content for message in ai_messages]  # type: ignore


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # embedding_model = HuggingFaceEmbeddingModel(
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )
    # embeddings = embedding_model.embed("This is a test")
    # print(embeddings)
    # print(len(embeddings))

    text = [
        """
    Hugging Face: Revolutionizing Natural Language Processing

    Introduction:

    In the rapidly evolving field of Natural Language Processing (NLP), Hugging Face has emerged as a prominent and innovative force. This article will explore the story and significance of Hugging Face, a company that has made remarkable contributions to NLP and AI as a whole. From its inception to its role in democratizing AI, Hugging Face has left an indelible mark on the industry.""",
        """The Birth of Hugging Face:

    Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. The name "Hugging Face" was chosen to reflect the company's mission of making AI models more accessible and friendly to humans, much like a comforting hug. Initially, they began as a chatbot company but later shifted their focus to NLP, driven by their belief in the transformative potential of this technology.""",
        """Transformative Innovations:
    Hugging Face is best known for its open-source contributions, particularly the "Transformers" library. This library has become the de facto standard for NLP and enables researchers, developers, and organizations to easily access and utilize state-of-the-art pre-trained language models, such as BERT, GPT-3, and more. These models have countless applications, from chatbots and virtual assistants to language translation and sentiment analysis.

    Key Contributions:
    1. **Transformers Library:** The Transformers library provides a unified interface for more than 50 pre-trained models, simplifying the development of NLP applications. It allows users to fine-tune these models for specific tasks, making it accessible to a wider audience.
    2. **Model Hub:** Hugging Face's Model Hub is a treasure trove of pre-trained models, making it simple for anyone to access, experiment with, and fine-tune models. Researchers and developers around the world can collaborate and share their models through this platform.
    """,
    ]

    summary_model = HuggingFaceSummarizationModel(
        model_name="facebook/bart-large-cnn",
        use_gpu=False,
        summarization_length=100,
    )
    result = summary_model.summarize(text)
    print(result)
