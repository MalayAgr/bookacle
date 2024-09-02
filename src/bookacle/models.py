import os
from functools import cached_property

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from transformers import PreTrainedTokenizerBase


class EmbeddingModel:
    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            multi_process=True,
            model_kwargs={"device": "cpu" if use_gpu is False else "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.tokenizer: PreTrainedTokenizerBase = self.model.client.tokenizer

    def embed(self, text: str) -> list[float]:
        return self.model.embed_query(text)


class SummarizationModel:
    SYSTEM_PROMPT = """As a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:

        1. Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.

        2. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.

        3. Rely strictly on the provided text, without including external information.

        4. Format the summary in paragraph form for easy understanding.
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

    def __init__(
        self, model_name: str, max_tokens: int = 100, *, use_gpu: bool = False
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.model = self._create_model(model_name=model_name, use_gpu=use_gpu)
        self.tokenizer: PreTrainedTokenizerBase = self.model.tokenizer

    def _create_model(self, model_name: str, use_gpu: bool = False):
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="summarization",
            device=None if use_gpu is False else 0,
            pipeline_kwargs=dict(
                max_new_tokens=self.max_tokens,
                do_sample=False,
                repetition_penalty=1.03,
            ),
        )

        chat_model = ChatHuggingFace(llm=llm)
        chat_model.tokenizer.chat_template = self.CHAT_TEMPLATE

        return chat_model

    def summarize(self, text: str):
        messages = [
            SystemMessage(content=f"{self.SYSTEM_PROMPT}"),
            HumanMessage(content=f"Summarize the text below:\n<text>\n{text}\n</text>"),
        ]

        ai_msg = self.model.invoke(messages)

        return ai_msg.content


if __name__ == "__main__":
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = (
        "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"
    )

    embedding_model = EmbeddingModel(
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

    summary_model = SummarizationModel(
        model_name="facebook/bart-large-cnn", max_tokens=300
    )
    result = summary_model.summarize(text)
    print(result)
    print(result)
