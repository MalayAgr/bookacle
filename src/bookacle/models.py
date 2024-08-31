import os
from functools import cached_property

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed(self, text: str) -> list[float]:
        return self.model.embed_query(text)


class SummarizationModel:
    SYSTEM_PROMPT = """ As a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:

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

    def __init__(self, model_name: str, max_tokens: int = 100) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens

    @cached_property
    def model(self) -> ChatHuggingFace:
        llm = HuggingFacePipeline.from_model_id(
            model_id=self.model_name,
            task="summarization",
            pipeline_kwargs=dict(
                max_new_tokens=512,
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
