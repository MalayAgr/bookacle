from typing import Protocol, overload

import numpy as np
from bookacle.tokenizer import TokenizerLike
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    pipeline,
)


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


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str, *, use_gpu: bool = False) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name_or_path=model_name, device="cuda" if use_gpu is True else "cpu"
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self.model.tokenizer

    @property
    def model_max_length(self) -> int:
        return self.model.max_seq_length

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        embeddings = self.model.encode(text, normalize_embeddings=True)

        assert isinstance(embeddings, np.ndarray)

        return embeddings.tolist()


class HuggingFaceSummarizationModel:
    def __init__(
        self,
        model_name: str,
        summarization_length: int = 100,
        *,
        use_gpu: bool = False,
    ) -> None:
        self.model_name = model_name
        self.summarization_length = summarization_length

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=512
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            task="summarization",
            model=self.model,
            tokenizer=self._tokenizer,
            device=0 if use_gpu else -1,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @overload
    def summarize(self, text: list[str]) -> list[str]: ...

    @overload
    def summarize(self, text: str) -> str: ...

    def summarize(self, text: str | list[str]) -> str | list[str]:
        summaries = self.pipeline(
            text, min_length=5, max_length=self.summarization_length, truncation=True
        )

        if isinstance(text, str):
            return summaries[0]["summary_text"]  # type: ignore

        return [summary["summary_text"] for summary in summaries]  # type: ignore


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    embeddings = embedding_model.embed(["This is a test", "This is another test"])
    print(embeddings)
    print(len(embeddings))

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
