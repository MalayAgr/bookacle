from typing import Iterable, Protocol, overload, runtime_checkable

from bookacle.models.message import Message
from bookacle.tokenizer import TokenizerLike
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    pipeline,
)


@runtime_checkable
class SummarizationModelLike(Protocol):
    """A protocol that defines the methods and attributes that a summarization model should implement."""

    @property
    def tokenizer(self) -> TokenizerLike:
        """
        Returns:
            The tokenizer used by the model.
        """
        ...

    @overload
    def summarize(self, text: list[str]) -> list[str]:
        """Summarize a list of input texts.

        Args:
            text: The list of input texts to summarize.

        Returns:
            The summaries of the input texts.
        """
        ...

    @overload
    def summarize(self, text: str) -> str:
        """Summarize a single input text.

        Args:
            text: The input text to summarize.

        Returns:
            The summary of the input text.
        """
        ...

    def summarize(self, text: str | list[str]) -> str | list[str]:
        """Summarize the input text or list of texts.

        Args:
            text: The input text or list of input texts to summarize.

        Returns:
            The summary of the input text or list of texts.
        """
        ...


class HuggingFaceSummarizationModel:
    """A class that uses a Hugging Face model for summarization.

    It implements the [SummarizationModelLike][bookacle.models.summarization.SummarizationModelLike] protocol.

    Attributes:
        model_name (str): The name of the Hugging Face model to use.
        summarization_length (int): The maximum length of the summary.
        use_gpu (bool): Whether to use the GPU for inference.
        model (AutoModelForSeq2SeqLM): The Hugging Face model for summarization.
        pipeline (transformers.Pipeline): The Hugging Face pipeline for summarization.
    """

    def __init__(
        self,
        model_name: str,
        summarization_length: int = 100,
        *,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the summarization model.

        Args:
            model_name: The name of the Hugging Face model to use.
            summarization_length: The maximum length of the summary.
            use_gpu: Whether to use the GPU for inference.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.summarization_length = summarization_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            task="summarization",
            model=self.model,
            tokenizer=self._tokenizer,
            device=0 if use_gpu else -1,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_name={self.model_name}, "
            f"summarization_length={self.summarization_length}, use_gpu={self.use_gpu})"
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Returns:
            The Hugging Face tokenizer used by the underlying model.
        """
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


class HuggingFaceLLMSummarizationModel:
    """A class that uses a Hugging Face LLM for summarization.

    It implements the [SummarizationModelLike][bookacle.models.summarization.SummarizationModelLike] protocol.

    Attributes:
        model_name (str): The name of the Hugging Face LLM to use.
        summarization_length (int): The maximum length of the summary.
        system_prompt (str): The system prompt passed to the LLM for summarization.
        use_gpu (bool): Whether to use the GPU for inference.
        model (AutoModelForCausalLM): The Hugging Face LLM for summarization.
        pipeline (transformers.Pipeline): The Hugging Face pipeline for summarization.
    """

    def __init__(
        self,
        model_name: str,
        summarization_length: int = 100,
        *,
        system_prompt: str = "",
        use_gpu: bool = False,
    ) -> None:
        """Initialize the summarization model.

        Args:
            model_name: The name of the Hugging Face model to use.
            summarization_length: The maximum length of the summary.
            system_prompt: The system prompt to use for LLM for summarization.
            use_gpu: Whether to use the GPU for inference.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.system_prompt = system_prompt
        self.summarization_length = summarization_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self._tokenizer,
            device=0 if use_gpu else -1,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Returns:
            The Hugging Face tokenizer used by the underlying model.
        """
        return self._tokenizer

    @overload
    def format_as_chat_message(self, text: list[str]) -> list[list[Message]]:
        """Format a list of texts as chat messages.

        A chat message is a dictionary with the keys 'role' and 'content'.

        For each text:
        - If the system prompt is provided, a list of lists containing the system prompt and user message is returned.
        - If the system prompt is not provided, a list of user messages is returned.

        Args:
            text: The list of texts to format.

        Returns:
            The formatted chat messages.
        """
        ...

    @overload
    def format_as_chat_message(self, text: str) -> list[Message]:
        """Format a list of texts as chat messages.

        A chat message is a dictionary with the keys 'role' and 'content'.

        For the text:
        - If the system prompt is provided, a list containing the system prompt and user message is returned.
        - If the system prompt is not provided, a list containing only the user message is returned.

        Args:
            text: The list of texts to format.

        Returns:
            The formatted chat messages.
        """
        ...

    def format_as_chat_message(
        self, text: str | list[str]
    ) -> list[Message] | list[list[Message]]:
        """Format the input text or list of texts as chat messages.

        A chat message is a dictionary with the keys 'role' and 'content'.

        If the input is a list of texts:
            - If the system prompt is provided, a list of lists containing the system prompt and user message is returned.
            - If the system prompt is not provided, a list of user messages is returned.

        If the input is a single text:
            - If the system prompt is provided, a list containing the system prompt and user message is returned.
            - If the system prompt is not provided, a list containing only the user message is returned.

        Args:
            text: The input text or list of texts to format.

        Returns:
            The formatted chat messages.

        Example:
            ```python exec="true" source="material-block"" result="python" title="Single Text"
            from bookacle.models.summarization import HuggingFaceLLMSummarizationModel
            model = HuggingFaceLLMSummarizationModel(model_name="Qwen/Qwen2-0.5B-Instruct")
            text = "This is a test"
            print(model.format_as_chat_message(text))
            ```

            ```python exec="true" source="material-block"" result="python" title="Mutliple Texts"
            from bookacle.models.summarization import HuggingFaceLLMSummarizationModel
            model = HuggingFaceLLMSummarizationModel(model_name="Qwen/Qwen2-0.5B-Instruct")
            text = ["This is a test", "This is another test"]
            print(model.format_as_chat_message(text))
            ```
        """
        system_message = None

        if self.system_prompt:
            system_message = Message(role="system", content=self.system_prompt)

        user_prompt = "Summarize the following in not more than {summarization_length} words:\n{text}"

        if isinstance(text, str):
            user_message = Message(
                role="user",
                content=user_prompt.format(
                    summarization_length=self.summarization_length, text=text
                ),
            )

            if system_message is not None:
                return [system_message, user_message]

            return [user_message]

        user_messages = (
            Message(
                role="user",
                content=user_prompt.format(
                    summarization_length=self.summarization_length, text=t
                ),
            )
            for t in text
        )

        if system_message is not None:
            return [[system_message, user_message] for user_message in user_messages]

        return [[user_message] for user_message in user_messages]

    @overload
    def summarize(self, text: list[str]) -> list[str]: ...

    @overload
    def summarize(self, text: str) -> str: ...

    def summarize(self, text: str | list[str]) -> str | list[str]:
        """Summarize the input text or list of texts.

        The input is first formatted into chat messages using
        [format_as_chat_message][bookacle.models.summarization.HuggingFaceLLMSummarizationModel.format_as_chat_message]
        and then passed to the underlying LLM for summarization.

        Args:
            text: The input text or list of texts to summarize.

        Returns:
            The summary of the input text or list of texts.
        """
        messages = self.format_as_chat_message(text)

        summaries = self.pipeline(
            messages,
            do_sample=True,
            max_new_tokens=self.summarization_length,
            return_full_text=False,
        )

        if isinstance(text, str):
            return summaries[0][0]["generated_text"]  # type: ignore

        return [summary[0]["generated_text"] for summary in summaries]  # type: ignore


if __name__ == "__main__":
    from bookacle.conf import settings

    text = [
        """Hugging Face: Revolutionizing Natural Language Processing

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

    summary_model = HuggingFaceLLMSummarizationModel(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        use_gpu=False,
        summarization_length=100,
        system_prompt=settings.SUMMARIZATION_SYSTEM_PROMPT,
    )
    result = summary_model.summarize(text)
    print(result)
