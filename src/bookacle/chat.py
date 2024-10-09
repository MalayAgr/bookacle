"""Chat module for conversing with the AI model."""

from collections.abc import Iterator
from pathlib import Path

from bookacle.models.message import Message
from bookacle.models.qa import QAModelLike
from bookacle.tree.retriever import RetrieverLike
from bookacle.tree.structures import Tree
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner


class Chat:
    """A terminal-based chat interface for interacting with a RAPTOR RAG-based LLM.

    Attributes:
        retriever (RetrieverLike): Retriever to use for retrieving relevant context.
        qa_model (QAModelLike): QA model to use for answering questions.
        console (Console): Rich Console to use for displaying messages.
        history_file (str): File to store chat history.
        user_avatar (str): Avatar to use for the user in the chat UI.
    """

    def __init__(
        self,
        retriever: RetrieverLike,
        qa_model: QAModelLike,
        console: Console,
        history_file: str = ".bookacle-chat-history.txt",
        user_avatar: str = "👤",
    ) -> None:
        """Initialize the chat interface.

        Args:
            retriever: Retriever to use for retrieving relevant context.
            qa_model: QA model to use for answering questions.
            console: Rich Console to use for displaying messages.
            history_file: File to store chat history. The file is created in the home directory.
            user_avatar: Avatar to use for the user in the chat UI.
        """
        self.retriever = retriever
        self.qa_model = qa_model
        self.console = console
        self.history_file = Path().home() / history_file
        self.user_avatar = user_avatar

    def display_ai_msg_stream(self, messages: Iterator[Message]) -> str:
        """Display an AI message stream in the chat UI.

        Args:
            messages: Stream of AI messages to display.

        Returns:
            The complete message as a string.
        """
        complete_message = ""
        panel = Columns(
            [
                Panel(Markdown(complete_message), title="🤖", title_align="left"),
                Spinner("aesthetic"),
            ]
        )

        with Live(panel, refresh_per_second=15, console=self.console) as live:
            for chunk in messages:
                complete_message += chunk["content"]
                updated_panel = Columns(
                    [
                        Panel(Markdown(complete_message), title="🤖", title_align="left"),
                        Spinner("aesthetic"),
                    ]
                )
                live.update(updated_panel)

            live.update(Panel(Markdown(complete_message), title="🤖", title_align="left"))

        return complete_message

    def invoke_qa_model(  # type: ignore
        self,
        tree: Tree,
        question: str,
        history: list[Message] | None = None,
        stream: bool = True,
        *args,
        **kwargs,
    ) -> Message:
        """Invoke the QA model to answer a question.

        Args:
            tree: RAPTOR tree that should be used for RAG.
            question: The question to answer.
            history: Chat history.
            stream: Whether to stream the AI response.
            **args (tuple[Any]): Additional positional arguments to pass to the retriever.
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the retriever.

        Returns:
            The response from the QA model.
        """
        retriever = self.retriever
        qa_model = self.qa_model

        _, context = retriever.retrieve(question, tree, *args, **kwargs)

        if stream is True:
            responses = qa_model.answer(
                question=question, context=context, history=history, stream=True
            )

            complete_message = self.display_ai_msg_stream(messages=responses)

            return Message(role="assistant", content=complete_message)

        response = qa_model.answer(
            question=question, context=context, history=history, stream=False
        )

        with Live(
            Spinner("aesthetic"),
            refresh_per_second=15,
            console=self.console,
            transient=True,
        ):
            panel = Panel(Markdown(response["content"]), title="🤖", title_align="left")
            self.console.print(panel)

        return response

    def run(  # type: ignore
        self,
        tree: Tree,
        initial_chat_message: str = "",
        system_prompt: str = "",
        stream: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Run the chat interface.

        Args:
            tree: RAPTOR tree that should be used for RAG.
            initial_chat_message: Initial message to display in the chat.
            system_prompt: System prompt that should be used for the QA model.
            stream: Whether to stream the AI response.
            *args (tuple[Any]): Additional positional arguments to pass to the retriever.
            **kwargs (dict[str, Any]): Additional keyword arguments to pass to the retriever.
        """
        self.console.clear()

        user_history = FileHistory(filename=self.history_file)
        session: PromptSession[str] = PromptSession(
            history=user_history, erase_when_done=True
        )

        messages: list[Message] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if initial_chat_message:
            self.console.print(f"{self.user_avatar} {initial_chat_message}")
            user_history.append_string(initial_chat_message)

        message_counter = 0

        while True:
            user_text: str = session.prompt(
                f"{self.user_avatar}: ", auto_suggest=AutoSuggestFromHistory()
            )

            if not user_text.strip():
                continue

            user_history.append_string(user_text)

            self.console.print(f"{self.user_avatar}: {user_text}")
            messages.append(Message(role="user", content=user_text))

            self.console.print("")

            qa_response = self.invoke_qa_model(
                tree,
                user_text,
                messages,
                stream,
                *args,
                **kwargs,
            )

            messages.append(qa_response)

            self.console.print("")

            message_counter += 1
