from typing import Any, NotRequired, TypedDict


class Document(TypedDict):
    page_content: str
    metadata: NotRequired[dict[str, Any]]
