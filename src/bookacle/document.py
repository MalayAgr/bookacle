from typing import Any, TypedDict


class Document(TypedDict):
    page_content: str
    metadata: dict[str, Any]
