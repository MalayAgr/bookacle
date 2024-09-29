from __future__ import annotations

import importlib
import os
from typing import Any, Type, TypeVar

from bookacle.models.embedding import EmbeddingModelLike
from bookacle.models.summarization import SummarizationModelLike
from bookacle.splitter import DocumentSplitterLike
from bookacle.tree.config import SelectionMode
from bookacle.tree.retriever import RetrieverLike
from dynaconf import Dynaconf, LazySettings, ValidationError, Validator

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

T = TypeVar("T")


def _import_attribute_from_module(dotted_path: str) -> Type[Any]:
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _cast_class_path_to_instance(class_path: str, arguments: dict[str, Any]) -> object:
    cls = _import_attribute_from_module(class_path)
    return cls(**arguments)


def _cast_document_splitter(value: dict[str, Any]) -> DocumentSplitterLike:
    class_path = value["splitter_class"]
    cls = _import_attribute_from_module(class_path)
    arguments: dict[str, Any] = value["splitter_arguments"]

    if (tokenizer_from := arguments.get("tokenizer_from")) is not None:
        if not isinstance(tokenizer_from, (EmbeddingModelLike, SummarizationModelLike)):
            raise ValidationError("Invalid tokenizer_from.")

        arguments.pop("tokenizer_from")
        arguments["tokenizer"] = tokenizer_from.tokenizer

    return cls(**arguments)


def _cast_retriever_config(value: dict[str, Any]) -> RetrieverLike:
    class_path = value["config_class"]
    cls = _import_attribute_from_module(class_path)
    arguments = value["config_arguments"]

    if (selection_mode := arguments.get("selection_mode")) is not None:
        if selection_mode not in (el.value for el in SelectionMode):
            raise ValidationError("Invalid selection mode.")

        arguments["selection_mode"] = SelectionMode(arguments["selection_mode"])

    return cls(**arguments)


settings: LazySettings = Dynaconf(
    envvar_prefix="BOOKACLE",
    root_path=ROOT_PATH,
    settings_files=["settings.toml", "prompts.toml"],
)

settings.validators.register(
    Validator(
        "custom_loaders_dir",
        "clustering_func",
        "stream_output",
        "embedding_model.model_class",
        "embedding_model.model_arguments",
        "summarization_model.model_class",
        "summarization_model.model_arguments",
        "document_splitter.splitter_class",
        "document_splitter.splitter_arguments",
        "retriever.retriever_class",
        "retriever.retriever_arguments",
        "clustering_backend.backend_class",
        "clustering_backend.backend_arguments",
        "qa_model.model_class",
        "qa_model.model_arguments",
        must_exist=True,
    ),
    Validator(
        "embedding_model",
        cast=lambda value: _cast_class_path_to_instance(
            value["model_class"], value["model_arguments"]
        ),
    ),
    Validator(
        "summarization_model",
        cast=lambda value: _cast_class_path_to_instance(
            class_path=value["model_class"],
            arguments=value["model_arguments"],
        ),
    ),
    Validator("document_splitter", cast=_cast_document_splitter),
    Validator("retriever_config", cast=_cast_retriever_config),
    Validator(
        "retriever",
        cast=lambda value: _cast_class_path_to_instance(
            class_path=value["retriever_class"],
            arguments=value["retriever_arguments"],
        ),
    ),
    Validator("clustering_func", cast=_import_attribute_from_module),
    Validator(
        "clustering_backend",
        cast=lambda value: _cast_class_path_to_instance(
            class_path=value["backend_class"],
            arguments=value["backend_arguments"],
        ),
    ),
    Validator(
        "tree_builder_config",
        cast=lambda value: _cast_class_path_to_instance(
            class_path=value["config_class"], arguments=value["config_arguments"]
        ),
    ),
    Validator(
        "tree_builder",
        cast=lambda value: _cast_class_path_to_instance(
            class_path=value["builder_class"],
            arguments=value["builder_arguments"],
        ),
    ),
    Validator(
        "qa_model",
        cast=lambda value: _cast_class_path_to_instance(
            class_path=value["model_class"], arguments=value["model_arguments"]
        ),
    ),
    Validator("chunk_size", default=None),
    Validator("chunk_overlap", default=None),
)
