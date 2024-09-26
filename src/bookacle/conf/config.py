from __future__ import annotations

import importlib
import os
from typing import Any, Type, TypedDict, TypeVar

from bookacle.models.embedding import EmbeddingModelLike
from bookacle.models.summarization import SummarizationModelLike
from bookacle.splitter import DocumentSplitterLike
from bookacle.tree.clustering import ClusteringFunctionLike
from bookacle.tree.config import SelectionMode
from bookacle.tree.retriever import RetrieverLike
from dynaconf import Dynaconf, ValidationError, Validator

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

T = TypeVar("T")


def _import_attribute_from_module(dotted_path: str) -> Type[Any]:
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _cast_class_path_to_instance(class_path: str, arguments: dict[str, Any]):
    cls = _import_attribute_from_module(class_path)
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


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    root_path=ROOT_PATH,
    settings_files=["settings.toml"],
    validate_on_update="all",
    validators=[
        Validator(
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
        Validator(
            "document_splitter",
            cast=lambda value: _cast_class_path_to_instance(
                class_path=value["splitter_class"],
                arguments=value["splitter_arguments"],
            ),
        ),
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
    ],
)
