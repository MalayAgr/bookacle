"""
This module manages loading and validating configuration settings for the application from TOML files.

It also provides utility functions for importing classes and instantiating objects based on their dotted paths.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Type, TypeVar

from bookacle.models.embedding import EmbeddingModelLike
from bookacle.models.summarization import SummarizationModelLike
from bookacle.splitters import DocumentSplitterLike
from bookacle.tree.config import SelectionMode
from bookacle.tree.retriever import RetrieverLike
from dynaconf import Dynaconf, LazySettings, ValidationError, Validator

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

T = TypeVar("T")


def import_attribute_from_module(dotted_path: str) -> Type[Any]:
    """
    Imports an attribute (class or function) from a dotted module path.

    Args:
        dotted_path: The full path to the attribute in the form of 'module.submodule.ClassName'.

    Returns:
        The imported attribute (e.g., a class or function).

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute does not exist in the module.
    """
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def cast_class_path_to_instance(class_path: str, arguments: dict[str, Any]) -> object:
    """
    Instantiates a class by its dotted path and passes the provided arguments.

    Args:
        class_path: The full path to the class in the form 'module.submodule.ClassName'.
        arguments: A dictionary of arguments to pass to the class constructor.

    Returns:
       An instance of the class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    cls = import_attribute_from_module(class_path)
    return cls(**arguments)


def cast_document_splitter(value: dict[str, Any]) -> DocumentSplitterLike:
    """
    Casts a dictionary containing document splitter configuration into an instance of DocumentSplitterLike.

    Args:
        value: Dictionary containing the class path of the document splitter and its arguments.

    Returns:
        An instance of the document splitter class.

    Raises:
        ValidationError: If the arguments has the `tokenizer_from` key
                         and it doesn't resolve to an implementation of
                         [EmbeddingModelLike][bookacle.models.embedding.EmbeddingModelLike] or
                         [SummarizationModelLike][bookacle.models.summarization.SummarizationModelLike].
    """
    class_path = value["splitter_class"]
    cls = import_attribute_from_module(class_path)
    arguments: dict[str, Any] = value["splitter_arguments"]

    if (tokenizer_from := arguments.get("tokenizer_from")) is not None:
        if not isinstance(tokenizer_from, (EmbeddingModelLike, SummarizationModelLike)):
            raise ValidationError("Invalid tokenizer_from.")

        arguments.pop("tokenizer_from")
        arguments["tokenizer"] = tokenizer_from.tokenizer

    return cls(**arguments)


def cast_retriever_config(value: dict[str, Any]) -> RetrieverLike:
    """
    Casts a dictionary containing retriever configuration into an instance of RetrieverLike.

    Args:
        value: Dictionary with the class path of the retriever and its arguments.

    Returns:
        An instance of the retriever class.

    Raises:
        ValidationError: If the arguments has the `selection_mode` key
                         and it is not a member of the [SelectionMode][bookacle.tree.config.SelectionMode]
                         Enum.
    """
    class_path = value["config_class"]
    cls = import_attribute_from_module(class_path)
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
"""Settings for the application.

The following validators are registered on the settings object.

```python
from dynaconf import Validator

[
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
    cast=lambda value: cast_class_path_to_instance(
        value["model_class"], value["model_arguments"]
    ),
),
Validator(
    "summarization_model",
    cast=lambda value: cast_class_path_to_instance(
        class_path=value["model_class"],
        arguments=value["model_arguments"],
    ),
),
Validator("document_splitter", cast=cast_document_splitter),
Validator("retriever_config", cast=cast_retriever_config),
Validator(
    "retriever",
    cast=lambda value: cast_class_path_to_instance(
        class_path=value["retriever_class"],
        arguments=value["retriever_arguments"],
    ),
),
Validator("clustering_func", cast=import_attribute_from_module),
Validator(
    "clustering_backend",
    cast=lambda value: cast_class_path_to_instance(
        class_path=value["backend_class"],
        arguments=value["backend_arguments"],
    ),
),
Validator(
    "tree_builder_config",
    cast=lambda value: cast_class_path_to_instance(
        class_path=value["config_class"], arguments=value["config_arguments"]
    ),
),
Validator(
    "tree_builder",
    cast=lambda value: cast_class_path_to_instance(
        class_path=value["builder_class"],
        arguments=value["builder_arguments"],
    ),
),
Validator(
    "qa_model",
    cast=lambda value: cast_class_path_to_instance(
        class_path=value["model_class"], arguments=value["model_arguments"]
    ),
),
Validator("chunk_size", default=None),
Validator("chunk_overlap", default=None),
]
```

Note that the validators are not applied automatically since there is overhead
(for example, in casting the embedding models and summarization models).

To apply the validators, call `settings.validators.validate()`.

For more details, see the Dynaconf documentation on validators: https://www.dynaconf.com/validation/.

The default settings are loaded from the `settings.toml` and `prompts.toml`.

- `settings.toml` contains the main configuration settings.
- `prompts.toml` contains the prompts for the user interface.

The settings can be accessed as a dictionary or as attributes. For example:

```python exec="true" source="material-block" result="python"
from bookacle.conf import settings
# Validate the settings
settings.validators.validate()
# Access as attribute
print(f"Default embedding model: {settings.EMBEDDING_MODEL}")
# Access as dictionary
print(f"Default QA model: {settings['qa_model']}")
```

For more details on managing settings, see [Configuration][].
"""

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
        cast=lambda value: cast_class_path_to_instance(
            value["model_class"], value["model_arguments"]
        ),
    ),
    Validator(
        "summarization_model",
        cast=lambda value: cast_class_path_to_instance(
            class_path=value["model_class"],
            arguments=value["model_arguments"],
        ),
    ),
    Validator("document_splitter", cast=cast_document_splitter),
    Validator("retriever_config", cast=cast_retriever_config),
    Validator(
        "retriever",
        cast=lambda value: cast_class_path_to_instance(
            class_path=value["retriever_class"],
            arguments=value["retriever_arguments"],
        ),
    ),
    Validator("clustering_func", cast=import_attribute_from_module),
    Validator(
        "clustering_backend",
        cast=lambda value: cast_class_path_to_instance(
            class_path=value["backend_class"],
            arguments=value["backend_arguments"],
        ),
    ),
    Validator(
        "tree_builder_config",
        cast=lambda value: cast_class_path_to_instance(
            class_path=value["config_class"], arguments=value["config_arguments"]
        ),
    ),
    Validator(
        "tree_builder",
        cast=lambda value: cast_class_path_to_instance(
            class_path=value["builder_class"],
            arguments=value["builder_arguments"],
        ),
    ),
    Validator(
        "qa_model",
        cast=lambda value: cast_class_path_to_instance(
            class_path=value["model_class"], arguments=value["model_arguments"]
        ),
    ),
    Validator("chunk_size", default=None),
    Validator("chunk_overlap", default=None),
)
