from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class ImproperlyConfigured(Exception): ...


class LazySettings:
    def __init__(self) -> None:
        self._wrapped: Settings | UserSettingsHolder | None = None

    def _setup(self, name: str | None = None) -> None:
        """
        Load the settings module specified by the environment variable.
        """
        settings_path = os.environ.get(
            "BOOKACLE_SETTINGS_MODULE", "bookacle.conf.defaults"
        )
        if not settings_path:
            raise ImproperlyConfigured(
                f"Requested {name or 'settings'}, but settings are not configured. "
                "You must either define the environment variable "
                "BOOKACLE_SETTINGS_MODULE or call settings.configure() before "
                "accessing settings."
            )

        if settings_path.endswith(".py"):
            # Treat the Python file as a module
            settings_path = self._load_from_file(settings_path)
        else:
            # Load settings from a module
            settings_path = settings_path

        self._wrapped = Settings(settings_path)

    def _load_from_file(self, filename: str) -> str:
        """
        Treat the provided Python file as a module by adding its directory to sys.path and returning the module name.
        """
        filepath = Path(filename)
        module_name = filepath.stem  # Get the filename without the .py extension
        sys.path.insert(0, str(filepath.parent))  # Add the directory to sys.path

        return module_name  # Return the module name to be imported

    def __getattr__(self, name: str) -> Any:
        if self._wrapped is None:
            self._setup(name)

        return getattr(self._wrapped, name)

    def configure(self, default_settings: ModuleType, **options: Any) -> None:
        if self._wrapped is not None:
            raise RuntimeError("Settings are already configured.")

        holder = UserSettingsHolder(default_settings)

        for key, value in options.items():
            if not key.isupper():
                raise TypeError(f"Setting {key} must be uppercase.")

            setattr(holder, key, value)

        self._wrapped = holder

    @property
    def configured(self) -> bool:
        return self._wrapped is not None


class Settings:
    def __init__(self, settings_module: str) -> None:
        """
        Load settings from the specified module and fallback to default settings.
        """
        # Important to ensure no circular imports
        from bookacle.conf import defaults as default_settings

        for setting in dir(default_settings):
            if setting.isupper():
                setattr(self, setting, getattr(default_settings, setting))

        mod = importlib.import_module(settings_module)
        self.SETTINGS_MODULE = settings_module

        for setting in dir(mod):
            if setting.isupper():
                setattr(self, setting, getattr(mod, setting))


class UserSettingsHolder:
    """Holds user-defined settings and defaults to global settings for missing attributes."""

    def __init__(self, default_settings: ModuleType):
        self.__dict__["_deleted"] = set()
        self.default_settings = default_settings

    def __getattr__(self, name: str) -> Any:
        if name.isupper() and name not in self._deleted:
            return getattr(self.default_settings, name)

        raise AttributeError(f"Setting '{name}' not found.")

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.isupper():
            raise TypeError("Settings names must be uppercase.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        self._deleted.add(name)

    def __repr__(self) -> str:
        return f"<UserSettingsHolder for {self.default_settings}>"


settings = LazySettings()
