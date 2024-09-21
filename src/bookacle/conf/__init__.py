from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

from bookacle.conf import defaults as default_settings


class ImproperlyConfigured(Exception): ...


class LazySettings:
    def __init__(self) -> None:
        self._wrapped: Settings | UserSettingsHolder | None = None

    def _setup(self, name: str | None = None) -> None:
        """
        Load the settings module specified by the environment variable.
        """
        settings_module = os.environ.get(
            "BOOKACLE_SETTINGS_MODULE", "bookacle.conf.defaults"
        )
        if not settings_module:
            raise ImproperlyConfigured(
                f"Requested {name or 'settings'}, but settings are not configured. "
                "You must either define the environment variable "
                "BOOKACLE_SETTINGS_MODULE or call settings.configure() before "
                "accessing settings."
            )

        self._wrapped = Settings(settings_module)

    def __getattr__(self, name: str) -> Any:
        if self._wrapped is None:
            self._setup(name)

        return getattr(self._wrapped, name)

    def configure(self, default_settings: ModuleType, **options):
        if self._wrapped is not None:
            raise RuntimeError("Settings are already configured.")

        holder = UserSettingsHolder(default_settings)

        for key, value in options.items():
            if not key.isupper():
                raise TypeError(f"Setting {key} must be uppercase.")

            setattr(holder, key, value)

        self._wrapped = holder

    @property
    def configured(self):
        return self._wrapped is not None


class Settings:
    def __init__(self, settings_module: str):
        """
        Load settings from the specified module and fallback to default settings.
        """
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

    def __getattr__(self, name):
        if name.isupper() and name not in self._deleted:
            return getattr(self.default_settings, name)
        raise AttributeError(f"Setting '{name}' not found.")

    def __setattr__(self, name: str, value):
        if not name.isupper():
            raise TypeError("Settings names must be uppercase.")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        self._deleted.add(name)

    def __repr__(self):
        return f"<UserSettingsHolder for {self.default_settings}>"


settings = LazySettings()
