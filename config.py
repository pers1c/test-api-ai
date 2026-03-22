from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from models import AppConfig, AuthConfig


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# Список файлов конфигурации по умолчанию (ищутся в указанном порядке)
_DEFAULT_CONFIG_CANDIDATES = ["config.yaml"]


def _resolve_env_vars(value: Any) -> Any:
    """Рекурсивно заменяет паттерны ${ИМЯ_ПЕРЕМЕННОЙ} в строках внутри словарей и списков."""
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            resolved = os.environ.get(var_name)
            if resolved is None:
                raise ValueError(
                    f"Переменная окружения '{var_name}' указана в конфиге, но не задана. "
                    f"Задайте её в окружении shell или в файле .env."
                )
            return resolved
        return _ENV_VAR_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Загружает конфигурацию приложения из YAML-файла с поддержкой переменных окружения.

    Приоритет разрешения (от высшего к низшему):
      1. Переменные окружения, заданные в shell
      2. Переменные из файла .env в текущей директории
      3. Значения из config.yaml
      4. Значения по умолчанию из Pydantic-моделей AppConfig

    Аргументы:
        config_path: Путь к YAML-файлу конфига. Если None, выполняется поиск файлов по умолчанию.

    Возвращает:
        AppConfig: Полностью разрешённая конфигурация приложения.

    Исключения:
        ValueError: Если указанная переменная окружения не задана.
        FileNotFoundError: Если явно указанный config_path не существует.
    """
    load_dotenv(override=False)  # Загружаем .env в os.environ (не перезаписываем существующие переменные)

    raw: dict = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        for candidate in _DEFAULT_CONFIG_CANDIDATES:
            if Path(candidate).exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                break

    resolved = _resolve_env_vars(raw)
    return AppConfig.model_validate(resolved)


def build_auth_headers(config: AppConfig) -> dict[str, str]:
    """
    Формирует HTTP-заголовки аутентификации на основе конфигурации.

    Аргументы:
        config: Загруженная конфигурация приложения.

    Возвращает:
        Словарь заголовок -> значение для добавления в каждый запрос.
        Возвращает пустой словарь, если тип аутентификации 'none' или токен не задан.
    """
    auth: AuthConfig = config.auth

    if auth.type == "none" or not auth.token:
        return {}

    if auth.type == "bearer":
        return {"Authorization": f"Bearer {auth.token}"}

    if auth.type == "api_key":
        header_name = auth.header_name or "X-API-Key"
        return {header_name: auth.token}

    return {}


def build_auth_query_params(config: AppConfig) -> dict[str, str]:
    """
    Формирует query-параметры аутентификации для API, использующих ключ в строке запроса.

    Аргументы:
        config: Загруженная конфигурация приложения.

    Возвращает:
        Словарь имя_параметра -> значение, или пустой словарь.
    """
    auth: AuthConfig = config.auth
    if auth.type == "api_key" and auth.query_param_name and auth.token:
        return {auth.query_param_name: auth.token}
    return {}
