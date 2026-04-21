from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class OpenAPISpec:
    """
    Распарсенное представление спецификации OpenAPI (2.x/3.x).

    Атрибуты:
        raw:           Полная распарсенная спецификация в виде словаря Python.
        title:         Название API из info.title.
        version:       Версия API из info.version.
        description:   Описание API из info.description.
        spec_base_url: Первый URL сервера из спецификации (servers[0].url).
        paths:         Объект paths со всеми эндпоинтами.
        components:    Объект components/definitions со схемами данных.
    """

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        info = raw.get("info", {})
        self.title: str = info.get("title", "Unknown API")
        self.version: str = info.get("version", "unknown")
        self.description: str = info.get("description", "")

        # OpenAPI 3.x использует servers[], Swagger 2.x использует host/basePath
        servers = raw.get("servers", [])
        if servers:
            self.spec_base_url: str = servers[0].get("url", "")
        else:
            # Запасной вариант для Swagger 2.x
            host = raw.get("host", "")
            base_path = raw.get("basePath", "")
            scheme = (raw.get("schemes") or ["https"])[0]
            self.spec_base_url = f"{scheme}://{host}{base_path}" if host else ""

        self.paths: dict = raw.get("paths", {})
        # OpenAPI 3.x использует components, Swagger 2.x использует definitions
        self.components: dict = raw.get("components", raw.get("definitions", {}))

    def to_json_string(self) -> str:
        """Сериализует полную спецификацию в компактную JSON-строку для отправки в LLM."""
        return json.dumps(self.raw, separators=(",", ":"), ensure_ascii=False)

    def get_endpoint_count(self) -> int:
        """Подсчитывает общее количество HTTP-операций по всем путям."""
        http_methods = {"get", "post", "put", "delete", "patch", "options", "head"}
        count = 0
        for path_item in self.paths.values():
            if isinstance(path_item, dict):
                for key in path_item:
                    if key.lower() in http_methods:
                        count += 1
        return count

    def get_summary_for_display(self) -> str:
        """Возвращает читаемую сводку для вывода в консоль."""
        parts = [
            f"  Название:   {self.title}",
            f"  Версия:     {self.version}",
            f"  Эндпоинты: {self.get_endpoint_count()}",
        ]
        if self.spec_base_url:
            parts.append(f"  Базовый URL: {self.spec_base_url}")
        if self.description:
            short = self.description[:200].replace("\n", " ")
            if len(self.description) > 200:
                short += "..."
            parts.append(f"  Описание:   {short}")
        return "\n".join(parts)


def load_spec(file_path: str) -> OpenAPISpec:
    """
    Загружает и парсит спецификацию OpenAPI из JSON или YAML файла.

    Аргументы:
        file_path: Путь к файлу спецификации (.json, .yaml или .yml).

    Возвращает:
        OpenAPISpec: Объект с распарсенной спецификацией.

    Исключения:
        FileNotFoundError: Если файл не найден.
        ValueError: Если формат файла не поддерживается, содержимое некорректно
                    или документ не является распознаваемой спецификацией OpenAPI.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл спецификации OpenAPI не найден: {file_path}")

    content = path.read_text(encoding="utf-8")

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _parse_json(content, file_path)
    elif suffix in (".yaml", ".yml"):
        raw = _parse_yaml(content, file_path)
    else:
        # Неизвестное расширение — сначала пробуем JSON, затем YAML
        raw = _try_parse_auto(content, file_path)

    _validate_openapi_structure(raw, file_path)
    return OpenAPISpec(raw)

# Вспомогательные функции
def _parse_json(content: str, source: str) -> dict:
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в файле спецификации '{source}': {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"JSON-спецификация должна быть объектом, получено: {type(result).__name__}")
    return result


def _parse_yaml(content: str, source: str) -> dict:
    try:
        result = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Некорректный YAML в файле спецификации '{source}': {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"YAML-спецификация должна быть объектом, получено: {type(result).__name__}")
    return result


def _try_parse_auto(content: str, source: str) -> Any:
    try:
        return _parse_json(content, source)
    except ValueError:
        pass
    try:
        return _parse_yaml(content, source)
    except ValueError:
        pass
    raise ValueError(
        f"Не удалось распарсить '{source}' как JSON или YAML. "
        "Используйте расширение файла .json, .yaml или .yml."
    )


def _validate_openapi_structure(raw: dict, source: str) -> None:
    """Выполняет минимальную структурную валидацию без полной проверки JSON-схемы."""
    if "openapi" not in raw and "swagger" not in raw:
        raise ValueError(
            f"'{source}' не является спецификацией OpenAPI/Swagger "
            "(отсутствует ключ 'openapi' или 'swagger' на корневом уровне)."
        )
    if "paths" not in raw:
        raise ValueError(
            f"'{source}' не содержит объекта 'paths' — нет эндпоинтов для тестирования."
        )
