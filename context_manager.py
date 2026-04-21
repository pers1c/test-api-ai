from __future__ import annotations

import re
from typing import Any, Dict, Optional

from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JsonPathParserError

from models import TestStep


_TEMPLATE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


class ContextManager:
    """
    Управляет общим состоянием (переменными) между шагами одного тест-кейса.

    Переменные заполняются через поле 'extract' завершённых шагов и
    подставляются как шаблоны {{имя_переменной}} в эндпоинты, заголовки,
    тело и query-параметры последующих шагов.
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    # Публичный интерфейс

    def set(self, name: str, value: str) -> None:
        """Сохраняет переменную в хранилище."""
        self._store[name] = value

    def get(self, name: str) -> Optional[str]:
        """Возвращает значение переменной или None, если она не задана."""
        return self._store.get(name)

    def has(self, name: str) -> bool:
        """Проверяет, существует ли переменная в хранилище."""
        return name in self._store

    def all_vars_present(self, var_names: list[str]) -> bool:
        """Возвращает True только если все переменные из списка присутствуют в хранилище."""
        return all(self.has(name) for name in var_names)

    def extract_values(self, response_body: Any, extract_map: Dict[str, str]) -> Dict[str, str]:
        """
        Извлекает значения из тела JSON-ответа с помощью JSONPath-выражений.

        Аргументы:
            response_body: Распарсенное тело ответа (dict, list или скаляр).
            extract_map:   Словарь имя_переменной -> JSONPath-выражение.

        Возвращает:
            Словарь успешно извлечённых пар имя -> значение.
            Отсутствующие пути молча пропускаются (переменная остаётся незаданной).
        """
        extracted: Dict[str, str] = {}

        if response_body is None:
            return extracted

        for var_name, path_expr in extract_map.items():
            try:
                value = self._jsonpath_extract(response_body, path_expr)
                if value is not None:
                    self._store[var_name] = str(value)
                    extracted[var_name] = str(value)
            except (JsonPathParserError, Exception):
                # Некритично — переменная просто не будет доступна следующим шагам
                pass

        return extracted

    def resolve_step(self, step: TestStep) -> TestStep:
        """
        Возвращает глубокую копию TestStep с подставленными значениями {{переменных}}.

        Подстановка выполняется в:
          - endpoint
          - значения заголовков (headers)
          - значения тела запроса (body, рекурсивно)
          - значения query-параметров (query_params, рекурсивно)

        Аргументы:
            step: Исходный шаг, потенциально содержащий плейсхолдеры {{var}}.

        Возвращает:
            Новый TestStep с подставленными значениями всех разрешимых плейсхолдеров.
        """
        step_dict = step.model_dump()

        step_dict["endpoint"] = self._resolve_string(step_dict["endpoint"])

        if step_dict.get("headers"):
            step_dict["headers"] = self._resolve_mapping(step_dict["headers"])

        if step_dict.get("body") is not None:
            step_dict["body"] = self._resolve_value(step_dict["body"])

        if step_dict.get("query_params") is not None:
            step_dict["query_params"] = self._resolve_value(step_dict["query_params"])

        return TestStep.model_validate(step_dict)

    def clear(self) -> None:
        """Очищает все сохранённые переменные (вызывать между тест-кейсами при необходимости)."""
        self._store.clear()

    def snapshot(self) -> Dict[str, str]:
        """Возвращает копию текущего хранилища переменных (для отчётности)."""
        return dict(self._store)

    # Вспомогательные функции
    def _jsonpath_extract(self, data: Any, expr: str) -> Any:
        """Выполняет JSONPath-выражение над данными и возвращает первое совпадение."""
        # Нормализуем простую dot-нотацию без ведущего $ (например "access_token")
        if not expr.startswith("$"):
            expr = "$." + expr

        jsonpath_expr = jsonpath_parse(expr)
        matches = jsonpath_expr.find(data)
        if matches:
            return matches[0].value
        return None

    def _resolve_string(self, value: str) -> str:
        """Заменяет все вхождения {{var}} на значения из хранилища."""
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return self._store.get(var_name, match.group(0))  # оставляем как есть, если не найдено
        return _TEMPLATE_PATTERN.sub(replacer, value)

    def _resolve_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Подставляет шаблоны во все значения плоского словаря."""
        return {k: self._resolve_string(v) if isinstance(v, str) else v
                for k, v in mapping.items()}

    def _resolve_value(self, value: Any) -> Any:
        """Рекурсивно подставляет шаблоны в любое JSON-совместимое значение."""
        if isinstance(value, str):
            return self._resolve_string(value)
        elif isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(item) for item in value]
        return value
