from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import httpx
from rich.console import Console

from models import AISettings, LLMProvider, TestSuite
from spec_parser import OpenAPISpec

console = Console()

# Промпты (оставлены на английском для минимизации (контектного окна) — отправляются напрямую в LLM API)

SYSTEM_PROMPT = """\
You are an expert API testing engineer with deep knowledge of REST API design patterns, \
OpenAPI specifications, HTTP semantics, and software quality assurance best practices.

Your task: analyze the provided OpenAPI specification and generate a comprehensive, \
realistic test suite covering three categories:
  - stateless   : independent tests that require no prior setup
  - contextual  : multi-step business workflow tests (login -> create -> verify -> delete)
  - status_code : tests that verify exact HTTP status codes for specific error conditions

CRITICAL OUTPUT RULE: Respond with ONLY a valid JSON object. No markdown code fences, \
no explanation text, no preamble, no trailing commentary. \
The response must be directly parseable by json.loads().\
\
LANGUAGE RULE: Write all "name", "description", and "step description" fields in RUSSIAN. \
All other fields (endpoints, methods, keys, values) remain in English as required by the schema.\
"""

# Схема вывода — компактная, только обязательные поля
_OUTPUT_SCHEMA = {
    "spec_title": "string",
    "spec_version": "string",
    "test_cases": [
        {
            "id": "tc_001",
            "name": "Название на русском",
            "description": "Что проверяет тест",
            "type": "stateless | contextual | status_code",
            "priority": "low | medium | high",
            "tags": ["tag"],
            "steps": [
                {
                    "description": "Что делает шаг",
                    "endpoint": "/path/{{var}}",
                    "method": "GET | POST | PUT | DELETE | PATCH",
                    "headers": {},
                    "body": {},
                    "query_params": {},
                    "expected_status": 200,
                    "extract": {"var_name": "$.field"},
                    "depends_on_vars": ["var_name"]
                }
            ]
        }
    ]
}


def _build_user_prompt(spec: OpenAPISpec, settings: AISettings) -> str:
    # Проверяем есть ли в спецификации схемы безопасности
    spec_has_security = bool(
        spec.raw.get("securityDefinitions") or       # Swagger 2.x
        spec.raw.get("components", {}).get("securitySchemes") or  # OpenAPI 3.x
        spec.raw.get("security")                     # глобальные требования
    )
    rules = [
        f"Generate between 1 and {settings.max_test_cases} test cases total. " \
            "Generate ONLY as many tests as needed to fully cover the API — do NOT pad " \
            "with redundant or invented tests just to reach the maximum number.",
        "For every endpoint include at least: one happy-path (2xx) test and one error test.",
        "Include contextual tests for all realistic business workflows detectable from the spec "
        "(e.g. register -> login -> create resource -> read it back -> delete it).",
        "Include status_code tests for validation errors, auth failures (401), "
        "forbidden actions (403), and not-found cases (404).",
    ]

    if settings.include_negative_tests:
        rules.append(
            "Include negative tests: omit required fields, send wrong data types, "
            "use boundary-violating values (e.g. age=-1, price=-0.01, empty string for required field)."
        )

    # Добавляем правило про авторизацию на основе спецификации
    if spec_has_security:
        rules.append(
            "Include auth tests (401, 403) only for endpoints that require authentication "
            "according to the security schemes defined in the spec."
        )
    else:
        rules.append(
            "IMPORTANT: The spec defines NO security/auth schemes. "
            "Do NOT generate any tests expecting 401 or 403 responses. "
            "All endpoints are publicly accessible."
        )
    if settings.include_edge_cases:
        rules.append(
            "Include edge cases: very long strings (256+ chars when maxLength is set), "
            "special characters, unicode, numerics at type boundaries (0, -1, MAX_INT)."
        )

    numbered_rules = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(rules))

    return f"""Analyze the OpenAPI specification below and generate a comprehensive test suite.

## OpenAPI Specification

```json
{spec.to_json_string()}
```

## Required Output Schema

Return a single JSON object matching this structure exactly:

```json
{json.dumps(_OUTPUT_SCHEMA, indent=2)}
```

## Generation Rules

{numbered_rules}

## Contextual Test Rules

A contextual test simulates a real user workflow across multiple sequential API calls.
- Each step must clearly describe its purpose (e.g. "Step 1: Register new user").
- Use the `extract` field to capture values from a response for use in later steps:
  Example: `"extract": {{"token": "$.access_token", "user_id": "$.data.id"}}`
- Reference captured values in subsequent steps as `{{{{variable_name}}}}` inside
  `headers`, `body`, `query_params`, and `endpoint`:
  Example headers: `{{"Authorization": "Bearer {{{{token}}}}"}}`
  Example endpoint: `/users/{{{{user_id}}}}`
- List the variables a step requires in `depends_on_vars`:
  Example: `"depends_on_vars": ["token", "user_id"]`
- If a step that extracts a required variable fails, subsequent dependent steps will be skipped.

CRITICAL VARIABLE RULES:
- If any step uses `depends_on_vars`, there MUST be an earlier step in the SAME test case
  that extracts that variable via the `extract` field.
- NEVER reference a variable in `depends_on_vars` or `{{{{...}}}}` templates that is not
  explicitly extracted by a previous step within the same test case.
- Stateless and status_code tests must NOT use `depends_on_vars` or `{{{{...}}}}` templates
  — they must be fully self-contained with hardcoded values only.

## Status Code Test Rules

The exact status code matters for quality testing:
- FastAPI / Pydantic frameworks return 422 (not 400) for request validation errors.
- 401 = unauthenticated (no or invalid token); 403 = authenticated but not authorized.
- 404 = resource not found; 409 = conflict (duplicate entry).
- Infer the likely framework from the spec description or x-* extensions if present.

## Test Data Rules

- Use realistic data: names like "John Smith", emails like "john.smith@example.com".
- For invalid-data tests, use clearly wrong values: age=-1, email="not-an-email", name="".
- For length-boundary tests, generate a string of exactly maxLength+1 characters.
- Test IDs for non-existent resources: use 99999 or a UUID like "00000000-0000-0000-0000-000000000000".
- Omit the "expected_response_schema" field entirely — it is not required and wastes tokens.
- Omit "tags" if not meaningful. Omit "priority" if not obvious.

## Final Reminder

Respond with ONLY the JSON object. No markdown fences. No prose. Nothing before or after the JSON.\
"""

# Основная публичная функция
def analyze_spec(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> TestSuite:
    """
    Отправляет OpenAPI спецификацию в LLM и получает структурированный TestSuite.

    Провайдер и модель задаются через settings.provider и settings.model.
    Поддерживаются: GPTunnel (OpenAI-совместимый) и Anthropic Claude напрямую.

    Аргументы:
        spec:     Распарсенная OpenAPI-спецификация.
        settings: Настройки генерации AI (модель, провайдер, количество тестов и т.д.).
        api_key:  Явный API-ключ. Если не указан — берётся из переменной окружения
                  GPTUNNEL_API_KEY (для GPTunnel) или ANTHROPIC_API_KEY (для Anthropic).
        verbose:  Выводить ли детали запроса и ответа.

    Возвращает:
        TestSuite: Валидированная Pydantic-модель со всеми сгенерированными тест-кейсами.

    Исключения:
        ValueError: Если модель вернула некорректный JSON или JSON не соответствует схеме.
        httpx.HTTPError: При сетевых ошибках.
    """
    user_prompt = _build_user_prompt(spec, settings)

    provider_label = settings.provider.value
    console.print(
        f"\n[bold cyan]Отправка спецификации в {provider_label} "
        f"(модель: {settings.model})...[/bold cyan]"
    )
    console.print(
        f"[dim]  Эндпоинтов: {spec.get_endpoint_count()} | "
        f"Макс. тест-кейсов: {settings.max_test_cases}[/dim]"
    )

    raw_response = _call_with_retry(
        user_prompt=user_prompt,
        settings=settings,
        api_key=api_key,
        verbose=verbose,
    )

    # Сохраняем сырой ответ модели в файл для отладки неполных ответов
    _save_raw_response(raw_response, settings)

    if verbose:
        console.print(f"\n[dim]Длина ответа: {len(raw_response)} символов[/dim]")

    suite = _parse_response(raw_response)
    console.print(
        f"[bold green]Тест-сьют сгенерирован: "
        f"{len(suite.test_cases)} тест-кейсов[/bold green]"
    )
    return suite


def _save_raw_response(raw_response: str, settings: "AISettings") -> None:
    """
    Сохраняет сырой ответ модели в файл llm_response.txt рядом со скриптом.

    Позволяет увидеть полный ответ модели — в том числе обрезанный JSON
    """
    from pathlib import Path
    from datetime import datetime

    output_path = Path("llm_response.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Сырой ответ модели\n")
        f.write(f"# Время:     {timestamp}\n")
        f.write(f"# Провайдер: {settings.provider.value}\n")
        f.write(f"# Модель:    {settings.model}\n")
        f.write(f"# Длина:     {len(raw_response)} символов\n")
        f.write(f"# {chr(9472) * 60}\n\n")
        f.write(raw_response)

    console.print(f"[dim]  Ответ модели сохранён → {output_path.resolve()}[/dim]")

# Диспетчер провайдеров

def _call_with_retry(
    user_prompt: str,
    settings: AISettings,
    api_key: Optional[str],
    verbose: bool,
    max_retries: int = 2,
) -> str:
    """Вызывает нужный провайдер с логикой повтора при ошибках."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            if settings.provider == LLMProvider.GPTUNNEL:
                return _call_gptunnel(user_prompt, settings, api_key, verbose)
            elif settings.provider == LLMProvider.ANTHROPIC:
                return _call_anthropic(user_prompt, settings, api_key, verbose)
            else:
                raise ValueError(f"Неизвестный провайдер: {settings.provider}")

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                console.print(f"[yellow]Попытка {attempt} неудачна ({exc}). Повтор...[/yellow]")
                time.sleep(2)
        except httpx.HTTPStatusError as exc:
            # 429 — превышен лимит запросов
            if exc.response.status_code == 429:
                console.print("[yellow]Превышен лимит запросов. Ожидание 60с...[/yellow]")
                time.sleep(60)
                last_error = exc
            else:
                raise

    raise ValueError(
        f"Модель не вернула корректный JSON после {max_retries} попыток. "
        f"Последняя ошибка: {last_error}"
    )

# GPTunnel — OpenAI-совместимый провайдер
def _call_gptunnel(
    user_prompt: str,
    settings: AISettings,
    api_key: Optional[str],
    verbose: bool,
) -> str:
    # Отправляет запрос к GPTunnel API (OpenAI-совместимый формат).

    key = api_key or os.environ.get("GPTUNNEL_API_KEY")
    if not key:
        raise ValueError(
            "API-ключ GPTunnel не найден. "
            "Задайте переменную окружения GPTUNNEL_API_KEY или передайте --api-key."
        )

    payload = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    if verbose:
        console.print(f"[dim]  POST https://gptunnel.ru/v1/chat/completions[/dim]")
        console.print(f"[dim]  Модель: {settings.model} | max_tokens: {settings.max_tokens}[/dim]")

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://gptunnel.ru/v1/chat/completions",
            headers={
                "Authorization": key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    data = response.json()

    if verbose:
        usage = data.get("usage", {})
        console.print(
            f"[dim]  Токены: вход={usage.get('prompt_tokens','?')} "
            f"выход={usage.get('completion_tokens','?')} "
            f"всего={usage.get('total_tokens','?')}[/dim]"
        )
        cost = usage.get("total_cost")
        if cost is not None:
            console.print(f"[dim]  Стоимость запроса: {cost}[/dim]")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Неожиданный формат ответа GPTunnel: {data}") from exc

# ---------------------------------------------------------------------------
# Парсинг ответа
# ---------------------------------------------------------------------------

def _parse_response(raw_text: str) -> TestSuite:
    """
    Парсит текстовый ответ модели в валидированную модель TestSuite.

    Обрабатывает случаи, когда модель оборачивает JSON в markdown-блоки,
    а также JavaScript-выражения вроде "A".repeat(N) вместо реальных строк.
    """
    cleaned = _strip_markdown_fences(raw_text.strip())
    cleaned = _fix_js_expressions(cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Пробуем извлечь только JSON-объект, если есть окружающий текст
        data = _extract_json_object(cleaned)
        if data is None:
            raise ValueError(
                f"Модель вернула некорректный JSON: {exc}\n"
                f"Начало ответа: {raw_text[:500]!r}"
            ) from exc

    try:
        return TestSuite.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"Ответ модели не соответствует схеме TestSuite: {exc}\n"
            f"Ключи ответа: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        ) from exc


def _fix_js_expressions(text: str) -> str:
    """
    Заменяет JavaScript-выражения которые модель иногда вставляет в JSON.

    Примеры замен:
      "A".repeat(257)  ->  "AAAA...AAA" (строка из 257 символов)
      "x".repeat(10)   ->  "xxxxxxxxxx"
    """
    import re

    def replacer(match: re.Match) -> str:
        char = match.group(1)   # символ для повторения
        count_str = match.group(2)  # количество повторений
        try:
            count = int(count_str)
            # Ограничиваем длину чтобы не раздувать JSON
            count = min(count, 500)
            return f'"{char * count}"'
        except ValueError:
            return match.group(0)  # оставляем как есть если не распарсили

    # Паттерн: "X".repeat(N) где X — любой символ, N — число
    pattern = re.compile(r'"(.)"\.repeat\((\d+)\)')
    return pattern.sub(replacer, text)


def _strip_markdown_fences(text: str) -> str:
    """Удаляет обёртки ```json ... ``` или ``` ... ```, если они присутствуют."""
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Пытается извлечь JSON-объект из текста, который может содержать окружающий текст.
    Находит первый '{' и соответствующий закрывающий '}' с помощью счётчика скобок.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None