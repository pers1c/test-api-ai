"""
AI-анализатор с двухэтапной генерацией тестов.

Этап 1 (Planner): модель получает всю спеку и возвращает компактный план —
список тест-кейсов с типом, целью и вовлечёнными эндпоинтами, но без деталей шагов.
Лёгкий ответ — модель не упирается в max_tokens и видит API целиком.

Этап 2 (Generator): для каждого пункта плана модель получает ТОЛЬКО релевантные
фрагменты спеки (схемы, параметры, требования безопасности) и генерирует полный
TestCase с детальными шагами. Маленький контекст → более качественные тесты.

Обе стадии используют:
  - temperature=0.3  — минимум фантазии, максимум следования схеме
  - response_format=json_object — гарантированный валидный JSON от GPTunnel/OpenAI
  - few-shot примеры — модель копирует паттерн, а не интерпретирует инструкции
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from models import AISettings, TestCase, TestSuite
from spec_parser import OpenAPISpec


logger = logging.getLogger(__name__)


# Низкоуровневый HTTP-клиент GPTunnel
_GPTUNNEL_URL = "https://gptunnel.ru/v1/chat/completions"


# ============================================================
# Аккумулятор токенов (thread-local, безопасный при параллельных прогонах)
# ============================================================
# При запуске analyze_spec создаём новый аккумулятор в текущем потоке;
# все вложенные вызовы _call_llm добавляют в него prompt/completion токены.
# По завершении analyze_spec читаем итог и возвращаем его через suite.
_usage_state = threading.local()


def _usage_reset() -> None:
    _usage_state.input = 0
    _usage_state.output = 0
    _usage_state.calls = 0


def _usage_add(prompt_tokens: int, completion_tokens: int) -> None:
    # Если аккумулятор не инициализирован в этом потоке — просто молчим
    if not hasattr(_usage_state, "input"):
        return
    _usage_state.input  += prompt_tokens or 0
    _usage_state.output += completion_tokens or 0
    _usage_state.calls  += 1


def _usage_snapshot() -> Dict[str, int]:
    return {
        "input_tokens":  getattr(_usage_state, "input",  0),
        "output_tokens": getattr(_usage_state, "output", 0),
        "calls":         getattr(_usage_state, "calls",  0),
    }


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    settings: AISettings,
    api_key: Optional,
    *,
    max_tokens: int,
    temperature: float = 0.3,
    json_mode: bool = True,
    verbose: bool = False,
    label: str = "",
) -> str:
    """Низкоуровневый вызов GPTunnel с response_format=json_object и низкой temperature."""
    key = api_key or os.environ.get("GPTUNNEL_API_KEY")
    if not key:
        raise ValueError(
            "API-ключ GPTunnel не найден. "
            "Задайте переменную окружения GPTUNNEL_API_KEY или передайте --api-key."
        )

    payload: Dict[str, Any] = {
        "model": settings.model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    # GPTunnel, как OpenAI-совместимый API, поддерживает response_format.
    # Это значительно повышает стабильность — модель не тратит токены
    # на форматирование ответа и фокусируется на содержимом.
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    if verbose and label:
        logger.info(
            f"  [{label}] POST GPTunnel | модель={settings.model} "
            f"| max_tokens={max_tokens} | temp={temperature}"
        )

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            _GPTUNNEL_URL,
            headers={
                "Authorization": key,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    data = response.json()

    # Аккумулируем токены (для последующего расчёта стоимости)
    usage = data.get("usage", {})
    _usage_add(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )

    if verbose:
        logger.info(
            f"  [{label}] Токены: вход={usage.get('prompt_tokens', '?')} "
            f"выход={usage.get('completion_tokens', '?')} "
            f"всего={usage.get('total_tokens', '?')}"
        )

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Неожиданный формат ответа GPTunnel: {data}") from exc


def _call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
    settings: AISettings,
    api_key: Optional,
    *,
    max_tokens: int,
    temperature: float,
    verbose: bool,
    label: str,
    max_retries: int = 2,
) -> str:
    """Вызов с повтором при транзиентных ошибках (429, невалидный JSON)."""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return _call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                settings=settings,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
                label=label,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("Превышен лимит запросов. Ожидание 60с...")
                time.sleep(60)
                last_error = exc
            else:
                raise
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                logger.warning(
                    f"[{label}] Попытка {attempt} неудачна ({exc}). Повтор..."
                )
                time.sleep(2)

    raise ValueError(
        f"[{label}] Модель не вернула корректный ответ после {max_retries} попыток. "
        f"Последняя ошибка: {last_error}"
    )


# ЭТАП 1: PLANNER — генерирует план тест-кейсов без деталей шагов
_PLANNER_SYSTEM = """\
You are a senior QA architect planning an API test suite.

Your job: analyze the OpenAPI spec and produce a lean PLAN of test cases.
The plan lists WHAT to test, not HOW — no request bodies, no headers, no assertions.
Another agent will fill in the details later.

CRITICAL: Respond with ONLY a valid JSON object — no markdown, no prose.
LANGUAGE: Write "name" and "goal" fields in RUSSIAN. Endpoints/methods stay in English.
"""

_PLANNER_EXAMPLE = {
    "plan": [
        {
            "id": "tc_001",
            "name": "Регистрация нового пользователя с валидными данными",
            "type": "stateless",
            "goal": "Проверить, что /users принимает корректный запрос и возвращает 201",
            "endpoints": ["POST /users"],
            "priority": "high",
        },
        {
            "id": "tc_002",
            "name": "Регистрация с невалидным email возвращает 422",
            "type": "status_code",
            "goal": "Проверить валидацию формата email на /users",
            "endpoints": ["POST /users"],
            "priority": "high",
        },
        {
            "id": "tc_003",
            "name": "Полный цикл: регистрация → логин → создание заказа → чтение → удаление",
            "type": "contextual",
            "goal": "Проверить основной бизнес-сценарий с аутентификацией",
            "endpoints": [
                "POST /users",
                "POST /auth/login",
                "POST /orders",
                "GET /orders/{id}",
                "DELETE /orders/{id}",
            ],
            "priority": "high",
        },
        {
            "id": "tc_004",
            "name": "Доступ к /orders без токена возвращает 401",
            "type": "status_code",
            "goal": "Проверить требование аутентификации для защищённого эндпоинта",
            "endpoints": ["GET /orders"],
            "priority": "medium",
        },
    ]
}


def _build_planner_prompt(spec: OpenAPISpec, settings: AISettings) -> str:
    has_security = bool(
        spec.raw.get("securityDefinitions")
        or spec.raw.get("components", {}).get("securitySchemes")
        or spec.raw.get("security")
    )

    # Компактная сводка эндпоинтов — модели легче её обработать, чем полный JSON спеки
    endpoint_summary = _build_endpoint_summary(spec)

    security_rule = (
        "Include auth-failure tests (401/403) for endpoints that require authentication."
        if has_security
        else "The spec defines NO security schemes — do NOT include 401/403 tests."
    )

    target_count = settings.max_test_cases
    # Рекомендуем распределение, чтобы модель не скатывалась к одному типу
    stateless_count = max(1, target_count * 40 // 100)
    status_code_count = max(1, target_count * 35 // 100)
    contextual_count = max(1, target_count * 25 // 100)

    return f"""Analyze this API and produce a test PLAN (not full tests — just the plan).

## Endpoint Summary

{endpoint_summary}

## Security

{security_rule}

## Target Distribution

Aim for ~{target_count} test cases total, roughly:
  - {stateless_count} stateless tests (basic happy-path + individual endpoint checks)
  - {status_code_count} status_code tests (validation errors, auth failures, not-found, conflicts)
  - {contextual_count} contextual tests (multi-step business workflows)

Generate only as many as the API genuinely needs — do NOT pad.

## Rules

1. Every endpoint must appear in at least one stateless OR contextual test (positive path).
2. For each endpoint with required fields, plan at least one status_code test for missing/invalid input.
3. For each write endpoint (POST/PUT/PATCH/DELETE), plan a negative test.
4. Contextual tests must follow realistic workflows discoverable from the spec:
   register → login → create → read → update → delete
   create parent → create child → list children → delete cascade
5. Prefer DEPTH of coverage (edge cases, boundary values) over BREADTH of trivial repetitions.

## Output Schema

```json
{{
  "plan": [
    {{
      "id": "tc_001",
      "name": "Russian descriptive name",
      "type": "stateless | contextual | status_code",
      "goal": "What this test verifies, in Russian",
      "endpoints": ["METHOD /path", "..."],
      "priority": "low | medium | high"
    }}
  ]
}}
```

## Example

```json
{json.dumps(_PLANNER_EXAMPLE, ensure_ascii=False, indent=2)}
```

Respond with ONLY the JSON object.
"""


def _build_endpoint_summary(spec: OpenAPISpec) -> str:
    """
    Создаёт компактную текстовую сводку эндпоинтов для планировщика.
    Модель лучше воспринимает структурированный текст, чем сырой JSON спеки.
    """
    http_methods = {"get", "post", "put", "delete", "patch"}
    lines: List = []

    for path, path_item in spec.paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in http_methods or not isinstance(op, dict):
                continue

            summary = op.get("summary") or op.get("description", "")
            summary = summary[:120].replace("\n", " ") if summary else ""

            # Определяем требуется ли авторизация
            security = op.get("security")
            requires_auth = bool(security) if security is not None else None

            auth_tag = ""
            if requires_auth is True:
                auth_tag = " [AUTH]"
            elif requires_auth is False:
                auth_tag = " [PUBLIC]"

            # Проверяем наличие обязательных параметров / тела
            has_body = "requestBody" in op
            params = op.get("parameters", [])
            required_params = [p.get("name") for p in params if p.get("required")]

            flags = []
            if has_body:
                flags.append("body")
            if required_params:
                flags.append(f"required: {', '.join(required_params[:5])}")

            flags_str = f" ({'; '.join(flags)})" if flags else ""

            line = f"  {method.upper():6} {path}{auth_tag}{flags_str}"
            if summary:
                line += f"\n         — {summary}"
            lines.append(line)

    return "\n".join(lines) if lines else "(no endpoints found)"


class TestPlanItem:
    """Один пункт плана — описание будущего теста без деталей шагов."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.name: str = data["name"]
        self.type: str = data["type"]
        self.goal: str = data.get("goal", "")
        self.endpoints: List = data.get("endpoints", [])
        self.priority: Optional = data.get("priority")

    def __repr__(self) -> str:
        return f"TestPlanItem(id={self.id}, type={self.type}, endpoints={self.endpoints})"


def _run_planner(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional,
    verbose: bool,
) -> List[TestPlanItem]:
    """Этап 1: получаем план тестов."""
    logger.info("\nЭтап 1/2: Планирование тест-сьюта...")

    user_prompt = _build_planner_prompt(spec, settings)
    raw = _call_llm_with_retry(
        system_prompt=_PLANNER_SYSTEM,
        user_prompt=user_prompt,
        settings=settings,
        api_key=api_key,
        max_tokens=4000,  # План компактный — 4k токенов хватает с запасом
        temperature=0.3,
        verbose=verbose,
        label="planner",
    )

    data = _parse_json_response(raw, "planner")
    plan_items = data.get("plan", [])
    if not plan_items:
        raise ValueError("Planner не вернул ни одного тест-кейса в плане")

    plan = [TestPlanItem(item) for item in plan_items]

    logger.info(f"  План составлен: {len(plan)} тест-кейсов")
    if verbose:
        type_counts: Dict[str, int] = {}
        for item in plan:
            type_counts[item.type] = type_counts.get(item.type, 0) + 1
        for t, c in sorted(type_counts.items()):
            logger.info(f"    {t}: {c}")

    return plan


# ЭТАП 2: GENERATOR — разворачивает каждый пункт плана в полный TestCase

_GENERATOR_SYSTEM = """\
You are a senior QA engineer writing ONE concrete API test case based on a plan item.

You receive:
  - A plan item (id, name, type, goal, involved endpoints)
  - The RELEVANT fragments of the OpenAPI spec for those endpoints only

Your job: produce one complete TestCase with fully-specified steps (method, URL, headers,
body, expected status, optional extract/depends_on_vars for contextual tests).

CRITICAL OUTPUT RULE: Respond with ONLY a valid JSON object — no markdown, no prose.
LANGUAGE: Write "name", "description", and step "description" in RUSSIAN.
All other fields (keys, values, endpoint paths) in English.

TEST DATA QUALITY RULES — BE CREATIVE AND SPECIFIC, NOT TRIVIAL:
- For emails, use realistic values: "maria.petrova@example.com", NOT "test@test.com".
- For names, use real-looking Russian/English names: "Анна Иванова", "John Smith".
- For invalid emails, use *specifically broken* cases: missing @, trailing dot,
  consecutive dots "a..b@x.com", unicode only "привет@ру", over-long local part.
- For strings violating minLength/maxLength, compute the exact violating length.
- For numeric boundaries, probe minimum-1, maximum+1, zero, negative, MAX_INT.
- For required fields, test EACH one missing individually (one step per field).
- For type violations, send string where number is expected AND vice versa.
- For enum fields, send a value NOT in the enum.
- For date fields, send malformed dates: "2024-13-45", "yesterday", "".
- For arrays with minItems, send exactly one item fewer.

CONTEXTUAL TEST RULES:
- Use `extract` to capture values from earlier responses: {"token": "$.access_token"}
- Reference them as {{variable_name}} in later steps' endpoint/headers/body.
- Add `depends_on_vars` listing the variables each step needs.
- Workflows must be REALISTIC: register → login → create → read → update → delete.

STATUS CODE TEST RULES:
- FastAPI / Pydantic return 422 (not 400) for request validation errors.
- 401 = no/invalid auth; 403 = authenticated but not authorized.
- 404 for missing resources — use ID 99999 or UUID 00000000-0000-0000-0000-000000000000.
- 409 for conflicts (duplicate email, etc.).
"""

_GENERATOR_EXAMPLE = {
    "test_case": {
        "id": "tc_002",
        "name": "Регистрация с невалидным email возвращает 422",
        "description": "Проверяет, что сервер отклоняет запросы с некорректным форматом email",
        "type": "status_code",
        "priority": "high",
        "steps": [
            {
                "description": "POST /users с email без символа @",
                "endpoint": "/users",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "Пётр Смирнов", "email": "not-an-email.example.com", "age": 30},
                "expected_status": 422,
            },
            {
                "description": "POST /users с email с двойной точкой в домене",
                "endpoint": "/users",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "Ольга Петрова", "email": "olga@example..com", "age": 25},
                "expected_status": 422,
            },
            {
                "description": "POST /users с пустой строкой вместо email",
                "endpoint": "/users",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "Иван Козлов", "email": "", "age": 40},
                "expected_status": 422,
            },
        ],
    }
}

_CONTEXTUAL_EXAMPLE = {
    "test_case": {
        "id": "tc_003",
        "name": "Полный цикл: регистрация → логин → создание заказа → удаление",
        "description": "Проверяет основной пользовательский сценарий с аутентификацией",
        "type": "contextual",
        "priority": "high",
        "steps": [
            {
                "description": "Шаг 1: Регистрация нового пользователя",
                "endpoint": "/users",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "name": "Мария Иванова",
                    "email": "maria.ivanova@example.com",
                    "password": "SecurePass123!",
                },
                "expected_status": 201,
                "extract": {"user_id": "$.id"},
            },
            {
                "description": "Шаг 2: Вход с полученными учётными данными",
                "endpoint": "/auth/login",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"email": "maria.ivanova@example.com", "password": "SecurePass123!"},
                "expected_status": 200,
                "extract": {"token": "$.access_token"},
                "depends_on_vars": ["user_id"],
            },
            {
                "description": "Шаг 3: Создание заказа с использованием токена",
                "endpoint": "/orders",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {{token}}",
                },
                "body": {"product_id": 1, "quantity": 2},
                "expected_status": 201,
                "extract": {"order_id": "$.id"},
                "depends_on_vars": ["token"],
            },
            {
                "description": "Шаг 4: Удаление созданного заказа",
                "endpoint": "/orders/{{order_id}}",
                "method": "DELETE",
                "headers": {"Authorization": "Bearer {{token}}"},
                "expected_status": 204,
                "depends_on_vars": ["token", "order_id"],
            },
        ],
    }
}


def _build_generator_prompt(
    plan_item: TestPlanItem,
    relevant_spec: Dict[str, Any],
) -> str:
    """Промпт для генерации одного полного TestCase."""

    # Выбираем подходящий пример под тип тест-кейса
    if plan_item.type == "contextual":
        example = _CONTEXTUAL_EXAMPLE
    else:
        example = _GENERATOR_EXAMPLE

    return f"""Generate ONE complete test case based on this plan item.

## Plan Item

- ID: {plan_item.id}
- Name: {plan_item.name}
- Type: {plan_item.type}
- Goal: {plan_item.goal}
- Endpoints: {", ".join(plan_item.endpoints)}

## Relevant Spec Fragments

```json
{json.dumps(relevant_spec, ensure_ascii=False, indent=2)}
```

## Output Schema

```json
{{
  "test_case": {{
    "id": "{plan_item.id}",
    "name": "...",
    "description": "...",
    "type": "{plan_item.type}",
    "priority": "low | medium | high",
    "steps": [
      {{
        "description": "...",
        "endpoint": "/path",
        "method": "GET | POST | PUT | DELETE | PATCH",
        "headers": {{}},
        "body": {{}},
        "query_params": {{}},
        "expected_status": 200,
        "extract": {{}},
        "depends_on_vars": []
      }}
    ]
  }}
}}
```

## Example for this test type

```json
{json.dumps(example, ensure_ascii=False, indent=2)}
```

## Reminders

- Keep the id EXACTLY as "{plan_item.id}".
- For stateless and status_code tests: DO NOT use `extract` or `depends_on_vars` or {{{{var}}}}.
- For status_code tests generating negative data, use MULTIPLE steps probing different invalid inputs.
- Use REALISTIC test data (Russian/English names, plausible emails, sensible values).
- Omit `expected_response_schema` — not needed.

Respond with ONLY the JSON object.
"""


def _extract_relevant_spec(
    spec: OpenAPISpec,
    plan_item: TestPlanItem,
) -> Dict[str, Any]:
    """
    Извлекает из спеки только те фрагменты, которые нужны для данного пункта плана:
    - определения вовлечённых эндпоинтов
    - схемы, на которые они ссылаются (через $ref, резолвятся рекурсивно)
    - схемы безопасности
    """
    result: Dict[str, Any] = {
        "info": {
            "title": spec.title,
            "version": spec.version,
        },
        "paths": {},
    }

    # Схемы безопасности — всегда включаем, они нужны для правильных заголовков
    if spec.raw.get("components", {}).get("securitySchemes"):
        result.setdefault("components", {})["securitySchemes"] = (
            spec.raw["components"]["securitySchemes"]
        )
    if spec.raw.get("securityDefinitions"):
        result["securityDefinitions"] = spec.raw["securityDefinitions"]

    # Парсим endpoints плана: "POST /users" -> (POST, /users)
    wanted: List = []
    for ep_str in plan_item.endpoints:
        parts = ep_str.strip().split(None, 1)
        if len(parts) == 2:
            wanted.append((parts[0].upper(), parts[1]))

    # Собираем operation-объекты и рекурсивно $ref'ы
    refs_to_resolve: set = set()

    for method, path in wanted:
        path_item = spec.paths.get(path)
        if not isinstance(path_item, dict):
            continue

        op = path_item.get(method.lower())
        if op is None:
            continue

        result["paths"].setdefault(path, {})[method.lower()] = op

        # Также копируем общие параметры path-уровня
        if "parameters" in path_item:
            result["paths"]["parameters"] = path_item["parameters"]

        # Собираем все $ref из этой операции
        _collect_refs(op, refs_to_resolve)

    # Рекурсивно резолвим $ref'ы до насыщения
    resolved_schemas: Dict[str, Any] = {}
    to_process = list(refs_to_resolve)
    while to_process:
        ref = to_process.pop()
        if ref in resolved_schemas:
            continue
        resolved = _resolve_ref(spec.raw, ref)
        if resolved is not None:
            resolved_schemas = resolved
            # Смотрим, нет ли в резолвнутой схеме вложенных $ref
            new_refs: set = set()
            _collect_refs(resolved, new_refs)
            for nr in new_refs:
                if nr not in resolved_schemas:
                    to_process.append(nr)

    # Встраиваем резолвнутые схемы в components/schemas (OpenAPI 3) или definitions (Swagger 2)
    if resolved_schemas:
        for ref, schema in resolved_schemas.items():
            # "#/components/schemas/User" -> components.schemas.User
            # "#/definitions/User" -> definitions.User
            parts = ref.lstrip("#/").split("/")
            target = result
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = schema

    return result


def _collect_refs(obj: Any, out: set) -> None:
    """Рекурсивно собирает все значения $ref из вложенного dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "$ref" and isinstance(v, str):
                out.add(v)
            else:
                _collect_refs(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_refs(item, out)


def _resolve_ref(spec_root: dict, ref: str) -> Optional[Any]:
    """Резолвит JSON-ref вида '#/components/schemas/User' в объект из спеки."""
    if not ref.startswith("#/"):
        return None
    parts = ref.lstrip("#/").split("/")
    cur: Any = spec_root
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur
        else:
            return None
    return cur


def _generate_test_case(
    plan_item: TestPlanItem,
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional,
    verbose: bool,
) -> Optional[TestCase]:
    """Этап 2 для одного пункта плана: генерируем полный TestCase."""
    relevant_spec = _extract_relevant_spec(spec, plan_item)
    user_prompt = _build_generator_prompt(plan_item, relevant_spec)

    try:
        raw = _call_llm_with_retry(
            system_prompt=_GENERATOR_SYSTEM,
            user_prompt=user_prompt,
            settings=settings,
            api_key=api_key,
            max_tokens=4000,  # Один тест-кейс легко влезает в 4k
            temperature=0.3,
            verbose=verbose,
            label=f"gen:{plan_item.id}",
        )
    except Exception as exc:
        logger.warning(f"Пропускаем {plan_item.id}: {exc}")
        return None

    try:
        data = _parse_json_response(raw, f"gen:{plan_item.id}")
        tc_data = data.get("test_case") or data  # модель иногда возвращает плоско
        return TestCase.model_validate(tc_data)
    except Exception as exc:
        logger.warning(
            f"Не удалось распарсить тест {plan_item.id}: {exc}"
        )
        return None


# Публичная точка входа
def analyze_spec(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> TestSuite:
    """
    Двухэтапный анализ спецификации: planner → generator.

    Этап 1: получаем компактный план всех тест-кейсов (один вызов LLM).
    Этап 2: для каждого пункта плана генерируем полный TestCase с деталями шагов
             (N вызовов LLM, каждый с маленьким контекстом).

    Это существенно повышает качество по сравнению с "всё за один вызов":
      - модель не обрывает ответ из-за max_tokens
      - внимание сфокусировано на одном тесте за раз
      - негативные тесты и данные становятся осмысленными

    progress_callback: опциональная функция (event_type: str, message: str, data: dict) -> None.
      Вызывается на ключевых шагах для стриминга прогресса в web-UI.
    """
    # Сбрасываем thread-local аккумулятор токенов для нового прогона
    _usage_reset()

    # Вспомогательная обёртка: безопасно вызывает callback если он задан
    def emit(event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        if progress_callback is not None:
            try:
                progress_callback(event_type, message, data or {})
            except Exception:
                # Проблемы в UI-слое не должны ронять генерацию
                pass

    logger.info(
        f"\nАнализ через GPTunnel (модель: {settings.model})"
    )
    logger.info(
        f"  Эндпоинтов: {spec.get_endpoint_count()} | "
        f"Целевое число тест-кейсов: {settings.max_test_cases}"
    )

    # --- Этап 1: планирование ---
    emit("log", f"Этап 1/2: составление плана тест-сьюта (модель: {settings.model})", {
        "endpoints": spec.get_endpoint_count(),
    })
    plan = _run_planner(spec, settings, api_key, verbose)
    emit("log", f"План составлен: {len(plan)} тест-кейсов", {"plan_size": len(plan)})

    # Обрезаем план, если модель превысила лимит
    if len(plan) > settings.max_test_cases:
        plan = plan[: settings.max_test_cases]
        logger.info(
            f"  План усечён до {settings.max_test_cases} кейсов"
        )

    # --- Этап 2: генерация каждого теста ---
    logger.info(
        f"\nЭтап 2/2: Генерация {len(plan)} тест-кейсов "
        f"по плану..."
    )
    emit("log", f"Этап 2/2: детальная генерация {len(plan)} тестов", {
        "plan_size": len(plan),
    })

    test_cases: List[TestCase] = []
    for i, plan_item in enumerate(plan, start=1):
        logger.info(
            f"  [{i}/{len(plan)}] {plan_item.id} "
            f"({plan_item.type}): {plan_item.name[:60]}..."
        )
        emit("progress", f"Генерация [{i}/{len(plan)}] {plan_item.id}: {plan_item.name}", {
            "phase": "generation",
            "index": i,
            "total": len(plan),
            "plan_item_id": plan_item.id,
            "plan_item_name": plan_item.name,
            "plan_item_type": plan_item.type,
        })
        tc = _generate_test_case(plan_item, spec, settings, api_key, verbose)
        if tc is not None:
            test_cases.append(tc)

    if not test_cases:
        raise ValueError(
            "Не удалось сгенерировать ни одного тест-кейса. "
            "Проверьте спецификацию и ответы модели (llm_response.txt)."
        )

    suite = TestSuite(
        test_cases=test_cases,
        spec_title=spec.title,
        spec_version=spec.version,
    )

    # Сохраняем финальный результат для отладки
    _save_suite_debug(suite, settings)

    success_rate = len(test_cases) / len(plan) * 100 if plan else 0
    logger.info(
        f"\nТест-сьют сгенерирован: "
        f"{len(test_cases)}/{len(plan)} кейсов "
        f"({success_rate:.0f}% успешной генерации)"
    )
    return suite


def analyze_spec_with_usage(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> tuple[TestSuite, Dict[str, int]]:
    """
    То же что analyze_spec, но дополнительно возвращает реально потреблённые токены:
      (suite, {"input_tokens": int, "output_tokens": int, "calls": int})

    Используется web-режимом для расчёта фактической стоимости прогона.
    """
    suite = analyze_spec(
        spec=spec,
        settings=settings,
        api_key=api_key,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    usage = _usage_snapshot()
    return suite, usage


# Утилиты парсинга и отладки
def _parse_json_response(raw_text: str, label: str) -> Dict[str, Any]:
    """Парсит ответ модели в dict. response_format=json_object почти всегда гарантирует
    чистый JSON, но на всякий случай оставляем fallback на strip/extract."""
    cleaned = raw_text.strip()
    cleaned = _strip_markdown_fences(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        data = _extract_json_object(cleaned)
        if data is None:
            raise ValueError(
                f"[{label}] Не удалось распарсить JSON. "
                f"Начало ответа: {raw_text[:300]!r}"
            )
        return data


def _strip_markdown_fences(text: str) -> str:
    """Удаляет обёртки ```json ... ``` если модель всё же их добавила."""
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> Optional:
    """Извлекает первый полный JSON-объект из текста через счётчик скобок."""
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
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _save_suite_debug(suite: TestSuite, settings: AISettings) -> None:
    """Сохраняет финальный суит в llm_response.txt для отладки."""
    from pathlib import Path
    from datetime import datetime

    output_path = Path("llm_response.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Финальный тест-сьют (двухэтапная генерация)\n")
        f.write(f"# Время:  {timestamp}\n")
        f.write(f"# Модель: {settings.model}\n")
        f.write(f"# Кейсов: {len(suite.test_cases)}\n")
        f.write(f"# {chr(9472) * 60}\n\n")
        f.write(suite.model_dump_json(indent=2))

    logger.info(f"  Суит сохранён → {output_path.resolve()}")
