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
from typing import Any, Dict, List, Optional, Tuple

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


def _compute_endpoint_stats(spec: OpenAPISpec) -> Dict[str, int]:
    """
    Считает структурные характеристики API для расчёта рекомендуемого числа тестов.

    Возвращает словарь с количествами:
      - total:                всего HTTP-операций
      - write:                POST/PUT/PATCH/DELETE
      - read:                 GET
      - auth:                 операций, требующих авторизации
      - with_body:            с requestBody
      - with_required_params: с обязательными параметрами
      - declared_404:         операций, явно декларирующих 404 в responses
      - declared_409:         операций, явно декларирующих 409 в responses
      - custom_validators:    полей тела с НЕТРИВИАЛЬНЫМИ ограничениями
                              (pattern/format/enum/min*/max* — то, что НЕ обеспечивается
                              автоматически type-проверкой Pydantic и реально может
                              содержать кастомные баги)
    """
    http_methods = {"get", "post", "put", "delete", "patch"}
    write_methods = {"post", "put", "patch", "delete"}
    # Ограничения, которые реально стоят отдельного негативного теста
    # (type-проверка Pydantic тривиальна и тестирует фреймворк, а не наш код)
    meaningful_constraint_keys = {
        "pattern", "format", "enum",
        "minLength", "maxLength",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minItems", "maxItems",
    }
    stats = {
        "total": 0,
        "write": 0,
        "read": 0,
        "auth": 0,
        "with_body": 0,
        "with_required_params": 0,
        "declared_404": 0,
        "declared_409": 0,
        "custom_validators": 0,
    }

    global_security = spec.raw.get("security")

    for path_item in spec.paths.values():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            m = method.lower()
            if m not in http_methods or not isinstance(op, dict):
                continue
            stats["total"] += 1
            if m in write_methods:
                stats["write"] += 1
            elif m == "get":
                stats["read"] += 1

            # security: операция явно объявила, либо есть глобальная и операция не отменила её []
            op_security = op.get("security")
            if op_security:
                stats["auth"] += 1
            elif op_security is None and global_security:
                stats["auth"] += 1

            if "requestBody" in op:
                stats["with_body"] += 1
            params = op.get("parameters", []) or []
            if any(p.get("required") for p in params if isinstance(p, dict)):
                stats["with_required_params"] += 1

            # Задекларированные коды — нужны, чтобы планировать только то,
            # что сервер обещает реально возвращать
            responses = op.get("responses") or {}
            if isinstance(responses, dict):
                if "404" in responses or 404 in responses:
                    stats["declared_404"] += 1
                if "409" in responses or 409 in responses:
                    stats["declared_409"] += 1

            # Кастомные валидаторы в теле — единственное место, где негативные
            # тесты ловят настоящие баги, а не работу Pydantic
            body_schema = _resolve_body_schema(op, spec.raw)
            if isinstance(body_schema, dict):
                props = body_schema.get("properties") or {}
                if isinstance(props, dict):
                    for prop in props.values():
                        if not isinstance(prop, dict):
                            continue
                        # Разворачиваем $ref, если поле — ссылка
                        if "$ref" in prop and isinstance(prop["$ref"], str):
                            resolved = _resolve_ref(spec.raw, prop["$ref"])
                            if isinstance(resolved, dict):
                                prop = resolved
                        if any(k in prop for k in meaningful_constraint_keys):
                            stats["custom_validators"] += 1

    return stats


def _compute_test_targets(
    spec: OpenAPISpec, settings: AISettings
) -> Dict[str, int]:
    """
    Рассчитывает рекомендуемое количество тест-кейсов по типам.

    Эвристика смещает фокус с «422 на каждое отсутствующее поле» (где работает
    Pydantic, а не наш код) на ВЫСОКОЦЕННЫЕ негативы — auth, бизнес-конфликты,
    кастомные валидаторы:

      stateless    = N эндпоинтов                     (по 1 happy-path на каждый)
      status_code  = auth × 2                         (401 без токена + 403 чужой ролью)
                   + declared_409                     (по 1 конфликту на эндпоинт где 409 объявлен)
                   + declared_404 // 2                (404 только где задекларирован, не на всех)
                   + custom_validators                (по 1 негативу на каждое поле с реальным ограничением)
                   + write × 0.3                      (минимальный заполнитель для write без auth/конфликтов)
      contextual   = max(3, N // 4)                   (бизнес-сценарии — главная ценность LLM)

    Итог ограничивается сверху settings.max_test_cases (пропорциональное масштабирование).
    """
    s = _compute_endpoint_stats(spec)
    total_endpoints = max(1, s["total"])

    # Stateless — только для эндпоинтов БЕЗ auth: positive-вызов с пустой БД
    # без токена. Auth-защищённые эндпоинты в принципе не могут быть positive
    # stateless — без токена они вернут 401.
    public_endpoints = max(0, s["total"] - s["auth"])
    stateless = max(1, public_endpoints)

    # Высокоценные негативы: auth, конфликты, кастомные валидаторы.
    # Pydantic-уровневые 422 (missing required / type mismatch) намеренно не считаем —
    # они тестируют фреймворк, а не бизнес-логику.
    status_code = (
        s["auth"] * 2
        + s["declared_409"]
        + s["declared_404"] // 2
        + s["custom_validators"]
    )
    minimum_negative = max(3, s["write"] * 3 // 10)
    status_code = max(status_code, minimum_negative)

    # Contextual — единственный способ покрыть auth-защищённые эндпоинты
    # positive-вызовами (через register → login → use token). Поэтому базовое
    # число + дополнительные потоки на группу auth-эндпоинтов.
    contextual = max(3, total_endpoints // 4) + s["auth"] // 3

    raw_total = stateless + status_code + contextual
    cap = settings.max_test_cases

    # Если расчётное число превышает потолок — пропорционально ужимаем
    if raw_total > cap:
        scale = cap / raw_total
        stateless = max(1, int(stateless * scale))
        status_code = max(1, int(status_code * scale))
        contextual = max(1, int(contextual * scale))

    return {
        "stateless": stateless,
        "status_code": status_code,
        "contextual": contextual,
        "total": stateless + status_code + contextual,
        "endpoints": s["total"],
        "write_endpoints": s["write"],
        "auth_endpoints": s["auth"],
        "declared_404": s["declared_404"],
        "declared_409": s["declared_409"],
        "custom_validators": s["custom_validators"],
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

    targets = _compute_test_targets(spec, settings)
    target_count = targets["total"]
    stateless_count = targets["stateless"]
    status_code_count = targets["status_code"]
    contextual_count = targets["contextual"]

    # Пользовательские инструкции (если заданы) — встраиваются ВЫШЕ всего остального,
    # чтобы модель восприняла их как приоритетные дополнительные требования к плану.
    user_instr = (settings.user_instructions or "").strip()
    user_block = (
        f"""## EXTRA USER INSTRUCTIONS (high priority)

The user explicitly asked to focus on / not miss the following. Treat these as
MANDATORY requirements for the plan, in addition to the general rules below.
If anything here conflicts with the general distribution, expand the plan to
satisfy the user — do not drop user requirements to stay within the suggested counts.

\"\"\"
{user_instr}
\"\"\"

"""
        if user_instr
        else ""
    )

    return f"""Analyze this API and produce a test PLAN (not full tests — just the plan).

{user_block}

## API Structure

- Total endpoints: {targets["endpoints"]}
- Write endpoints (POST/PUT/PATCH/DELETE): {targets["write_endpoints"]}
- Endpoints requiring authentication: {targets["auth_endpoints"]}  ← each MUST be covered by a contextual test (register → login → use token)
- Endpoints declaring 404 in responses: {targets["declared_404"]}
- Endpoints declaring 409 in responses: {targets["declared_409"]}
- Body fields with custom validators (pattern/enum/min*/max*/etc.): {targets["custom_validators"]}

## Endpoint Summary

{endpoint_summary}

## Security

{security_rule}

## Target Distribution

Produce approximately {target_count} test cases:
  - {stateless_count} stateless tests   (positive happy-path: at least ONE per endpoint)
  - {status_code_count} status_code tests — see priority guidance below
  - {contextual_count} contextual tests  (multi-step business workflows — highest VALUE)

## status_code Priority Guidance — WHAT TO TEST AND WHAT TO SKIP

status_code tests should focus on REAL business behaviour, not on testing the framework.
Use this priority order (descending value):

🟢 HIGH PRIORITY — always include if the API has them:
   1. 401 (no token / bad token) on every authenticated endpoint
   2. 403 (authenticated but wrong role) where roles/permissions exist
   3. 409 (business conflicts) on endpoints declaring 409 — duplicates, state-machine
      violations like "cannot ship a delivered order"
   4. Custom validator violations: a step that violates a field's pattern / format / enum /
      minLength / maxLength / minimum / maximum / exclusiveMin / exclusiveMax. ONLY for
      keywords that ACTUALLY appear in the spec — see Field Constraints in summary.

🟡 MEDIUM PRIORITY — include sparingly:
   5. 404 (missing resource) for by-id endpoints (GET/PUT/PATCH/DELETE /x/{{id}}) called with
      a non-existent id like 99999. This applies EVEN IF only 200/422 are documented — the
      handler raises 404 in code. Do NOT plan 404 on collection endpoints (GET /x). Do NOT plan
      a non-existent-id negative on AGGREGATE/derived reads (.../{{id}}/rating, .../average,
      /stats) — they return 200 with null/0 for any id, never 404/422.
   6. One negative for cross-field business rules visible in the spec description.

🔴 DO NOT PLAN — these waste budget and test the framework, not your code:
   7. "Missing required field returns 422" — this is Pydantic, not your logic.
      EXCEPTION: include exactly ONE such test per API as a smoke check.
   8. "String field as number returns 422" / "Number field as string returns 422" — Pydantic.
   9. "Empty string for a field without minLength → 422" — HALLUCINATION, the spec allows it.
  10. "Huge number for a field without maximum → 422" — HALLUCINATION.
  11. "Special characters for a field without pattern → 422" — HALLUCINATION.
  12. "404 on a COLLECTION endpoint (GET /x)" — it returns 200 with an empty list, not 404.
      (By-id endpoints /x/{{id}} DO return 404 for non-existent ids — see #5 above.)

Lean status_code budget toward HIGH-priority tests. If the API has zero authentication and
zero declared 409s and zero custom validators (typical bare CRUD on FastAPI/Pydantic), it
genuinely needs FEW status_code tests — a handful at most. Don't pad.

## Rules

1. EVERY endpoint MUST appear in at least one stateless OR contextual test (positive path).
   Missing an endpoint is a critical failure. Do NOT skip "boring" endpoints (root, /stats,
   GET-by-id, DELETE, PATCH, nested reviews/rating) to save budget — every operation listed in
   the Endpoint Summary must be covered. (A deterministic post-pass will add any endpoint you
   miss, but plan them yourself so the workflows are coherent.)
2. For EVERY endpoint with declared custom validators (pattern/enum/min*/max* — visible in
   the endpoint summary), plan ONE status_code test that violates one of those constraints.
   Combine multiple violations of the SAME field into a single plan item with multiple steps.
3. For EVERY write endpoint that declares 409, plan ONE business-conflict test (duplicate
   resource, illegal state transition).
4. For EVERY endpoint that requires authentication, plan ONE 401 (no token) AND, where
   roles/permissions are mentioned, ONE 403 (wrong role). These are the most valuable
   negative tests — they catch real security regressions.
5. Plan a 404 (missing-resource) test for by-id endpoints (.../{{id}}) using a non-existent id
   like 99999 — the handler raises 404 even when only 200/422 are documented. Expected code is
   404, NOT 422 (422 is only for a malformed id). Do NOT plan 404 on collection endpoints
   (GET /x) — they return an empty 200 list instead.
6. Contextual tests are the MAIN VALUE delivered by this generator. Build realistic workflows
   from the spec — these catch integration bugs the framework cannot:
   register → login → create → read → update → delete
   create parent → create child → list children → delete cascade
   pending → confirmed → shipped → delivered (full state machine)
7. AVOID padding status_code tests with framework-level checks. "POST /x without required
   field returns 422" is testing Pydantic, not your API. Include AT MOST ONE such smoke
   test for the entire API, not one per endpoint.
8. CRITICAL — only test constraints THAT ACTUALLY EXIST in the spec. The endpoint summary
   above lists for each endpoint: required body fields, query params (with their type/enum),
   and declared response codes. A negative test is meaningful ONLY when it violates one of
   these. Do NOT plan tests like:
     • "empty title" for a string field with no minLength
     • "very long name" for a string field with no maxLength
     • "huge price" for a number with only exclusiveMinimum and no maximum
   These will fail at execution because the server actually accepts them.
   (Note: a missing-ID 404 on a by-id endpoint is NOT a hallucination — see status_code rule
   #5; the server really does return 404 there.)
9. For endpoints with QUERY PARAMETERS (visible in the summary), include AT LEAST ONE
   positive test that uses the filter (with a realistic value matching its type/enum),
   not just a bare call.
10. CRITICAL — auth-protected positive tests MUST be contextual. If an endpoint is marked
   [AUTH] in the endpoint summary, its positive (2xx) test CANNOT be stateless — without a
   token the server returns 401. The plan item MUST be type=contextual and its endpoint list
   MUST start with the auth flow:
       POST /auth/register, POST /auth/login, then the target endpoint
   (or just POST /auth/login if registration is not required for that role).
   The ONLY stateless tests on auth-protected endpoints are negative-path 401 tests
   (type=status_code, expected 401 with no token).
11. CRITICAL — every test case that registers a new user MUST use a UNIQUE username
   different from all other test cases in this plan. Many servers return 500 or 409 on
   duplicate registration. Pick distinct realistic names per plan item:
       tc_005 → "anna_petrova_05", tc_012 → "ivan_smirnov_12", etc.
   (The generator step will additionally append a per-run token, so the same plan re-run
   against a persistent DB stays unique — you only need distinct names within the plan.)
12. CRITICAL — happy-path (2xx) tests on GET/PUT/PATCH/DELETE /{{id}} MUST be type="contextual"
   with a create step first — NEVER "stateless" AND NEVER "status_code". The resource will not
   exist otherwise and you get 404. "stateless" is only for collection GETs / create POSTs /
   truly stateless reads. "status_code" is only for NEGATIVE paths (401/403/404/409/422), never
   for a 200/204 read of a specific id.
13. CRITICAL — a NEGATIVE validation/conflict test (expected 422 or 409) on an AUTH-protected
   endpoint ([AUTH] in the summary) MUST be type="contextual" and start with register→login,
   then send the invalid/conflicting request WITH the Bearer token. Without a token the auth
   layer returns 401 BEFORE validation, so a standalone status_code 422 test always fails.
   Only a pure 401 (missing-token) test may be standalone status_code.
14. CRITICAL — endpoints marked [ADMIN] in the summary (or whose description says "только
   admin" / admin-only — e.g. create/update/delete of books or authors) require an ADMIN role,
   not just any logged-in user. Plan them as contextual with the register→login→target chain;
   the generator will register with role:"admin" so the token has admin rights. A normal
   authenticated user gets 403 there.

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


def _resolve_body_schema(
    op: Dict[str, Any], spec_root: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Возвращает разрешённую (после $ref) JSON-схему тела запроса операции,
    либо None, если у операции нет body или схема невалидна.
    Поддерживает OpenAPI 3 и Swagger 2.
    """
    schema: Optional[Dict[str, Any]] = None
    rb = op.get("requestBody")
    if isinstance(rb, dict):
        if "$ref" in rb:
            ref_val = rb["$ref"]
            if isinstance(ref_val, str):
                resolved_rb = _resolve_ref(spec_root, ref_val)
                if isinstance(resolved_rb, dict):
                    rb = resolved_rb
        content = rb.get("content") if isinstance(rb, dict) else None
        if isinstance(content, dict) and content:
            first_media = next(iter(content.values()))
            if isinstance(first_media, dict):
                schema = first_media.get("schema")

    if schema is None:
        for p in op.get("parameters", []) or []:
            if isinstance(p, dict) and p.get("in") == "body":
                schema = p.get("schema")
                break

    if not isinstance(schema, dict):
        return None

    seen_refs: set = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref_val = schema["$ref"]
        if not isinstance(ref_val, str) or ref_val in seen_refs:
            return None
        seen_refs.add(ref_val)
        resolved = _resolve_ref(spec_root, ref_val)
        if not isinstance(resolved, dict):
            return None
        schema = resolved

    return schema if isinstance(schema, dict) else None


def _extract_body_required_fields(
    op: Dict[str, Any], spec_root: Dict[str, Any]
) -> List[str]:
    """Возвращает список обязательных полей тела запроса для операции."""
    schema = _resolve_body_schema(op, spec_root)
    if schema is None:
        return []
    required = schema.get("required")
    if isinstance(required, list):
        return [str(x) for x in required if isinstance(x, (str, int))]
    return []


# Ключи схемы, которые описывают РЕАЛЬНЫЕ ограничения и могут быть
# использованы для генерации НЕГАТИВНЫХ тестов. Если поля в этом списке
# в схеме НЕТ, негативный тест на это ограничение генерировать НЕЛЬЗЯ —
# это будет галлюцинация (модели любят выдумывать minLength/maxLength).
_CONSTRAINT_KEYWORDS = (
    "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties",
    "enum", "const",
)


def _format_property_constraints(
    prop_name: str, prop_schema: Dict[str, Any]
) -> Optional[str]:
    """
    Возвращает строку вида "price: type=number, exclusiveMinimum=0"
    для одного свойства, либо None если нет полезных ограничений.
    """
    parts: List[str] = []
    type_val = prop_schema.get("type")
    if isinstance(type_val, str):
        parts.append(f"type={type_val}")
    elif isinstance(type_val, list):
        parts.append(f"type={'|'.join(str(t) for t in type_val)}")

    for kw in _CONSTRAINT_KEYWORDS:
        if kw in prop_schema:
            val = prop_schema[kw]
            if isinstance(val, list):
                rendered = "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in val) + "]"
            else:
                rendered = json.dumps(val, ensure_ascii=False)
            parts.append(f"{kw}={rendered}")

    # nullable / anyOf с null
    any_of = prop_schema.get("anyOf")
    if isinstance(any_of, list):
        has_null = any(
            isinstance(s, dict) and s.get("type") == "null" for s in any_of
        )
        if has_null:
            parts.append("nullable=true")

    if len(parts) <= 1 and (not type_val):
        return None
    return f"{prop_name}: " + ", ".join(parts)


def _extract_body_field_constraints(
    op: Dict[str, Any], spec_root: Dict[str, Any]
) -> List[str]:
    """
    Извлекает РЕАЛЬНО задекларированные ограничения для каждого поля тела.
    Помогает модели не выдумывать ограничения, которых в спеке нет.
    """
    schema = _resolve_body_schema(op, spec_root)
    if schema is None:
        return []

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []

    required_set = set(schema.get("required") or [])
    lines: List[str] = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue

        # Если поле — $ref, разворачиваем
        if "$ref" in prop and isinstance(prop["$ref"], str):
            resolved = _resolve_ref(spec_root, prop["$ref"])
            if isinstance(resolved, dict):
                prop = resolved

        formatted = _format_property_constraints(str(name), prop)
        if formatted is None:
            continue
        prefix = "required" if name in required_set else "optional"
        lines.append(f"[{prefix}] {formatted}")
    return lines


def _extract_declared_responses(op: Dict[str, Any]) -> List[str]:
    """Возвращает отсортированный список задекларированных статус-кодов операции."""
    responses = op.get("responses")
    if not isinstance(responses, dict):
        return []
    codes: List[str] = []
    for k in responses.keys():
        s = str(k)
        if s.isdigit() or s.lower() == "default":
            codes.append(s)
    # Числовые сортируем как числа, default — в конец
    def _key(c: str) -> tuple:
        return (0, int(c)) if c.isdigit() else (1, 0)
    codes.sort(key=_key)
    return codes


def _extract_query_params(op: Dict[str, Any]) -> List[str]:
    """
    Возвращает список query-параметров операции в формате
    "name (required, enum=[...])" / "name? (optional)" и т.п.
    """
    out: List[str] = []
    for p in op.get("parameters", []) or []:
        if not isinstance(p, dict):
            continue
        if p.get("in") != "query":
            continue
        name = str(p.get("name") or "?")
        required = bool(p.get("required"))
        schema = p.get("schema") if isinstance(p.get("schema"), dict) else {}
        bits: List[str] = []
        # Тип
        t = schema.get("type")
        if isinstance(t, str):
            bits.append(f"type={t}")
        # Перечисление, диапазоны
        for kw in ("enum", "minimum", "maximum", "pattern", "format"):
            if kw in schema:
                bits.append(f"{kw}={json.dumps(schema[kw], ensure_ascii=False)}")
        label = name if required else f"{name}?"
        if bits:
            out.append(f"{label} ({'required' if required else 'optional'}: {', '.join(bits)})")
        else:
            out.append(f"{label} ({'required' if required else 'optional'})")
    return out


def _build_endpoint_summary(spec: OpenAPISpec) -> str:
    """
    Создаёт компактную текстовую сводку эндпоинтов для планировщика.
    Модель лучше воспринимает структурированный текст, чем сырой JSON спеки.

    Включает обязательные поля тела запроса (с разворачиванием $ref) —
    без этого планировщик не знал бы, для каких полей нужны негативные тесты.
    """
    http_methods = {"get", "post", "put", "delete", "patch"}
    lines: List = []

    for path, path_item in spec.paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in http_methods or not isinstance(op, dict):
                continue

            # Берём summary И description: admin-признак («только admin») часто только в
            # description, а summary автогенерится из имени функции («Create Book»).
            summary_text = op.get("summary") or ""
            desc_text = op.get("description") or ""
            combined = f"{summary_text} {desc_text}".strip()
            summary = combined[:160].replace("\n", " ") if combined else ""

            # Определяем требуется ли авторизация
            security = op.get("security")
            requires_auth = bool(security) if security is not None else None

            auth_tag = ""
            if requires_auth is True:
                auth_tag = " [AUTH]"
            elif requires_auth is False:
                auth_tag = " [PUBLIC]"

            # Явный маркер admin-only, чтобы планировщик строил admin-цепочку.
            # Только реальные admin-only — без эндпоинтов с совместным доступом.
            if _is_admin_only(op):
                auth_tag += " [ADMIN]"

            # Параметры (path/query/header)
            params = op.get("parameters", [])
            required_params = [
                p.get("name") for p in params
                if isinstance(p, dict) and p.get("required")
                and p.get("in") not in ("body", "query")
            ]
            query_params = _extract_query_params(op)

            # Тело запроса: обязательные поля (по схеме, с разворотом $ref)
            has_body = "requestBody" in op or any(
                isinstance(p, dict) and p.get("in") == "body"
                for p in (op.get("parameters") or [])
            )
            body_required = _extract_body_required_fields(op, spec.raw)

            # Задекларированные коды ответов — ключ для негативных тестов:
            # планировщик не должен планировать 404 там, где сервер вернёт 200/422.
            declared_responses = _extract_declared_responses(op)

            flags = []
            if required_params:
                flags.append(f"params required: {', '.join(required_params[:6])}")
            if query_params:
                flags.append(f"query: {', '.join(query_params[:5])}")
            if body_required:
                flags.append(f"body required: {', '.join(body_required[:10])}")
            elif has_body:
                flags.append("body")
            # Реальные ограничения полей тела — чтобы планировщик знал, ГДЕ есть
            # кастомные валидаторы и не выдумывал их там, где их нет. Правила
            # планировщика ссылаются на «constraints visible in the summary» —
            # без этой строки они опирались бы на отсутствующие данные.
            field_constraints = _extract_body_field_constraints(op, spec.raw)
            if field_constraints:
                flags.append("constraints: " + "; ".join(field_constraints[:6]))
            if declared_responses:
                flags.append(f"declared responses: {', '.join(declared_responses)}")

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

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для передачи в UI и обратного восстановления через конструктор."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "goal": self.goal,
            "endpoints": list(self.endpoints),
            "priority": self.priority,
        }


def _op_requires_auth(op: Dict[str, Any], global_security: Any) -> bool:
    """Требует ли операция авторизации — та же логика, что в _compute_endpoint_stats."""
    op_security = op.get("security")
    if op_security:
        return True
    if op_security is None and global_security:
        return True
    return False


# Сигналы admin-only в тексте: явное «только admin» / «admin only»,
# но НЕ совместный доступ вида «автор или admin», «admin видит все».
_ADMIN_ONLY_MARKERS = ("только admin", "admin only", "admin-only", "admins only", "только администратор")
_ADMIN_SHARED_MARKERS = ("или admin", "or admin", "admin видит", "admin sees", "автор отзыва")


def _is_admin_only(op: Dict[str, Any]) -> bool:
    """
    Возвращает True, только если операция ДЕЙСТВИТЕЛЬНО требует роль admin.

    Раньше признак определялся по голому вхождению подстроки "admin", из-за чего
    эндпоинты с СОВМЕСТНЫМ доступом («автор отзыва или admin», «admin видит все,
    user — только свои») ошибочно помечались admin-only — и для них генерировались
    заведомо провальные 403-тесты. Теперь формулировки совместного доступа исключаются,
    а admin-only признаётся только по явному маркеру «только admin» либо по
    задекларированному 403.
    """
    text = f"{op.get('summary','')} {op.get('description','')}".lower()
    # Совместный доступ перекрывает всё: обычный пользователь тоже имеет права.
    if any(m in text for m in _ADMIN_SHARED_MARKERS):
        return False
    responses = op.get("responses") or {}
    if isinstance(responses, dict) and ("403" in responses or 403 in responses):
        return True
    return any(m in text for m in _ADMIN_ONLY_MARKERS)


def _register_accepts_role(spec: OpenAPISpec) -> bool:
    """
    True, если тело регистрации (POST на эндпоинт, содержащий "register" в пути)
    принимает поле "role".

    Если поля нет — admin-токен через API получить НЕЛЬЗЯ, а значит позитивные (2xx)
    тесты на admin-эндпоинты недостижимы в принципе (сервер всегда вернёт 403).
    Нужно, чтобы не инструктировать модель регистрироваться с role:"admin" там, где
    это бессмысленно, и не выдавать гарантированно провальные тесты за баги API.
    """
    for path, path_item in spec.paths.items():
        if not isinstance(path_item, dict):
            continue
        if "register" not in path.lower():
            continue
        op = path_item.get("post")
        if not isinstance(op, dict):
            continue
        schema = _resolve_body_schema(op, spec.raw)
        if isinstance(schema, dict):
            props = schema.get("properties")
            if isinstance(props, dict) and "role" in props:
                return True
    return False


def _fill_coverage_gaps(
    plan: List["TestPlanItem"], spec: OpenAPISpec, settings: AISettings
) -> tuple:
    """
    Детерминированно гарантирует, что КАЖДАЯ операция спеки покрыта хотя бы одним
    пунктом плана. Планировщик-LLM регулярно пропускает эндпоинты (на практике
    покрытие ~40%), поэтому после него мы программно находим непокрытые операции
    и добавляем для них синтетические пункты плана.

    Возвращает (дополненный_план, список_добавленных_эндпоинтов "METHOD /path").
    """
    http_methods = {"get", "post", "put", "delete", "patch"}
    global_security = spec.raw.get("security")

    # Что уже покрыто планом (нормализованные (METHOD, path) по реальным операциям спеки)
    covered: set = set()
    for item in plan:
        for method, path, _op in _iter_plan_item_ops(item, spec):
            covered.add((method.upper(), path))

    # Все операции спеки в порядке объявления
    all_ops: List[tuple] = []
    for path, path_item in spec.paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() in http_methods and isinstance(op, dict):
                all_ops.append((method.upper(), path, op))

    # Существуют ли auth-эндпоинты для построения цепочки контекста
    def _has_op(m: str, p: str) -> bool:
        pi = spec.paths.get(p)
        return isinstance(pi, dict) and m.lower() in pi

    register_ep = "POST /auth/register" if _has_op("POST", "/auth/register") else None
    login_ep = "POST /auth/login" if _has_op("POST", "/auth/login") else None
    # Можно ли вообще получить admin-токен через API (есть ли поле role в регистрации)
    register_accepts_role = _register_accepts_role(spec)

    # Следующий свободный номер id (продолжаем нумерацию tc_NNN)
    max_n = 0
    for item in plan:
        m = re.match(r"tc_(\d+)", str(item.id))
        if m:
            max_n = max(max_n, int(m.group(1)))

    added_eps: List[str] = []
    new_items: List[TestPlanItem] = []

    for method, path, op in all_ops:
        if (method, path) in covered:
            continue

        requires_auth = _op_requires_auth(op, global_security)
        has_path_param = "{" in path
        target = f"{method} {path}"
        # Признак admin-only: задекларированный 403 ЛИБО явное «только admin»
        # (исключая эндпоинты с совместным доступом).
        needs_admin = _is_admin_only(op)

        endpoints: List[str] = []
        # Auth-цепочка для защищённых эндпоинтов
        if requires_auth:
            if register_ep:
                endpoints.append(register_ep)
            if login_ep:
                endpoints.append(login_ep)

        # Для эндпоинтов с path-параметром добавляем «родительский» creator (POST коллекции)
        if has_path_param:
            collection = path.split("{", 1)[0].rstrip("/")
            if collection and _has_op("POST", collection):
                creator = f"POST {collection}"
                if creator != target and creator not in endpoints:
                    endpoints.append(creator)

        endpoints.append(target)

        # Тип: защищённые и зависящие от ресурса — contextual; публичные «холодные» — stateless
        if requires_auth or has_path_param or len(endpoints) > 1:
            tc_type = "contextual"
        elif method == "GET" or (method == "POST" and not has_path_param):
            tc_type = "stateless"
        else:
            tc_type = "contextual"

        max_n += 1
        tc_id = f"tc_{max_n:03d}"

        prereq = ""
        if requires_auth:
            if needs_admin and register_accepts_role:
                role_hint = " с role=\"admin\" (эндпоинт требует прав администратора)"
            elif needs_admin:
                # admin-токен через API недостижим — честно говорим об этом модели,
                # чтобы она не строила заведомо провальную «admin-positive» цепочку.
                role_hint = (
                    " (ВНИМАНИЕ: схема регистрации НЕ принимает поле role — admin-токен "
                    "через API получить нельзя; ожидаемый код для этого эндпоинта 403, "
                    "а не 2xx)"
                )
            else:
                role_hint = ""
            prereq = (
                f" Сначала зарегистрируй нового пользователя{role_hint} с УНИКАЛЬНЫМ именем "
                f"(база на основе id {tc_id} + суффикс {{{{run_nonce}}}}) и залогинься, затем "
                "используй полученный Bearer-токен."
            )
        elif has_path_param and len(endpoints) > 1:
            prereq = " Сначала создай нужный ресурс, затем используй его id в пути."

        goal = (
            f"Позитивная проверка эндпоинта {target}: убедиться, что он отвечает "
            f"успешным кодом (2xx).{prereq} "
            "Этот кейс добавлен автоматически для полного покрытия API."
        )

        new_items.append(TestPlanItem({
            "id": tc_id,
            "name": f"Покрытие {target}",
            "type": tc_type,
            "goal": goal,
            "endpoints": endpoints,
            "priority": "medium",
        }))
        added_eps.append(target)

    return plan + new_items, added_eps


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
        max_tokens=10000,  # План может быть большим (>50 кейсов), нужен запас
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

NEGATIVE TEST RULES — DO NOT INVENT CONSTRAINTS:
- A negative test is valid ONLY if it violates a constraint THE SPEC ACTUALLY DECLARES.
- Look at the "Field Constraints" block (if present) for this endpoint. ONLY these keywords
  define real constraints: type, minLength, maxLength, pattern, format, minimum, maximum,
  exclusiveMinimum, exclusiveMaximum, multipleOf, minItems, maxItems, uniqueItems, enum, const.
- If a string field has NO minLength → empty string "" IS VALID, do NOT test it as 422.
- If a string field has NO maxLength → long strings ARE VALID.
- If a string field has NO pattern → spaces / special chars / digits-only ARE VALID.
- If a number has only `exclusiveMinimum: 0` → there is NO upper bound; huge values ARE VALID.
- If a field is NOT in `required` AND has no constraints → omitting it is VALID.
- Test only what the schema literally forbids. "Common sense" constraints are a hallucination.

RESPONSE CODE RULES — DECLARED CODES + REALISTIC 404:
- Look at the "Declared Responses" block for this endpoint (if present). Those status
  codes are guaranteed by the spec.
- MISSING-RESOURCE 404: for a by-id endpoint (GET/PUT/PATCH/DELETE /resource/{id}) called
  with a clearly non-existent id (e.g. 99999), expected_status MUST be 404 — EVEN IF the
  spec documents only 200/422. FastAPI handlers raise 404 in code while the auto-generated
  OpenAPI lists only 422. 422 is for a MALFORMED id (e.g. "abc" where an integer is
  required), NOT for a well-formed id that simply does not exist. Never expect 422 for a
  "resource not found" test.
- COLLECTION endpoints (GET /resource) never return 404 for "no data" — they return 200
  with an empty list. Do NOT write 404 tests against collection endpoints.
- AGGREGATE / DERIVED read endpoints (e.g. .../{id}/rating, .../{id}/average, /stats,
  .../count) typically DO NOT validate existence: for a well-formed but non-existent id they
  return 200 with null / 0 / empty aggregates, NOT 404 and NEVER 422. Do NOT generate an
  existence-negative (404 or 422) for a non-existent id on such an endpoint. The ONLY valid
  negative there is a MALFORMED id (e.g. "abc" for an integer) → 422. If you are unsure
  whether an endpoint aggregates, expect 200 for a non-existent id rather than 404/422.
- 422 IS EXCLUSIVELY FOR MALFORMED / CONSTRAINT-VIOLATING INPUT (wrong type, failed
  pattern/enum/min/max). A well-formed value that merely does not exist (id 99999, a valid
  integer) is NEVER a 422 — it is 404 (if the handler checks existence) or 200 (if it does
  not). Do not conflate "not found" with "validation error".
- For any OTHER error code you have not seen declared and that is not the missing-resource
  404 above: do not invent it — expect whichever code IS declared for the error case.

QUERY-PARAMETER COVERAGE:
- For endpoints with query parameters (see "Query Params" block), positive tests should
  EXERCISE the filters, not just call the endpoint bare. Add at least one step with
  realistic filter values matching the parameter's type/enum/format.

POSITIVE-PATH BODY RULES — CRITICAL:
- For any step expecting 2xx on POST/PUT/PATCH, the request body MUST include EVERY field
  listed in the schema's `required` array. Do not silently drop fields you think are optional
  — if the spec marks them required, include them with realistic values.
- Cross-check against the "Required Body Fields" block below (if present) before submitting.
  If that block lists fields you didn't include, your test will fail with 422.
- "Realistic value" still means realistic: for `price` use a plausible number (e.g. 19.99),
  for `isbn` use a valid ISBN-13 like "978-3-16-148410-0", for dates use ISO format.

STATELESS TEST RULES — CRITICAL:
- A stateless test runs WITHOUT any pre-existing resources in the database.
- NEVER put a hardcoded resource ID (e.g. "12345", a UUID) in a stateless test and
  expect a 200/204 response — the resource does not exist and the server will return 404.
- Stateless tests may only call collection endpoints (GET /items) or create endpoints
  (POST /items). Any happy-path test for GET/PUT/PATCH/DELETE /{id} MUST use the
  "contextual" type with an explicit create step first.
- If you receive a stateless plan item for an endpoint like GET/PUT/DELETE /{id} that
  expects success: convert it to a contextual test by adding a creation step at the top
  and updating "type" to "contextual".
- A positive (2xx) POST /auth/login needs an existing account: it MUST be contextual with a
  register step first, using the SAME "username...{{run_nonce}}" + password in both steps.
  A standalone login of a never-registered user returns 401, not 200.

AUTHENTICATION RULES — CRITICAL:
- Look at the spec's `security` field on the target operation AND the global `security`.
  Also consult the "Auth Requirements" block in the prompt (if present) — it lists
  per-endpoint auth state.
- If the operation requires authentication, ANY test step that calls it AND expects to
  reach the business logic of the endpoint (2xx happy-path OR 422 validation OR 409
  conflict OR 404 not-found) MUST include a valid Bearer token. The auth layer runs
  BEFORE validation — without a token, ALL of these become 401, not what you expected.
- The ONLY exception is when expected_status is exactly 401 (no token) or 403 (wrong
  role). For these, you DO want to skip / corrupt the token.
- A test that needs a valid token CANNOT be stateless. It MUST be type=contextual with
  these leading steps:
      1. POST /auth/register  body: {"username": "<base>_{{run_nonce}}", "password": "<valid>"}
         expected_status: 201   (add "role": "admin" here if the target needs admin — see ROLE RULES)
      2. POST /auth/login     body: {"username": "<base>_{{run_nonce}}", "password": "<same>"}
         expected_status: 200, extract: {"token": "$.access_token"}
      3. <target operation>   headers: {"Authorization": "Bearer {{token}}"}
- This includes 422 / 409 / 404 negative tests on auth-protected endpoints — they ALSO
  need the register+login prefix. Otherwise you'll get 401, not 422/409/404.
- NEVER write `"Authorization": "Bearer ****"`, `"Bearer token"`, `"Bearer xxx"`, or any
  hardcoded placeholder. The ONLY valid Authorization value is `"Bearer {{token}}"`
  where `{{token}}` is extracted from a real login step in the same test case.
- If you receive a stateless plan item whose endpoint requires auth: CONVERT it. Change
  type to "contextual", prepend register+login steps, attach Authorization header.
- A 401-negative test (expected_status=401) IS allowed to be stateless: just call the
  endpoint WITHOUT any Authorization header at all. Do not include the header field.

AUTH-ON-PUBLIC RULES — CRITICAL:
- Before planning ANY 401 test, consult "Auth Requirements". A 401 test makes sense ONLY
  for endpoints listed as "REQUIRES auth". Endpoints listed as "public" CANNOT return 401
  — they accept any caller. Generating a 401 test for a public endpoint will always fail.
- Common trap: GET /resource (list) is often public while POST /resource (create) requires
  auth. Treat each method+path pair separately.

UNIQUE-DATA RULES — CRITICAL (uniqueness must hold ACROSS runs, not just within a plan):
- The runner injects a built-in variable {{run_nonce}} — a short token UNIQUE to each run.
  It is available to EVERY step from the start (no extract needed).
- Any field that must be globally unique (username / login / email) MUST embed {{run_nonce}}.
  The target database PERSISTS between runs, so a literal username like "anna_petrova_05"
  succeeds the first run and then returns 400 "username already taken" on every later run.
  Embedding {{run_nonce}} makes it unique each run.
- Construct usernames as: base name + test id + {{run_nonce}}, e.g.:
      tc_005 → "anna_petrova_05_{{run_nonce}}"
      tc_012 → "ivan_smirnov_12_{{run_nonce}}"
  The register step AND the matching login step in the same test MUST use the IDENTICAL
  string (same base, same {{run_nonce}}) — otherwise login fails with 401.
- Same principle for emails: "anna.petrova.05.{{run_nonce}}@example.com".
- Do NOT also list run_nonce in depends_on_vars — it is always present.

ROLE / PRIVILEGE RULES — CRITICAL:
- Some write endpoints require an ELEVATED role (admin), not just any authenticated user.
  Signals: the endpoint declares a 403 response, or its summary/description mentions
  "admin" / "только admin" / "privileges". A normally-registered user gets 403 there.
- If the register schema (RegisterIn) accepts an optional "role" field, then to obtain an
  admin token for such endpoints, register WITH that field:
      POST /auth/register  body: {"username": "...{{run_nonce}}", "password": "...", "role": "admin"}
  then login with the same credentials and use the resulting Bearer token.
- Only request role:"admin" when the target operation actually needs it (declares 403 /
  says admin-only). For ordinary authenticated endpoints, register a normal user.
- A 403-negative test (expected_status=403) is the opposite: register a NORMAL user (no
  role field, or role:"user") and call the admin-only endpoint with that token.
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


def _collect_required_fields_for_plan_item(
    plan_item: TestPlanItem, spec: OpenAPISpec
) -> List[str]:
    """
    Для каждого эндпоинта плана (метод+путь) собирает строку вида
    "POST /books — body required: title, author, isbn, price, published_date".
    Возвращает список таких строк (только для тех, где есть body-схема с required).
    """
    lines: List[str] = []
    for ep in plan_item.endpoints:
        parts = ep.strip().split(None, 1)
        if len(parts) != 2:
            continue
        method, path = parts[0].upper(), parts[1]
        path_item = spec.paths.get(path)
        if not isinstance(path_item, dict):
            continue
        op = path_item.get(method.lower())
        if not isinstance(op, dict):
            continue
        required = _extract_body_required_fields(op, spec.raw)
        if required:
            lines.append(
                f"  {method} {path} — body required: {', '.join(required)}"
            )
    return lines


def _filter_phantom_endpoints(
    plan: List["TestPlanItem"], spec: OpenAPISpec
) -> tuple:
    """
    Отфильтровывает plan items, которые ссылаются на эндпоинты, ОТСУТСТВУЮЩИЕ
    в спеке. Модели иногда галлюцинируют пути вроде "GET /orders/{order_id}"
    при наличии только "GET /orders" — такие тесты гарантированно бесполезны.

    Считаем эндпоинт "явно вспомогательным" (register/login), если он есть в
    спеке — это нормально, что они появляются в контекстных тестах.

    Эндпоинт считается несуществующим, если:
      - путь отсутствует в spec.paths, ИЛИ
      - для пути не определён метод этой операции

    Plan item отбрасывается, если ХОТЯ БЫ ОДИН его эндпоинт несуществующий.
    Возвращает (filtered_plan, dropped_list), где dropped_list — пары
    (test_case_id, bad_endpoint_string).
    """
    http_methods = {"get", "post", "put", "delete", "patch"}
    filtered: List[TestPlanItem] = []
    dropped: List[tuple] = []

    for item in plan:
        bad: Optional[str] = None
        for ep in item.endpoints:
            parts = ep.strip().split(None, 1)
            if len(parts) != 2:
                bad = ep  # некорректный формат — тоже галлюцинация
                break
            method, path = parts[0].lower(), parts[1]
            if method not in http_methods:
                bad = ep
                break
            path_item = spec.paths.get(path)
            if not isinstance(path_item, dict) or method not in path_item:
                bad = ep
                break
        if bad is None:
            filtered.append(item)
        else:
            dropped.append((item.id, bad))

    return filtered, dropped


def _iter_plan_item_ops(
    plan_item: TestPlanItem, spec: OpenAPISpec
) -> List[tuple]:
    """Итератор (method, path, op_dict) по эндпоинтам плана."""
    out: List[tuple] = []
    for ep in plan_item.endpoints:
        parts = ep.strip().split(None, 1)
        if len(parts) != 2:
            continue
        method, path = parts[0].upper(), parts[1]
        path_item = spec.paths.get(path)
        if not isinstance(path_item, dict):
            continue
        op = path_item.get(method.lower())
        if not isinstance(op, dict):
            continue
        out.append((method, path, op))
    return out


def _collect_field_constraints_for_plan_item(
    plan_item: TestPlanItem, spec: OpenAPISpec
) -> List[str]:
    """
    Собирает РЕАЛЬНЫЕ ограничения полей тела для каждого эндпоинта плана.
    Это «противоядие» от выдумывания minLength/maxLength/maximum, которых
    в спеке нет.
    """
    lines: List[str] = []
    for method, path, op in _iter_plan_item_ops(plan_item, spec):
        constraints = _extract_body_field_constraints(op, spec.raw)
        if not constraints:
            continue
        lines.append(f"  {method} {path}:")
        for c in constraints:
            lines.append(f"    - {c}")
    return lines


def _collect_declared_responses_for_plan_item(
    plan_item: TestPlanItem, spec: OpenAPISpec
) -> List[str]:
    """Собирает задекларированные коды ответов для каждого эндпоинта плана."""
    lines: List[str] = []
    for method, path, op in _iter_plan_item_ops(plan_item, spec):
        codes = _extract_declared_responses(op)
        if codes:
            lines.append(f"  {method} {path} → {', '.join(codes)}")
    return lines


def _collect_query_params_for_plan_item(
    plan_item: TestPlanItem, spec: OpenAPISpec
) -> List[str]:
    """Собирает query-параметры для каждого эндпоинта плана."""
    lines: List[str] = []
    for method, path, op in _iter_plan_item_ops(plan_item, spec):
        qs = _extract_query_params(op)
        if qs:
            lines.append(f"  {method} {path}: {'; '.join(qs)}")
    return lines


def _collect_auth_requirements_for_plan_item(
    plan_item: TestPlanItem, spec: OpenAPISpec
) -> List[str]:
    """
    Для каждого эндпоинта плана определяет, требуется ли авторизация.
    Возвращает строки вида "POST /authors — REQUIRES auth" / "GET /books — public".
    """
    global_security = spec.raw.get("security")
    register_accepts_role = _register_accepts_role(spec)
    lines: List[str] = []
    for method, path, op in _iter_plan_item_ops(plan_item, spec):
        op_sec = op.get("security")
        # admin-признак: задекларированный 403 ЛИБО явное «только admin»
        # (эндпоинты с совместным доступом admin-only НЕ считаются).
        if _is_admin_only(op):
            if register_accepts_role:
                admin_tag = " — ADMIN role required (register with role:\"admin\")"
            else:
                admin_tag = (
                    " — ADMIN-only, but the register schema has NO 'role' field: an admin "
                    "token CANNOT be obtained via this API, so a positive 2xx test here is "
                    "unachievable — expect 403 instead"
                )
        else:
            admin_tag = ""
        # Спецификация: если operation объявила security — она актуальна;
        # если operation security == [] — это явный публичный override;
        # если None — наследует global.
        if op_sec:
            schemes = []
            for entry in op_sec:
                if isinstance(entry, dict):
                    schemes.extend(entry.keys())
            scheme_str = ", ".join(schemes) if schemes else "auth"
            lines.append(f"  {method} {path} — REQUIRES auth ({scheme_str}){admin_tag}")
        elif op_sec == []:
            lines.append(f"  {method} {path} — public (explicit override)")
        elif global_security:
            lines.append(f"  {method} {path} — REQUIRES auth (inherited global){admin_tag}")
        else:
            lines.append(f"  {method} {path} — public")
    return lines


def _build_generator_prompt(
    plan_item: TestPlanItem,
    relevant_spec: Dict[str, Any],
    user_instructions: Optional[str] = None,
    required_fields_lines: Optional[List[str]] = None,
    field_constraints_lines: Optional[List[str]] = None,
    declared_responses_lines: Optional[List[str]] = None,
    query_params_lines: Optional[List[str]] = None,
    auth_requirements_lines: Optional[List[str]] = None,
) -> str:
    """Промпт для генерации одного полного TestCase."""

    # Выбираем подходящий пример под тип тест-кейса
    if plan_item.type == "contextual":
        example = _CONTEXTUAL_EXAMPLE
    else:
        example = _GENERATOR_EXAMPLE

    required_block = ""
    if required_fields_lines:
        required_block = (
            "## Required Body Fields (MUST be present in positive-path bodies)\n\n"
            "These are pulled directly from the spec's `required` arrays. For ANY 2xx step\n"
            "on these endpoints, your request body MUST include ALL listed fields. Missing\n"
            "even one will cause a 422.\n\n"
            + "\n".join(required_fields_lines)
            + "\n\n"
        )

    constraints_block = ""
    if field_constraints_lines:
        constraints_block = (
            "## Field Constraints (the ONLY constraints the spec declares)\n\n"
            "Use this list as the SOLE source of truth for negative tests. If a field is\n"
            "absent from this block, or a particular keyword (minLength, maximum, pattern…)\n"
            "is missing for it — the corresponding constraint DOES NOT EXIST and you MUST\n"
            "NOT generate a negative test that assumes it.\n\n"
            "Examples of FORBIDDEN hallucinations:\n"
            "  • field has type=string but NO minLength → do not test empty-string as 422\n"
            "  • field has type=string but NO pattern → do not test special chars as 422\n"
            "  • field has exclusiveMinimum=0 but NO maximum → do not test huge numbers as 422\n\n"
            + "\n".join(field_constraints_lines)
            + "\n\n"
        )

    responses_block = ""
    if declared_responses_lines:
        responses_block = (
            "## Declared Responses (the ONLY status codes guaranteed by the spec)\n\n"
            "For each endpoint, ONLY these codes are documented. If you want to write a\n"
            "negative test that expects a code NOT in this list, do NOT — the server may\n"
            "return something completely different. In particular, DO NOT assume 404 for\n"
            "missing resources unless 404 is explicitly listed.\n\n"
            + "\n".join(declared_responses_lines)
            + "\n\n"
        )

    query_block = ""
    if query_params_lines:
        query_block = (
            "## Query Parameters\n\n"
            "Positive tests for these endpoints SHOULD exercise the filters with realistic\n"
            "values matching the declared type/enum/format — not just call the endpoint bare.\n\n"
            + "\n".join(query_params_lines)
            + "\n\n"
        )

    auth_block = ""
    if auth_requirements_lines:
        any_auth = any("REQUIRES auth" in ln for ln in auth_requirements_lines)
        any_public = any(" — public" in ln for ln in auth_requirements_lines)
        warnings: List[str] = []

        if any_auth:
            # Грубая эвристика — если в имени/цели плана упомянут 401, считаем
            # что тест целенаправленно проверяет отсутствие токена. Иначе нужен
            # auth flow (включая 422/409/404 на защищённых эндпоинтах).
            looks_like_401_test = (
                "401" in (plan_item.name or "")
                or "401" in (plan_item.goal or "")
                or "без токена" in (plan_item.name or "").lower()
                or "no token" in (plan_item.name or "").lower()
            )
            if looks_like_401_test:
                warnings.append(
                    "⚠️  This appears to be a 401-negative test. Do NOT include the\n"
                    "    Authorization header at all — that's the whole point of the test.\n"
                )
            else:
                warnings.append(
                    "⚠️  AT LEAST ONE ENDPOINT BELOW REQUIRES AUTH. Whatever you expect\n"
                    "    (2xx happy-path, 422 validation, 409 conflict, 404 not-found),\n"
                    "    you MUST prepend register+login steps and pass\n"
                    "    `Authorization: Bearer {{{{token}}}}` to the auth-protected calls.\n"
                    "    The auth layer runs BEFORE validation — without a real token you\n"
                    "    get 401, not 422/409/404. Set `type` to \"contextual\".\n"
                    "    NEVER hardcode \"Bearer ****\" or any placeholder string.\n"
                )

        if any_public:
            # Если плановый тест предполагает 401 на публичный эндпоинт — это бессмыслица.
            if "401" in (plan_item.name or "") or "401" in (plan_item.goal or ""):
                warnings.append(
                    "⚠️  This plan item targets 401, but at least one endpoint below is\n"
                    "    PUBLIC. Public endpoints CANNOT return 401. If ALL endpoints in\n"
                    "    this item are public, this test is impossible — return a minimal\n"
                    "    valid test case anyway, but do not invent 401 behavior.\n"
                )

        # Явное предупреждение про admin-роль, если хоть один эндпоинт её требует
        if any("ADMIN role required" in ln for ln in auth_requirements_lines) and not (
            "403" in (plan_item.name or "") or "403" in (plan_item.goal or "")
        ):
            warnings.append(
                "⚠️  AN ENDPOINT BELOW REQUIRES THE ADMIN ROLE. In the register step send\n"
                "    {\"username\": \"...{{{{run_nonce}}}}\", \"password\": \"...\", \"role\": \"admin\"}\n"
                "    so the issued token has admin rights — otherwise the call returns 403.\n"
            )

        auth_block = (
            "## Auth Requirements (per endpoint)\n\n"
            + "".join(warnings)
            + ("\n" if warnings else "")
            + "\n".join(auth_requirements_lines)
            + "\n\n"
        )

    instr = (user_instructions or "").strip()
    user_block = (
        f"""## EXTRA USER INSTRUCTIONS (high priority)

Apply these user-supplied instructions when they are relevant to this specific
test case. If they don't apply to this endpoint, ignore them silently.

\"\"\"
{instr}
\"\"\"

"""
        if instr
        else ""
    )

    return f"""Generate ONE complete test case based on this plan item.

{user_block}{auth_block}{required_block}{constraints_block}{responses_block}{query_block}

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
            resolved_schemas[ref] = resolved
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
            cur = cur[p]
        else:
            return None
    return cur


def _generate_test_case(
    plan_item: TestPlanItem,
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional,
    verbose: bool,
    max_attempts: int = 3,
) -> Tuple[Optional[TestCase], Optional[str]]:
    """Этап 2 для одного пункта плана: генерируем полный TestCase.

    Возвращает (TestCase, None) при успехе или (None, причина) при провале — чтобы
    вызывающий код мог показать потерянный кейс, а не списать его молча.

    Весь цикл «вызов → парсинг → валидация схемы» повторяется до max_attempts раз.
    _call_llm_with_retry лечит только транспортные сбои (429, битый конверт ответа);
    но самая частая причина потери кейса — обрезанный по лимиту токенов или невалидный
    JSON самого тест-кейса у дешёвых моделей. На повторах поднимаем лимит токенов
    (борьба с обрезкой длинных contextual-кейсов) и слегка повышаем temperature,
    чтобы выйти из «залипшего» плохого ответа.
    """
    relevant_spec = _extract_relevant_spec(spec, plan_item)
    required_lines = _collect_required_fields_for_plan_item(plan_item, spec)
    constraints_lines = _collect_field_constraints_for_plan_item(plan_item, spec)
    responses_lines = _collect_declared_responses_for_plan_item(plan_item, spec)
    query_lines = _collect_query_params_for_plan_item(plan_item, spec)
    auth_lines = _collect_auth_requirements_for_plan_item(plan_item, spec)
    user_prompt = _build_generator_prompt(
        plan_item,
        relevant_spec,
        settings.user_instructions,
        required_lines,
        constraints_lines,
        responses_lines,
        query_lines,
        auth_lines,
    )

    last_reason = "неизвестная причина"
    for attempt in range(1, max_attempts + 1):
        # На повторах даём вдвое больше токенов (частая причина потери — обрезка
        # длинного contextual-кейса на 4k) и чуть выше temperature ради иного ответа.
        max_tokens = 4000 if attempt == 1 else 8000
        temperature = 0.3 if attempt == 1 else 0.4
        try:
            raw = _call_llm_with_retry(
                system_prompt=_GENERATOR_SYSTEM,
                user_prompt=user_prompt,
                settings=settings,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=verbose,
                label=f"gen:{plan_item.id}",
            )
            data = _parse_json_response(raw, f"gen:{plan_item.id}")
            tc_data = data.get("test_case") or data  # модель иногда возвращает плоско
            return TestCase.model_validate(tc_data), None
        except Exception as exc:
            last_reason = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                logger.warning(
                    f"[gen:{plan_item.id}] попытка {attempt}/{max_attempts} неудачна "
                    f"({last_reason}). Повтор с лимитом 8000 токенов..."
                )

    logger.error(
        f"Не удалось сгенерировать тест {plan_item.id} «{plan_item.name}» "
        f"после {max_attempts} попыток: {last_reason}"
    )
    return None, last_reason


# Публичная точка входа
def _emit_factory(progress_callback: Optional[Any]):
    """Возвращает безопасную обёртку emit(event_type, message, data) над progress_callback."""
    def emit(event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        if progress_callback is not None:
            try:
                progress_callback(event_type, message, data or {})
            except Exception:
                # Проблемы в UI-слое не должны ронять генерацию
                pass
    return emit


def plan_test_suite(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[TestPlanItem]:
    """
    Этап 1 (планирование): план тест-сьюта без деталей шагов.

    Делает: planner (LLM) → фильтр галлюцинаций (_filter_phantom_endpoints) →
    детерминированный догенератор покрытия (_fill_coverage_gaps) → усечение по лимиту
    с сохранением покрытия.

    Выделен в отдельную функцию, чтобы web-режим мог поставить пайплайн на паузу
    между планированием и генерацией (подтверждение плана пользователем).
    """
    emit = _emit_factory(progress_callback)

    if (settings.user_instructions or "").strip():
        preview = settings.user_instructions.strip().replace("\n", " ")[:200]
        logger.info(f"  Пользовательские инструкции: {preview}")
        emit("log", f"Учитываю пользовательские инструкции: {preview}", {
            "user_instructions": settings.user_instructions,
        })

    emit("log", f"Этап 1/2: составление плана тест-сьюта (модель: {settings.model})", {
        "endpoints": spec.get_endpoint_count(),
    })
    plan = _run_planner(spec, settings, api_key, verbose)
    emit("log", f"План составлен: {len(plan)} тест-кейсов", {"plan_size": len(plan)})

    # Фикс A: отбрасываем plan items с эндпоинтами, которых НЕТ в спеке —
    # это галлюцинации модели (например, "GET /orders/{id}" при наличии
    # только "GET /orders" в спеке). Такие тесты гарантированно упадут
    # с 404 / неправильным кодом и тратят токены впустую.
    plan, dropped_phantoms = _filter_phantom_endpoints(plan, spec)
    if dropped_phantoms:
        logger.info(
            f"  Отброшено {len(dropped_phantoms)} кейсов с несуществующими "
            f"эндпоинтами (галлюцинации модели):"
        )
        for tc_id, bad_ep in dropped_phantoms:
            logger.info(f"    {tc_id}: эндпоинт «{bad_ep}» не существует в спеке")
        emit("log", f"Отброшено {len(dropped_phantoms)} кейсов с несуществующими эндпоинтами", {
            "dropped": [
                {"id": tc_id, "endpoint": bad_ep}
                for tc_id, bad_ep in dropped_phantoms
            ],
        })

    # Детерминированно добиваем покрытие: каждая операция спеки должна попасть
    # хотя бы в один пункт плана. LLM-планировщик это правило часто нарушает.
    plan, added_eps = _fill_coverage_gaps(plan, spec, settings)
    if added_eps:
        logger.info(
            f"  Догенерировано {len(added_eps)} кейсов для полного покрытия API: "
            f"{', '.join(added_eps)}"
        )
        emit("log", f"Догенерировано {len(added_eps)} кейсов для полного покрытия API", {
            "added_endpoints": added_eps,
        })

    # Усечение по лимиту с сохранением покрытия (если модель раздула план)
    if len(plan) > settings.max_test_cases:
        plan = _truncate_plan_keep_coverage(plan, spec, settings.max_test_cases)
        logger.info(f"  План усечён до {len(plan)} кейсов (покрытие сохранено)")

    return plan


def _truncate_plan_keep_coverage(
    plan: List["TestPlanItem"], spec: OpenAPISpec, cap: int
) -> List["TestPlanItem"]:
    """
    Усекает план до cap пунктов, НЕ удаляя единственный пункт, покрывающий эндпоинт.

    Сначала считаем, сколько пунктов покрывают каждую операцию; затем проходим план
    с конца и выкидываем «избыточные» пункты (все их операции покрыты и другими),
    пока не уложимся в лимит.
    """
    if len(plan) <= cap:
        return plan

    cover_count: Dict[tuple, int] = {}
    item_ops: Dict[str, set] = {}
    for item in plan:
        ops = {(m.upper(), p) for m, p, _ in _iter_plan_item_ops(item, spec)}
        item_ops[item.id] = ops
        for op in ops:
            cover_count[op] = cover_count.get(op, 0) + 1

    kept = list(plan)
    # Идём с конца — последними добавлены догенерированные кейсы покрытия, но они
    # как раз часто единственные для своих эндпоинтов, поэтому защита по cover_count
    # не даст их удалить.
    for item in reversed(plan):
        if len(kept) <= cap:
            break
        ops = item_ops[item.id]
        # Можно удалить, только если каждая операция покрыта ещё кем-то
        if ops and all(cover_count.get(op, 0) > 1 for op in ops):
            kept.remove(item)
            for op in ops:
                cover_count[op] -= 1

    return kept


def generate_suite_from_plan(
    plan: List[TestPlanItem],
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> TestSuite:
    """
    Этап 2 (генерация): разворачивает каждый пункт плана в полный TestCase.
    N вызовов LLM, каждый с маленьким сфокусированным контекстом.
    """
    emit = _emit_factory(progress_callback)

    logger.info(
        f"\nЭтап 2/2: Генерация {len(plan)} тест-кейсов по плану..."
    )
    emit("log", f"Этап 2/2: детальная генерация {len(plan)} тестов", {
        "plan_size": len(plan),
    })

    test_cases: List[TestCase] = []
    failed: List[Dict[str, Any]] = []  # потерянные кейсы с причиной (не теряем молча)
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
        tc, reason = _generate_test_case(plan_item, spec, settings, api_key, verbose)
        if tc is not None:
            test_cases.append(tc)
        else:
            failed.append({
                "id": plan_item.id,
                "name": plan_item.name,
                "type": plan_item.type,
                "stage": "generation",
                "reason": reason or "неизвестная причина",
            })

    if failed:
        logger.warning(
            f"  Не сгенерировано {len(failed)}/{len(plan)} кейсов после ретраев: "
            f"{', '.join(f['id'] for f in failed)}"
        )
        emit("log",
             f"Не удалось сгенерировать {len(failed)} из {len(plan)} кейсов "
             f"(кейсы не теряются молча — см. список)",
             {"failed_generations": failed})

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

    # Детерминированный пост-валидатор: чинит уникальность учёток, добавляет
    # недостающие register→login, снимает токен у 401-тестов и отбраковывает
    # заведомо невыполнимые кейсы (admin без доступного admin-токена и т.п.).
    # Импорт ленивый, чтобы избежать цикла импорта на уровне модуля.
    try:
        from suite_validator import validate_and_fix_suite

        raw_count = len(suite.test_cases)
        ids_before = {tc.id for tc in suite.test_cases}
        suite, actions = validate_and_fix_suite(suite, spec)
        ids_after = {tc.id for tc in suite.test_cases}
        if actions:
            logger.info(f"  Пост-валидация: {len(actions)} действий, "
                        f"кейсов {raw_count} → {len(suite.test_cases)}")
            for a in actions:
                logger.info(f"    {a}")
            emit("log",
                 f"Пост-валидация сьюта: {raw_count} → {len(suite.test_cases)} кейсов",
                 {"actions": actions,
                  "before": raw_count,
                  "after": len(suite.test_cases)})

        # Кейсы, отбракованные валидатором, тоже не должны исчезать молча —
        # причина лежит в строке действия вида "<id>: отбракован — ...".
        plan_by_id = {p.id: p for p in plan}
        for tc_id in ids_before - ids_after:
            reason = next(
                (a for a in actions if a.split(":", 1)[0].strip() == tc_id),
                "отбракован пост-валидатором",
            )
            p = plan_by_id.get(tc_id)
            failed.append({
                "id": tc_id,
                "name": p.name if p else tc_id,
                "type": p.type if p else "",
                "stage": "validation",
                "reason": reason,
            })
    except Exception as exc:
        # Валидатор не должен ронять генерацию — при сбое используем сырой сьют.
        logger.warning(f"  Пост-валидация пропущена из-за ошибки: {exc}")

    # Прикрепляем список потерянных кейсов к сьюту, чтобы run_manager сохранил его
    # в meta.json и показал пользователю (gap «план N → выполнено M» становится явным).
    suite.failed_generations = failed or None

    # Сохраняем финальный результат для отладки
    _save_suite_debug(suite, settings)

    success_rate = len(suite.test_cases) / len(plan) * 100 if plan else 0
    logger.info(
        f"\nТест-сьют сгенерирован: "
        f"{len(suite.test_cases)}/{len(plan)} кейсов "
        f"({success_rate:.0f}% успешной генерации)"
    )
    return suite


def plan_test_suite_with_usage(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> tuple:
    """plan_test_suite + снапшот потреблённых токенов фазы планирования."""
    _usage_reset()
    plan = plan_test_suite(spec, settings, api_key, verbose, progress_callback)
    return plan, _usage_snapshot()


def generate_suite_from_plan_with_usage(
    plan: List[TestPlanItem],
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional[str] = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> tuple:
    """generate_suite_from_plan + снапшот потреблённых токенов фазы генерации."""
    _usage_reset()
    suite = generate_suite_from_plan(plan, spec, settings, api_key, verbose, progress_callback)
    return suite, _usage_snapshot()


def analyze_spec(
    spec: OpenAPISpec,
    settings: AISettings,
    api_key: Optional = None,
    verbose: bool = False,
    progress_callback: Optional[Any] = None,
) -> TestSuite:
    """
    Двухэтапный анализ спецификации: planner → generator (без паузы).

    Тонкая обёртка над plan_test_suite + generate_suite_from_plan — используется CLI
    (main.py), где подтверждение плана не требуется. Web-режим вызывает обе фазы
    раздельно, вставляя между ними подтверждение пользователя.
    """
    # Сбрасываем thread-local аккумулятор токенов для нового прогона
    _usage_reset()

    logger.info(f"\nАнализ через GPTunnel (модель: {settings.model})")
    logger.info(
        f"  Эндпоинтов: {spec.get_endpoint_count()} | "
        f"Целевое число тест-кейсов: {settings.max_test_cases}"
    )

    plan = plan_test_suite(spec, settings, api_key, verbose, progress_callback)
    suite = generate_suite_from_plan(plan, spec, settings, api_key, verbose, progress_callback)
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
