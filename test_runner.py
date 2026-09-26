"""
HTTP-исполнитель тестов: выполняет сгенерированный LLM тест-сьют против целевого API.

Все вывод-события передаются через опциональный on_test_complete callback —
ничего не печатается в stdout, так как модуль работает в контексте web-сервера.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from config import build_auth_headers, build_auth_query_params
from context_manager import ContextManager
from models import (
    AppConfig,
    StepResult,
    TestCase,
    TestResult,
    TestStep,
    TestSuite,
)


# Публичная точка входа
async def run_test_suite_async(
    suite: TestSuite,
    base_url: str,
    config: AppConfig,
    on_test_complete: Optional[Any] = None,
    spec: Optional[Any] = None,
) -> List[TestResult]:
    """
    Выполняет все тест-кейсы из сьюта против целевого сервера.

    Аргументы:
        suite:            Сгенерированный тест-сьют от LLM.
        base_url:         Базовый URL API-сервера (например "https://api.example.com").
        config:           Конфигурация приложения (аутентификация, таймауты, задержки).
        on_test_complete: Async callback (index, total, result) -> None,
                          вызывается после завершения каждого тест-кейса.
        spec:             OpenAPISpec — если передан и включена валидация тела ответа,
                          тело успешных ответов проверяется против схемы из спеки.

    Возвращает:
        Список TestResult, по одному на каждый тест-кейс.
    """
    return await _run_suite_async(
        suite, base_url, config,
        on_test_complete=on_test_complete,
        spec=spec,
    )


def run_test_suite(
    suite: TestSuite,
    base_url: str,
    config: AppConfig,
    verbose: bool = False,
    spec: Optional[Any] = None,
) -> List[TestResult]:
    """
    Синхронная обёртка над run_test_suite_async — для CLI и скриптов вне event loop.
    """
    return asyncio.run(
        run_test_suite_async(suite, base_url, config, spec=spec)
    )


async def _run_suite_async(
    suite: TestSuite,
    base_url: str,
    config: AppConfig,
    on_test_complete: Optional[Any] = None,
    spec: Optional[Any] = None,
) -> List[TestResult]:
    """Асинхронная реализация, выполняющая все тест-кейсы последовательно."""
    auth_headers = build_auth_headers(config)
    auth_query_params = build_auth_query_params(config)
    settings = config.test_settings
    results: List[TestResult] = []

    # Нормализуем базовый URL — убираем завершающий слэш
    base_url = base_url.rstrip("/")

    # Защита от задвоенного протокола (например "http:/http://...")
    # Возникает если пользователь случайно передал URL с дублем
    base_url = re.sub(r'^https?:/+(?=https?://)', '', base_url)

    # Проверяем что URL начинается с http:// или https://
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(
            f"Некорректный base-url: '{base_url}'.\n"
            "URL должен начинаться с http:// или https://, "
            "например: http://localhost:8000"
        )

    total = len(suite.test_cases)

    # Уникальный для прогона идентификатор. Доступен всем шагам как {{run_nonce}}.
    # Нужен, чтобы регистрируемые логины/email были уникальны МЕЖДУ прогонами —
    # иначе повторный прогон против персистентной БД получает 400 "username taken".
    run_nonce = uuid.uuid4().hex[:8]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.timeout),
        follow_redirects=True,
        verify=True,
        # Отключаем системный прокси для локальных адресов — иначе httpx на Windows
        # пытается проксировать 127.0.0.1
        proxy=None,
        trust_env=False,
    ) as http_client:
        for i, test_case in enumerate(suite.test_cases, start=1):
            result = await _run_test_case(
                test_case=test_case,
                base_url=base_url,
                http_client=http_client,
                auth_headers=auth_headers,
                auth_query_params=auth_query_params,
                settings=settings,
                run_nonce=run_nonce,
                spec=spec,
            )
            results.append(result)

            # Уведомляем web-UI о завершении тест-кейса
            if on_test_complete is not None:
                try:
                    await on_test_complete(i, total, result)
                except Exception:
                    # Сбои в UI-колбэке не должны ронять прогон
                    pass

            # Задержка между тест-кейсами
            if settings.delay_between_requests > 0:
                await asyncio.sleep(settings.delay_between_requests)

    return results


async def _run_test_case(
    test_case: TestCase,
    base_url: str,
    http_client: httpx.AsyncClient,
    auth_headers: Dict[str, str],
    auth_query_params: Dict[str, str],
    settings: Any,
    run_nonce: str = "",
    spec: Optional[Any] = None,
) -> TestResult:
    """
    Выполняет все шаги одного тест-кейса по порядку.

    Пропускает последующие шаги, если требуемая переменная недоступна
    из-за падения предыдущего шага с извлечением.

    run_nonce — уникальный для прогона маркер, доступный шагам как {{run_nonce}}.
    """
    validate_body = bool(spec is not None and getattr(settings, "validate_response_body", False))
    context = ContextManager(initial={"run_nonce": run_nonce} if run_nonce else None)
    step_results: List[StepResult] = []
    case_start = time.perf_counter()
    overall_status = "passed"

    for step_index, step in enumerate(test_case.steps):
        # Проверяем, доступны ли переменные, от которых зависит этот шаг
        if step.depends_on_vars:
            missing = [v for v in step.depends_on_vars if not context.has(v)]
            if missing:
                skipped_result = StepResult(
                    step_description=step.description,
                    endpoint=step.endpoint,
                    method=step.method,
                    request_url=f"{base_url}{step.endpoint}",
                    request_headers={},
                    actual_status=None,
                    expected_status=step.expected_status,
                    passed=False,
                    duration_ms=0.0,
                    skipped=True,
                    error_message=f"Пропущен: требуемые переменные недоступны: {missing}",
                )
                step_results.append(skipped_result)
                overall_status = "failed"
                continue

        # Подставляем значения шаблонов {{переменная}} в шаг
        resolved_step = context.resolve_step(step)

        step_result = await _run_step(
            step=resolved_step,
            base_url=base_url,
            http_client=http_client,
            context=context,
            auth_headers=auth_headers,
            auth_query_params=auth_query_params,
            spec=spec if validate_body else None,
        )
        step_results.append(step_result)

        if step_result.error_message and not step_result.skipped:
            overall_status = "error"
        elif not step_result.passed and not step_result.skipped:
            overall_status = "failed"

        # Задержка между шагами внутри тест-кейса
        if settings.delay_between_requests > 0 and step_index < len(test_case.steps) - 1:
            await asyncio.sleep(settings.delay_between_requests / 2)

    # Определяем итоговый статус
    if all(sr.skipped for sr in step_results):
        overall_status = "skipped"
    elif overall_status == "passed":
        if not all(sr.passed or sr.skipped for sr in step_results):
            overall_status = "failed"

    duration_ms = (time.perf_counter() - case_start) * 1000

    return TestResult(
        test_case_id=test_case.id,
        test_case_name=test_case.name,
        type=test_case.type,
        status=overall_status,
        steps_results=step_results,
        duration_ms=round(duration_ms, 2),
    )


async def _run_step(
    step: TestStep,
    base_url: str,
    http_client: httpx.AsyncClient,
    context: ContextManager,
    auth_headers: Dict[str, str],
    auth_query_params: Dict[str, str],
    spec: Optional[Any] = None,
) -> StepResult:
    """
    Выполняет один HTTP-запрос и возвращает его результат.

    Объединяет заголовки аутентификации с заголовками шага (заголовки шага имеют приоритет).
    Извлекает переменные из ответа для использования в последующих шагах.

    Если передан spec, тело ответа при совпадении статус-кода проверяется против
    объявленной в спеке схемы (см. response_validator).
    """
    # Формируем полный URL — обрабатываем path-параметры {param}, уже подставленные ранее
    endpoint = step.endpoint
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    request_url = f"{base_url}{endpoint}"

    # Объединяем заголовки: дефолты аутентификации + специфичные для шага (шаг перекрывает auth)
    merged_headers = {**auth_headers, **step.headers}

    # Объединяем query-параметры: дефолты аутентификации + специфичные для шага
    merged_query = {**(auth_query_params or {}), **(step.query_params or {})}

    start_time = time.perf_counter()
    error_message: Optional[str] = None
    response_body: Optional[Any] = None
    response_headers_dict: Optional[Dict[str, str]] = None
    actual_status: Optional[int] = None

    try:
        response = await http_client.request(
            method=step.method,
            url=request_url,
            headers=merged_headers if merged_headers else None,
            params=merged_query if merged_query else None,
            json=step.body if step.body else None,
        )
        actual_status = response.status_code
        response_headers_dict = dict(response.headers)

        # Пробуем распарсить ответ как JSON
        try:
            response_body = response.json()
        except Exception:
            text = response.text
            response_body = text if text else None

    except httpx.TimeoutException as exc:
        error_message = f"Таймаут запроса: {exc}"
    except httpx.ConnectError as exc:
        error_message = f"Ошибка подключения: {exc}"
    except httpx.RequestError as exc:
        error_message = f"Ошибка HTTP-запроса: {exc}"
    except Exception as exc:
        error_message = f"Неожиданная ошибка: {exc}"

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Определяем успех/провал по статус-коду
    status_ok = (actual_status == step.expected_status) if actual_status is not None else False
    passed = status_ok

    # Валидация тела ответа против схемы из спеки (только когда статус совпал —
    # тогда тело должно быть «успешным» и соответствовать объявленной схеме).
    schema_errors: Optional[List[str]] = None
    if spec is not None and status_ok and error_message is None and response_body is not None:
        try:
            from response_validator import validate_response_body
            errs = validate_response_body(
                spec, step.method, step.endpoint, actual_status, response_body
            )
            if errs:
                schema_errors = errs
                passed = False
        except Exception:
            # Валидация схемы не должна ронять прогон — при сбое просто пропускаем.
            schema_errors = None

    # Извлекаем переменные из ответа для последующих шагов.
    # Гейтим по status_ok, а не по passed: если HTTP-вызов успешен, переменные
    # (токен, id) должны извлечься даже при незначительном расхождении тела со
    # схемой — иначе одно нарушение схемы каскадно роняет весь контекстный кейс.
    extracted_values: Optional[Dict[str, str]] = None
    if status_ok and step.extract and response_body is not None:
        try:
            extracted_values = context.extract_values(response_body, step.extract)
        except Exception:
            # Некритично: ошибка извлечения игнорируется, но не провалит шаг
            pass

    # Маскируем чувствительные заголовки в выводе
    safe_headers = _sanitise_headers(merged_headers)

    return StepResult(
        step_description=step.description,
        endpoint=step.endpoint,
        method=step.method,
        request_url=request_url,
        request_headers=safe_headers,
        request_body=step.body,
        actual_status=actual_status,
        expected_status=step.expected_status,
        passed=passed,
        response_body=response_body,
        response_headers=response_headers_dict,
        duration_ms=duration_ms,
        error_message=error_message,
        extracted_values=extracted_values,
        skipped=False,
        schema_errors=schema_errors,
    )


# Вспомогательные функции
def _sanitise_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Маскирует чувствительные значения заголовков (токены) для безопасного хранения в отчётах."""
    sensitive = {"authorization", "x-api-key", "api-key", "x-auth-token"}
    result: Dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in sensitive and value:
            # Показываем только схему/префикс (например "Bearer ****")
            if " " in value:
                scheme, _ = value.split(" ", 1)
                result[name] = f"{scheme} ****"
            else:
                result[name] = "****"
        else:
            result[name] = value
    return result
