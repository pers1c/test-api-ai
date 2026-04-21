"""
Формирование итогового JSON-отчёта о прогоне тестов.

Отчёт сохраняется на диск и отдаётся через web-API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from models import ReportSummary, TestReport, TestResult


def build_report(
    results: List[TestResult],
    base_url: str,
    spec_file: str,
    config_file: Optional[str] = None,
) -> TestReport:
    """
    Собирает полный TestReport из результатов выполнения тестов.

    Аргументы:
        results:     Список результатов тест-кейсов от runner'а.
        base_url:    Базовый URL тестируемого API.
        spec_file:   Путь к файлу OpenAPI-спецификации (для справки).
        config_file: Путь к использованному файлу конфига (необязательно).

    Возвращает:
        Pydantic-модель TestReport, готовую к сериализации.
    """
    total_duration = sum(r.duration_ms for r in results)
    timestamp = datetime.now(timezone.utc).isoformat()

    summary = ReportSummary(
        total=len(results),
        passed=sum(1 for r in results if r.status == "passed"),
        failed=sum(1 for r in results if r.status == "failed"),
        errors=sum(1 for r in results if r.status == "error"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        duration_ms=round(total_duration, 2),
        timestamp=timestamp,
    )

    return TestReport(
        summary=summary,
        test_cases=results,
        base_url=base_url,
        spec_file=spec_file,
        config_file=config_file,
    )
