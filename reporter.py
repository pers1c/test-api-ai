"""
Формирование итогового JSON-отчёта о прогоне тестов.

Отчёт сохраняется на диск и отдаётся через web-API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

from models import ReportSummary, TestReport, TestResult


def build_report(
    results: List[TestResult],
    base_url: str,
    spec_file: str,
    config_file: Optional[str] = None,
    spec: Optional[Any] = None,
) -> TestReport:
    """
    Собирает полный TestReport из результатов выполнения тестов.

    Аргументы:
        results:     Список результатов тест-кейсов от runner'а.
        base_url:    Базовый URL тестируемого API.
        spec_file:   Путь к файлу OpenAPI-спецификации (для справки).
        config_file: Путь к использованному файлу конфига (необязательно).
        spec:        OpenAPISpec — если передан, в отчёт добавляется матрица
                     покрытия эндпоинтов (coverage). Без него поле остаётся None.

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

    coverage = None
    if spec is not None:
        try:
            from endpoint_coverage import compute_coverage
            coverage = compute_coverage(results, spec)
        except Exception:
            # Покрытие — необязательная аналитика; её сбой не должен ронять отчёт.
            coverage = None

    return TestReport(
        summary=summary,
        test_cases=results,
        base_url=base_url,
        spec_file=spec_file,
        config_file=config_file,
        coverage=coverage,
    )


# ── Текстовый вывод и экспорт (CLI / CI) ─────────────────────────────────────
def save_report(report: TestReport, path: str) -> None:
    """Сохраняет отчёт в JSON-файл."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")


def print_summary(report: TestReport) -> None:
    """Печатает краткую сводку прогона в stdout (plain text, без зависимостей)."""
    s = report.summary
    print("\n" + "─" * 60)
    print(f"Итог: {s.passed}/{s.total} прошло "
          f"| провалов: {s.failed} | ошибок: {s.errors} | пропущено: {s.skipped}")
    print(f"Время: {s.duration_ms / 1000:.2f}с")
    if report.coverage is not None:
        c = report.coverage
        print(f"Покрытие: {c.tested}/{c.total_endpoints} эндпоинтов тестируются, "
              f"{c.happy_path} с happy-path"
              + (f", {c.flagged} с замечаниями" if c.flagged else ""))
    print("─" * 60)


def print_failures(report: TestReport) -> None:
    """Печатает детали проваленных/ошибочных тест-кейсов."""
    bad = [tc for tc in report.test_cases if tc.status in ("failed", "error")]
    if not bad:
        return
    print(f"\nПровалы ({len(bad)}):")
    for tc in bad:
        print(f"\n  ✕ {tc.test_case_id} [{tc.type}] {tc.test_case_name} — {tc.status}")
        for st in tc.steps_results:
            if st.passed or st.skipped:
                continue
            line = (f"      {st.method} {st.endpoint}: "
                    f"ожидался {st.expected_status}, получен {st.actual_status}")
            print(line)
            if st.error_message:
                print(f"        ошибка: {st.error_message}")
            if st.schema_errors:
                for se in st.schema_errors[:5]:
                    print(f"        схема: {se}")


def _xml_escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def to_junit_xml(report: TestReport) -> str:
    """
    Сериализует отчёт в JUnit XML — для CI (GitLab/GitHub/Jenkins показывают его нативно).
    Один <testcase> на тест-кейс; провал/ошибка/пропуск — соответствующими тегами.
    """
    s = report.summary
    lines: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(
        f'<testsuite name="api-tester" tests="{s.total}" failures="{s.failed}" '
        f'errors="{s.errors}" skipped="{s.skipped}" time="{s.duration_ms / 1000:.3f}">'
    )
    for tc in report.test_cases:
        name = _xml_escape(f"{tc.test_case_id} {tc.test_case_name}")
        classname = _xml_escape(tc.type)
        t = f"{tc.duration_ms / 1000:.3f}"
        lines.append(f'  <testcase name="{name}" classname="{classname}" time="{t}">')
        if tc.status in ("failed", "error"):
            detail_parts: List[str] = []
            for st in tc.steps_results:
                if st.passed or st.skipped:
                    continue
                detail_parts.append(
                    f"{st.method} {st.endpoint}: expected {st.expected_status}, "
                    f"got {st.actual_status}"
                )
                if st.error_message:
                    detail_parts.append(f"  error: {st.error_message}")
                for se in (st.schema_errors or [])[:5]:
                    detail_parts.append(f"  schema: {se}")
            detail = _xml_escape("\n".join(detail_parts) or tc.error_message or tc.status)
            tag = "error" if tc.status == "error" else "failure"
            lines.append(f'    <{tag} message="{tag}">{detail}</{tag}>')
        elif tc.status == "skipped":
            lines.append('    <skipped/>')
        lines.append('  </testcase>')
    lines.append('</testsuite>')
    return "\n".join(lines)
