from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from models import ReportSummary, StepResult, TestReport, TestResult

console = Console()

# Формирование отчёта
def build_report(
    results: List[TestResult],
    base_url: str,
    spec_file: str,
    config_file: str = None,
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

# Сохранение файла
def save_report(report: TestReport, output_path: str) -> None:
    """
    Сериализует отчёт в JSON-файл.

    Аргументы:
        report:      Полный отчёт о тестировании.
        output_path: Путь к файлу назначения (создаётся или перезаписывается).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False, default=str)

    console.print(f"\n[bold green]Отчёт сохранён:[/bold green] {path.resolve()}")

# Вывод в консоль
def print_summary(report: TestReport) -> None:
    """
    Выводит форматированную таблицу результатов и панель статистики в консоль.

    Аргументы:
        report: Полный отчёт о тестировании.
    """
    summary = report.summary

    # --- Таблица по тест-кейсам ---
    table = Table(
        title="Результаты тестирования",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold white",
        expand=False,
    )
    table.add_column("ID", style="dim", width=8)
    table.add_column("Название", style="white", min_width=28)
    table.add_column("Тип", style="cyan", width=12)
    table.add_column("Статус", width=9)
    table.add_column("Шаги", justify="right", width=7)
    table.add_column("Время", justify="right", width=10)
    table.add_column("Детали", style="dim", min_width=30)

    for result in report.test_cases:
        status_cell = _status_cell(result.status)
        details = _result_details(result)

        table.add_row(
            result.test_case_id,
            result.test_case_name,
            result.type,
            status_cell,
            str(len(result.steps_results)),
            f"{result.duration_ms:.0f} мс",
            details,
        )

    console.print()
    console.print(table)

    # --- Панель сводки ---
    pass_rate = (
        f"{(summary.passed / summary.total * 100):.1f}%"
        if summary.total > 0
        else "Н/Д"
    )
    duration_s = summary.duration_ms / 1000

    lines = [
        f"[bold]Всего:[/bold]     {summary.total}",
        f"[bold green]Успешно:[/bold green]   {summary.passed}",
        f"[bold red]Провалено:[/bold red] {summary.failed}",
        f"[bold yellow]Ошибки:[/bold yellow]    {summary.errors}",
        f"[dim]Пропущено:[/dim] {summary.skipped}",
        "",
        f"[bold]Успешность:[/bold] {pass_rate}",
        f"[bold]Время:[/bold]      {duration_s:.2f}с",
        f"[dim]Дата:[/dim]       {summary.timestamp}",
    ]

    panel_style = "green" if summary.failed == 0 and summary.errors == 0 else "red"
    console.print(
        Panel(
            "\n".join(lines),
            title="Сводка",
            border_style=panel_style,
            expand=False,
        )
    )


def print_failures(report: TestReport) -> None:
    """
    Выводит детальную информацию по всем непройденным тест-кейсам.

    Аргументы:
        report: Полный отчёт о тестировании.
    """
    failing = [r for r in report.test_cases if r.status in ("failed", "error")]
    if not failing:
        return

    console.print("\n[bold red]Детали ошибок[/bold red]")
    console.rule(style="red")

    for result in failing:
        console.print(
            f"\n[bold]{result.test_case_id}[/bold] | "
            f"[red]{result.status.upper()}[/red] | "
            f"{result.test_case_name}"
        )
        if result.error_message:
            console.print(f"  Ошибка: [red]{result.error_message}[/red]")

        for i, step in enumerate(result.steps_results, start=1):
            if step.skipped:
                console.print(
                    f"  Шаг {i}: [yellow]ПРОПУЩЕН[/yellow] — {step.step_description}"
                )
                if step.error_message:
                    console.print(f"    Причина: [dim]{step.error_message}[/dim]")
            elif not step.passed:
                console.print(
                    f"  Шаг {i}: [red]ПРОВАЛ[/red] — {step.step_description}"
                )
                console.print(
                    f"    {step.method} {step.request_url}"
                )
                console.print(
                    f"    Ожидался: [bold]{step.expected_status}[/bold]  "
                    f"Получен: [bold red]{step.actual_status}[/bold red]"
                )
                if step.error_message:
                    console.print(f"    Ошибка: [dim]{step.error_message}[/dim]")
                if step.response_body is not None:
                    body_preview = _truncate(str(step.response_body), 300)
                    console.print(f"    Ответ: [dim]{body_preview}[/dim]")

# Вспомогательные функции
def _status_cell(status: str) -> str:
    """Возвращает строку статуса с Rich-форматированием и цветом."""
    mapping = {
        "passed":  "[bold green]УСПЕХ[/bold green]",
        "failed":  "[bold red]ПРОВАЛ[/bold red]",
        "error":   "[bold yellow]ОШИБКА[/bold yellow]",
        "skipped": "[dim]ПРОПУЩЕН[/dim]",
    }
    return mapping.get(status, status.upper())


def _result_details(result: TestResult) -> str:
    """Формирует компактную однострочную строку деталей для таблицы результатов."""
    if result.status == "passed":
        return "все шаги успешны"

    failed_steps = [
        s for s in result.steps_results
        if not s.passed and not s.skipped
    ]
    skipped_steps = [s for s in result.steps_results if s.skipped]

    parts: list[str] = []
    if failed_steps:
        step = failed_steps[0]
        parts.append(
            f"шаг провален: ожидался {step.expected_status}, "
            f"получен {step.actual_status}"
        )
    if skipped_steps:
        parts.append(f"{len(skipped_steps)} шаг(ов) пропущено")
    if result.error_message:
        parts.append(_truncate(result.error_message, 60))

    return "; ".join(parts) if parts else result.status


def _truncate(text: str, max_len: int) -> str:
    """Обрезает строку до max_len символов, добавляя многоточие при необходимости."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
