from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from ai_analyzer import analyze_spec
from config import load_config
from reporter import build_report, print_failures, print_summary, save_report
from spec_parser import load_spec
from test_runner import run_test_suite

console = Console()

DEFAULT_OUTPUT = "results.json"

# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="api-tester",
        description="Инструмент автоматизированного тестирования REST API на основе Claude AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --spec openapi.yaml --base-url https://api.example.com
  python main.py --spec petstore.json --base-url http://localhost:8080 --verbose
  python main.py --spec api.yaml --base-url https://staging.api.com \\
                 --config config.yaml --output reports/run.json
        """,
    )

    parser.add_argument(
        "--spec",
        required=True,
        metavar="FILE",
        help="Путь к файлу OpenAPI-спецификации (JSON или YAML).",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        metavar="URL",
        help="Базовый URL тестируемого API-сервера (например https://api.example.com).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help=(
            "Путь к YAML-файлу конфигурации. "
            "По умолчанию ищется config.yaml / config.yml / .api-tester.yaml."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Путь для сохранения JSON-отчёта (по умолчанию: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод: запросы, ответы и извлечённые значения по каждому шагу.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help=(
            "API-ключ провайдера LLM. "
            "Переопределяет переменные окружения GPTUNNEL_API_KEY / ANTHROPIC_API_KEY."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        metavar="PROVIDER",
        help="Провайдер LLM: gptunnel | anthropic. Переопределяет значение из конфига.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "Модель LLM. Примеры: gpt-4o, gpt-4o-mini, claude-opus-4-6. "
            "Переопределяет значение из конфига."
        ),
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=None,
        metavar="N",
        help="Переопределить максимальное количество генерируемых тест-кейсов.",
    )
    parser.add_argument(
        "--no-negative",
        action="store_true",
        help="Пропустить генерацию негативных тестов (невалидные входные данные).",
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Пропустить генерацию граничных тест-кейсов.",
    )

    return parser

# Основной пайплайн
def main() -> int:
    """
    Оркестрирует полный пайплайн:
      1. Парсинг аргументов CLI
      2. Загрузка конфигурации
      3. Парсинг OpenAPI-спецификации
      4. Анализ спецификации через Claude AI -> генерация тест-сьюта
      5. Выполнение тест-сьюта против целевого сервера
      6. Формирование и сохранение JSON-отчёта
      7. Вывод сводки в консоль

    Возвращает:
        Код завершения: 0 если все тесты прошли, 1 в противном случае.
    """
    parser = _build_parser()
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]AI-тестировщик API[/bold]",
        border_style="cyan",
    ))

    # --- 1. Конфигурация ---
    console.print(Rule("[dim]Конфигурация[/dim]"))
    try:
        config = load_config(args.config)
        console.print(f"  Провайдер: [cyan]{config.ai_settings.provider.value}[/cyan]  Модель: [cyan]{config.ai_settings.model}[/cyan]")
        console.print(f"  Тип аутентификации: [cyan]{config.auth.type}[/cyan]")
        console.print(f"  Таймаут:            {config.test_settings.timeout}с")
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Ошибка конфигурации:[/bold red] {exc}")
        return 1

    # Применяем переопределения из CLI к настройкам AI
    if args.max_tests is not None:
        config.ai_settings.max_test_cases = args.max_tests
    if args.no_negative:
        config.ai_settings.include_negative_tests = False
    if args.no_edge_cases:
        config.ai_settings.include_edge_cases = False
    if args.provider is not None:
        from models import LLMProvider
        config.ai_settings.provider = LLMProvider(args.provider)
    if args.model is not None:
        config.ai_settings.model = args.model

    # --- 2. OpenAPI-спецификация ---
    console.print(Rule("[dim]OpenAPI-спецификация[/dim]"))
    try:
        spec = load_spec(args.spec)
        console.print(spec.get_summary_for_display())
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Ошибка спецификации:[/bold red] {exc}")
        return 1

    # --- 3. Анализ через AI ---
    console.print(Rule("[dim]Анализ AI[/dim]"))
    try:
        suite = analyze_spec(
            spec=spec,
            settings=config.ai_settings,
            api_key=args.api_key,
            verbose=args.verbose,
        )
    except Exception as exc:
        console.print(f"[bold red]Ошибка анализа AI:[/bold red] {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if not suite.test_cases:
        console.print("[yellow]Предупреждение: Claude не сгенерировал ни одного тест-кейса.[/yellow]")
        return 0

    # Выводим разбивку по типам тест-кейсов
    type_counts: dict[str, int] = {}
    for tc in suite.test_cases:
        type_counts[tc.type] = type_counts.get(tc.type, 0) + 1
    for tc_type, count in sorted(type_counts.items()):
        console.print(f"  {tc_type}: [bold]{count}[/bold]")

    # --- 4. Выполнение тестов ---
    console.print(Rule("[dim]Выполнение тестов[/dim]"))
    console.print(f"  Цель: [bold cyan]{args.base_url}[/bold cyan]")

    run_start = time.perf_counter()
    try:
        results = run_test_suite(
            suite=suite,
            base_url=args.base_url,
            config=config,
            verbose=args.verbose,
        )
    except Exception as exc:
        console.print(f"[bold red]Ошибка выполнения:[/bold red] {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    run_duration = (time.perf_counter() - run_start) * 1000
    console.print(f"\n  Завершено за [bold]{run_duration / 1000:.2f}с[/bold]")

    # --- 5. Отчёт ---
    console.print(Rule("[dim]Отчёт[/dim]"))
    report = build_report(
        results=results,
        base_url=args.base_url,
        spec_file=str(Path(args.spec).resolve()),
        config_file=str(Path(args.config).resolve()) if args.config else None,
    )

    print_summary(report)

    if args.verbose or any(r.status in ("failed", "error") for r in results):
        print_failures(report)

    try:
        save_report(report, args.output)
    except OSError as exc:
        console.print(f"[bold red]Не удалось сохранить отчёт:[/bold red] {exc}")
        return 1

    # --- Код завершения ---
    all_passed = report.summary.failed == 0 and report.summary.errors == 0
    if all_passed:
        console.print("\n[bold green]Все тесты прошли успешно.[/bold green]")
        return 0
    else:
        console.print(
            f"\n[bold red]{report.summary.failed} тест(ов) провалено, "
            f"{report.summary.errors} ошибок.[/bold red]"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())