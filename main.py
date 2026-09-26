"""
CLI-обёртка над пайплайном: генерация (или загрузка готового сьюта) → выполнение →
отчёт. Пригодна для CI: код возврата 0 при полном успехе, 1 при провалах/ошибках;
опциональный экспорт JUnit XML.

Примеры:
  python main.py --spec openapi.json --base-url http://localhost:8000
  python main.py --spec openapi.json --base-url http://localhost:8000 \
                 --suite runs/run_x/suite.json          # перепрогон без LLM
  python main.py --spec openapi.json --base-url http://localhost:8000 \
                 --junit report.xml --output report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ai_analyzer import analyze_spec
from config import load_config
from models import TestSuite
from reporter import (
    build_report,
    print_failures,
    print_summary,
    save_report,
    to_junit_xml,
)
from spec_parser import load_spec
from test_runner import run_test_suite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="api-tester",
        description="LLM-генерация и выполнение тестов REST API по OpenAPI-спецификации.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spec", required=True, metavar="FILE",
                        help="OpenAPI-спецификация (JSON или YAML).")
    parser.add_argument("--base-url", required=True, metavar="URL",
                        help="Базовый URL тестируемого API.")
    parser.add_argument("--suite", default=None, metavar="FILE",
                        help="Готовый suite.json — выполнить его БЕЗ обращения к LLM "
                             "(перепрогон/регрессия). Без него сьют генерируется заново.")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="YAML-конфиг (по умолчанию config.yaml в текущей папке).")
    parser.add_argument("--output", default="results.json", metavar="FILE",
                        help="Куда сохранить JSON-отчёт (по умолчанию results.json).")
    parser.add_argument("--junit", default=None, metavar="FILE",
                        help="Дополнительно сохранить отчёт в JUnit XML (для CI).")
    parser.add_argument("--api-key", default=None, metavar="KEY",
                        help="Ключ GPTunnel (иначе берётся из GPTUNNEL_API_KEY).")
    parser.add_argument("--model", default=None, metavar="MODEL",
                        help="Переопределить модель LLM из конфига.")
    parser.add_argument("--max-tests", type=int, default=None, metavar="N",
                        help="Переопределить лимит числа тест-кейсов.")
    parser.add_argument("--instructions", default=None, metavar="TEXT",
                        help="Доп. инструкции модели (или @path/to/file.txt).")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # 1. Конфиг
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1
    if args.model:
        config.ai_settings.model = args.model
    if args.max_tests is not None:
        config.ai_settings.max_test_cases = args.max_tests
    if args.instructions is not None:
        instr = args.instructions
        if instr.startswith("@") and len(instr) > 1:
            try:
                instr = Path(instr[1:]).read_text(encoding="utf-8")
            except OSError as exc:
                print(f"Не удалось прочитать файл инструкций: {exc}", file=sys.stderr)
                return 1
        config.ai_settings.user_instructions = instr

    # 2. Спека
    try:
        spec = load_spec(args.spec)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка спецификации: {exc}", file=sys.stderr)
        return 1
    print(f"Спека: {spec.title} ({spec.get_endpoint_count()} эндпоинтов)")

    # 3. Сьют: готовый (перепрогон) или генерация через LLM
    if args.suite:
        try:
            suite = TestSuite.model_validate(
                json.loads(Path(args.suite).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            print(f"Не удалось загрузить сьют {args.suite}: {exc}", file=sys.stderr)
            return 1
        print(f"Загружен готовый сьют: {len(suite.test_cases)} кейсов (без обращения к LLM)")
    else:
        print(f"Генерация сьюта через LLM (модель: {config.ai_settings.model})…")
        try:
            suite = analyze_spec(
                spec=spec, settings=config.ai_settings,
                api_key=args.api_key, verbose=args.verbose,
            )
        except Exception as exc:
            print(f"Ошибка генерации: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    if not suite.test_cases:
        print("Сьют пуст — нечего выполнять.", file=sys.stderr)
        return 1

    # 4. Выполнение
    print(f"Выполнение против {args.base_url}…")
    run_start = time.perf_counter()
    try:
        results = run_test_suite(
            suite=suite, base_url=args.base_url, config=config,
            verbose=args.verbose, spec=spec,
        )
    except Exception as exc:
        print(f"Ошибка выполнения: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    print(f"Завершено за {(time.perf_counter() - run_start):.2f}с")

    # 5. Отчёт
    report = build_report(
        results=results,
        base_url=args.base_url,
        spec_file=str(Path(args.spec).resolve()),
        config_file=str(Path(args.config).resolve()) if args.config else None,
        spec=spec,
    )
    print_summary(report)
    print_failures(report)

    try:
        save_report(report, args.output)
        print(f"\nОтчёт сохранён: {args.output}")
        if args.junit:
            Path(args.junit).parent.mkdir(parents=True, exist_ok=True)
            Path(args.junit).write_text(to_junit_xml(report), encoding="utf-8")
            print(f"JUnit XML сохранён: {args.junit}")
    except OSError as exc:
        print(f"Не удалось сохранить отчёт: {exc}", file=sys.stderr)
        return 1

    # 6. Код возврата для CI
    s = report.summary
    return 0 if (s.failed == 0 and s.errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
