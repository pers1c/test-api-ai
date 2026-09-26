"""
Фоновое выполнение прогонов тестирования и шина событий для SSE.

Архитектура:
  - Каждый запуск имеет уникальный run_id и свою asyncio.Queue для событий
  - Воркер (run_pipeline) шлёт события в очередь по мере прогресса
  - SSE-эндпоинт подписывается на очередь и стримит события клиенту
  - Артефакты (spec, suite, report) сохраняются в runs/<run_id>/

События имеют тип (stage, progress, log, complete, error) и произвольные данные.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_analyzer import (
    TestPlanItem,
    generate_suite_from_plan_with_usage,
    plan_test_suite_with_usage,
)
from config import load_config
from cost_estimator import compute_actual_cost, estimate_run
from models import AppConfig, TestCase, TestReport, TestResult, TestSuite
from reporter import build_report
from spec_parser import OpenAPISpec, load_spec
from test_runner import run_test_suite_async


# Корневая директория для хранения прогонов
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


@dataclass
class RunEvent:
    """Одно событие прогресса прогона."""

    type: str              # stage | log | progress | complete | error
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Форматирует событие в SSE-формат: 'data: {...}\\n\\n'."""
        payload = asdict(self)
        return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@dataclass
class RunState:
    """Текущее состояние прогона в памяти."""

    run_id: str
    status: str                          # pending | running | awaiting_confirmation | completed | failed | cancelled
    created_at: str
    base_url: str
    spec_filename: str
    model: str
    max_tests: int
    queue: asyncio.Queue[RunEvent] = field(default_factory=asyncio.Queue)
    events_history: List[RunEvent] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    report: Optional[TestReport] = None
    error: Optional[str] = None
    cost: Optional[Dict[str, Any]] = None   # {"estimated": {...}, "actual": {...}}
    # Подтверждение плана: пайплайн встаёт на паузу после планирования и ждёт решения
    plan: Optional[List[Dict[str, Any]]] = None        # сериализованный план для UI
    plan_event: asyncio.Event = field(default_factory=asyncio.Event)
    plan_decision: Optional[Dict[str, Any]] = None     # {action, excluded_ids, extra_instructions}
    # Пункты плана, не дошедшие до выполнения (провал генерации/отбраковка валидатором).
    # Делает явным разрыв «в плане N кейсов → выполнено M». Элемент: {id,name,type,stage,reason}.
    generation_dropped: Optional[List[Dict[str, Any]]] = None

    def meta_dict(self) -> Dict[str, Any]:
        """Возвращает сериализуемое краткое описание для списков и карточек."""
        d: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "base_url": self.base_url,
            "spec_filename": self.spec_filename,
            "model": self.model,
            "max_tests": self.max_tests,
        }
        if self.report is not None:
            d["summary"] = self.report.summary.model_dump()
        if self.error:
            d["error"] = self.error
        if self.cost:
            d["cost"] = self.cost
        if self.plan is not None:
            d["plan"] = self.plan
        if self.generation_dropped:
            d["generation_dropped"] = self.generation_dropped
        return d


# Глобальный реестр активных/завершённых прогонов.
# Для продакшена лучше заменить на БД, но для локального инструмента этого хватает.
_RUNS: Dict[str, RunState] = {}


def list_runs() -> List[Dict[str, Any]]:
    """Возвращает список всех прогонов (в памяти + подхваченные с диска)."""
    # Подхватываем прогоны с диска, если их нет в памяти (после рестарта)
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_id in _RUNS:
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Восстанавливаем только основные поля — задачу не восстанавливаем
            state = RunState(
                run_id=run_id,
                status=meta.get("status", "completed"),
                created_at=meta.get("created_at", ""),
                base_url=meta.get("base_url", ""),
                spec_filename=meta.get("spec_filename", ""),
                model=meta.get("model", ""),
                max_tests=meta.get("max_tests", 0),
                error=meta.get("error"),
                cost=meta.get("cost"),
            )
            # Пробуем загрузить отчёт с диска
            report_path = run_dir / "report.json"
            if report_path.exists():
                try:
                    report_data = json.loads(report_path.read_text(encoding="utf-8"))
                    state.report = TestReport.model_validate(report_data)
                except Exception:
                    pass
            _RUNS[run_id] = state
        except Exception:
            continue

    # Сортируем по времени создания (новые сверху)
    runs = sorted(_RUNS.values(), key=lambda s: s.created_at, reverse=True)
    return [r.meta_dict() for r in runs]


def get_run(run_id: str) -> Optional[RunState]:
    """Возвращает состояние прогона по id, подгружая с диска при необходимости."""
    if run_id in _RUNS:
        return _RUNS[run_id]

    # Пробуем подгрузить с диска
    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        state = RunState(
            run_id=run_id,
            status=meta.get("status", "completed"),
            created_at=meta.get("created_at", ""),
            base_url=meta.get("base_url", ""),
            spec_filename=meta.get("spec_filename", ""),
            model=meta.get("model", ""),
            max_tests=meta.get("max_tests", 0),
            error=meta.get("error"),
            cost=meta.get("cost"),
        )
        report_path = run_dir / "report.json"
        if report_path.exists():
            try:
                report_data = json.loads(report_path.read_text(encoding="utf-8"))
                state.report = TestReport.model_validate(report_data)
            except Exception:
                pass
        _RUNS[run_id] = state
        return state
    except Exception:
        return None


def create_run(
    spec_content: str,
    spec_filename: str,
    base_url: str,
    model: Optional[str] = None,
    max_tests: Optional[int] = None,
    include_negative: bool = True,
    include_edge_cases: bool = True,
    auth_type: str = "none",
    auth_token: Optional[str] = None,
    auth_header_name: str = "Authorization",
    user_instructions: Optional[str] = None,
) -> RunState:
    """
    Создаёт запись прогона, сохраняет спеку на диск и запускает фоновый воркер.
    Возвращает RunState сразу, не дожидаясь завершения.
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем спеку
    spec_path = run_dir / spec_filename
    spec_path.write_text(spec_content, encoding="utf-8")

    # Загружаем конфиг с сервера, применяем переопределения из формы
    config = load_config()
    if model:
        config.ai_settings.model = model
    if max_tests is not None:
        config.ai_settings.max_test_cases = max_tests
    config.ai_settings.include_negative_tests = include_negative
    config.ai_settings.include_edge_cases = include_edge_cases
    if user_instructions is not None and user_instructions.strip():
        config.ai_settings.user_instructions = user_instructions.strip()

    # Аутентификация для тестируемого API
    from models import AuthConfig  # локальный импорт, чтобы избежать циклов
    config.auth = AuthConfig(
        type=auth_type,  # type: ignore[arg-type]
        token=auth_token,
        header_name=auth_header_name,
    )

    state = RunState(
        run_id=run_id,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
        base_url=base_url,
        spec_filename=spec_filename,
        model=config.ai_settings.model,
        max_tests=config.ai_settings.max_test_cases,
    )
    _RUNS[run_id] = state

    # Сразу пишем meta.json — чтобы run был виден даже до старта
    _save_meta(state)

    # Стартуем фоновый воркер
    state.task = asyncio.create_task(
        _run_pipeline(state, spec_path, config)
    )

    return state


async def _emit(state: RunState, event: RunEvent) -> None:
    """Отправляет событие в очередь прогона и сохраняет его в истории."""
    state.events_history.append(event)
    await state.queue.put(event)


async def _run_pipeline(
    state: RunState,
    spec_path: Path,
    config: AppConfig,
) -> None:
    """
    Основной фоновый воркер: парсинг спеки → генерация тестов → выполнение → отчёт.
    Все значимые шаги сопровождаются событиями в очередь.
    """
    run_dir = RUNS_DIR / state.run_id
    state.status = "running"
    _save_meta(state)

    try:
        # --- Парсинг спеки ---
        await _emit(state, RunEvent(
            type="stage",
            message="Парсинг OpenAPI-спецификации",
            data={"stage": "parsing"},
        ))
        spec: OpenAPISpec = load_spec(str(spec_path))
        await _emit(state, RunEvent(
            type="log",
            message=f"Спека загружена: {spec.title} ({spec.get_endpoint_count()} эндпоинтов)",
            data={"title": spec.title, "endpoints": spec.get_endpoint_count()},
        ))

        # --- Оценка стоимости до запуска ---
        estimate = estimate_run(
            spec=spec,
            model=config.ai_settings.model,
            max_tests=config.ai_settings.max_test_cases,
            pricing_config=config.pricing,
        )
        state.cost = {"estimated": estimate.to_dict()}
        _save_meta(state)

        if estimate.price_known:
            await _emit(state, RunEvent(
                type="log",
                message=(
                    f"Ожидаемая стоимость: ~{estimate.cost_rub:.2f} ₽ "
                    f"(~{estimate.total_tokens:,} токенов)"
                ),
                data={"estimate": estimate.to_dict()},
            ))
        else:
            await _emit(state, RunEvent(
                type="log",
                message=(
                    f"Цена модели '{config.ai_settings.model}' неизвестна — "
                    f"оценка стоимости недоступна (добавьте модель в pricing.models в конфиге)"
                ),
            ))

        # Фаза планирования и генерации выполняются синхронно (httpx.Client),
        # поэтому запускаем их в thread executor, чтобы не блокировать event loop.
        loop = asyncio.get_running_loop()
        progress_callback = _make_sync_progress_callback(state, loop)

        # --- Этап планирования (LLM + детерминированный догенератор покрытия) ---
        await _emit(state, RunEvent(
            type="stage",
            message="Планирование тест-сьюта",
            data={"stage": "planning"},
        ))

        plan_items, plan_usage = await loop.run_in_executor(
            None,
            lambda: plan_test_suite_with_usage(
                spec=spec,
                settings=config.ai_settings,
                api_key=None,  # возьмётся из env GPTUNNEL_API_KEY
                verbose=False,
                progress_callback=progress_callback,
            ),
        )

        # --- Подтверждение плана пользователем (пауза пайплайна) ---
        # Может крутить цикл «перепланировать», аккумулируя usage планирования.
        plan_items, plan_usage = await _await_plan_confirmation(
            state, spec, config, loop, progress_callback, plan_items, plan_usage
        )
        if plan_items is None:
            # Пользователь отменил прогон
            state.status = "cancelled"
            _save_meta(state)
            await _emit(state, RunEvent(
                type="complete",
                message="Прогон отменён пользователем на этапе подтверждения плана",
                data={"cancelled": True},
            ))
            return

        # --- Этап генерации тест-кейсов ---
        await _emit(state, RunEvent(
            type="stage",
            message="Генерация тест-сьюта через LLM",
            data={"stage": "generating"},
        ))

        suite, gen_usage = await loop.run_in_executor(
            None,
            lambda: generate_suite_from_plan_with_usage(
                plan=plan_items,
                spec=spec,
                settings=config.ai_settings,
                api_key=None,
                verbose=False,
                progress_callback=progress_callback,
            ),
        )

        # Суммарное потребление токенов = планирование + генерация
        usage = {
            "input_tokens":  plan_usage["input_tokens"] + gen_usage["input_tokens"],
            "output_tokens": plan_usage["output_tokens"] + gen_usage["output_tokens"],
            "calls":         plan_usage["calls"] + gen_usage["calls"],
        }

        # --- Фактическая стоимость (по usage из ответов GPTunnel) ---
        actual_cost = compute_actual_cost(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            model=config.ai_settings.model,
            pricing_config=config.pricing,
        )
        state.cost = {
            "estimated": estimate.to_dict(),
            "actual": actual_cost.to_dict(),
        }
        _save_meta(state)

        if actual_cost.price_known:
            await _emit(state, RunEvent(
                type="log",
                message=(
                    f"Фактическая стоимость генерации: {actual_cost.cost_rub:.2f} ₽ "
                    f"({actual_cost.total_tokens:,} токенов, {usage['calls']} вызовов)"
                ),
                data={"actual": actual_cost.to_dict()},
            ))

        # Сохраняем сгенерированный сьют для просмотра/отладки
        (run_dir / "suite.json").write_text(
            suite.model_dump_json(indent=2),
            encoding="utf-8",
        )

        await _emit(state, RunEvent(
            type="log",
            message=f"Сгенерировано тестов: {len(suite.test_cases)}",
            data={"test_count": len(suite.test_cases)},
        ))

        # Если часть плана не дошла до выполнения — фиксируем это явно (в meta.json и UI),
        # чтобы разрыв «в плане N → выполнено M» не выглядел как тихая потеря тестов.
        dropped = getattr(suite, "failed_generations", None)
        if dropped:
            state.generation_dropped = dropped
            _save_meta(state)
            await _emit(state, RunEvent(
                type="log",
                message=(
                    f"⚠ {len(dropped)} кейс(ов) из плана не дошли до выполнения "
                    f"(см. generation_dropped в meta.json): "
                    f"{', '.join(d['id'] for d in dropped)}"
                ),
                data={"generation_dropped": dropped},
            ))

        # --- Выполнение тестов ---
        await _emit(state, RunEvent(
            type="stage",
            message="Выполнение тестов",
            data={"stage": "executing", "total": len(suite.test_cases)},
        ))

        async def on_test_progress(idx: int, total: int, result: TestResult) -> None:
            await _emit(state, RunEvent(
                type="progress",
                message=f"[{idx}/{total}] {result.test_case_id}: {result.status}",
                data={
                    "index": idx,
                    "total": total,
                    "test_case_id": result.test_case_id,
                    "test_case_name": result.test_case_name,
                    "status": result.status,
                    "duration_ms": result.duration_ms,
                },
            ))

        results: List[TestResult] = await run_test_suite_async(
            suite=suite,
            base_url=state.base_url,
            config=config,
            on_test_complete=on_test_progress,
            spec=spec,
        )

        # --- Формирование отчёта ---
        report = build_report(
            results=results,
            base_url=state.base_url,
            spec_file=str(spec_path),
            spec=spec,
        )
        # Прикрепляем стоимость к отчёту
        report.cost = state.cost
        state.report = report

        # Сохраняем отчёт на диск
        (run_dir / "report.json").write_text(
            json.dumps(report.model_dump(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        state.status = "completed"
        _save_meta(state)

        await _emit(state, RunEvent(
            type="complete",
            message="Прогон завершён",
            data={"summary": report.summary.model_dump()},
        ))

    except Exception as exc:
        state.status = "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        _save_meta(state)

        tb = traceback.format_exc()
        await _emit(state, RunEvent(
            type="error",
            message=str(exc),
            data={"traceback": tb},
        ))

    finally:
        # Сигнализируем конец потока событий — подписчики корректно закроют SSE
        await state.queue.put(RunEvent(type="_end", message=""))


def create_rerun(source_run_id: str, base_url: Optional[str] = None) -> Optional[RunState]:
    """
    Создаёт новый прогон, ВЫПОЛНЯЯ уже сгенерированный сьют исходного прогона
    без повторного обращения к LLM (регрессия / прогон против другой среды).

    Возвращает RunState нового прогона или None, если у источника нет suite.json.
    """
    src_dir = RUNS_DIR / source_run_id
    suite_src = src_dir / "suite.json"
    meta_path = src_dir / "meta.json"
    if not suite_src.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    spec_filename = meta.get("spec_filename") or "openapi.json"
    spec_src = src_dir / spec_filename
    if not spec_src.exists():
        return None

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Переносим спеку и готовый сьют в новый прогон
    (run_dir / spec_filename).write_text(spec_src.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "suite.json").write_text(suite_src.read_text(encoding="utf-8"), encoding="utf-8")

    config = load_config()
    target_url = base_url or meta.get("base_url", "")

    state = RunState(
        run_id=run_id,
        status="pending",
        created_at=datetime.now(timezone.utc).isoformat(),
        base_url=target_url,
        spec_filename=spec_filename,
        model=meta.get("model", ""),
        max_tests=meta.get("max_tests", 0),
    )
    _RUNS[run_id] = state
    _save_meta(state)

    state.task = asyncio.create_task(
        _run_rerun_pipeline(state, run_dir / spec_filename, run_dir / "suite.json", config)
    )
    return state


async def _run_rerun_pipeline(
    state: RunState,
    spec_path: Path,
    suite_path: Path,
    config: AppConfig,
) -> None:
    """Лёгкий пайплайн: загрузить сохранённый сьют → выполнить → отчёт (без LLM)."""
    run_dir = RUNS_DIR / state.run_id
    state.status = "running"
    _save_meta(state)

    try:
        await _emit(state, RunEvent(
            type="stage", message="Перепрогон: загрузка сохранённого сьюта",
            data={"stage": "loading", "rerun": True},
        ))
        spec: OpenAPISpec = load_spec(str(spec_path))
        suite = TestSuite.model_validate(
            json.loads(suite_path.read_text(encoding="utf-8"))
        )
        await _emit(state, RunEvent(
            type="log",
            message=f"Сьют загружен: {len(suite.test_cases)} кейсов (без обращения к LLM)",
            data={"test_count": len(suite.test_cases)},
        ))

        await _emit(state, RunEvent(
            type="stage", message="Выполнение тестов",
            data={"stage": "executing", "total": len(suite.test_cases)},
        ))

        async def on_test_progress(idx: int, total: int, result: TestResult) -> None:
            await _emit(state, RunEvent(
                type="progress",
                message=f"[{idx}/{total}] {result.test_case_id}: {result.status}",
                data={
                    "index": idx, "total": total,
                    "test_case_id": result.test_case_id,
                    "test_case_name": result.test_case_name,
                    "status": result.status,
                    "duration_ms": result.duration_ms,
                },
            ))

        results: List[TestResult] = await run_test_suite_async(
            suite=suite,
            base_url=state.base_url,
            config=config,
            on_test_complete=on_test_progress,
            spec=spec,
        )

        report = build_report(
            results=results, base_url=state.base_url,
            spec_file=str(spec_path), spec=spec,
        )
        state.report = report
        (run_dir / "report.json").write_text(
            json.dumps(report.model_dump(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        state.status = "completed"
        _save_meta(state)
        await _emit(state, RunEvent(
            type="complete", message="Перепрогон завершён",
            data={"summary": report.summary.model_dump()},
        ))
    except Exception as exc:
        state.status = "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        _save_meta(state)
        await _emit(state, RunEvent(
            type="error", message=str(exc),
            data={"traceback": traceback.format_exc()},
        ))
    finally:
        await state.queue.put(RunEvent(type="_end", message=""))


async def rerun_single_case(
    source_run_id: str,
    case_id: Optional[str],
    base_url: Optional[str] = None,
    edited_case: Optional[Dict[str, Any]] = None,
) -> Optional[TestResult]:
    """
    Выполняет ОДИН тест-кейс из сохранённого сьюта (для «повторить» / «изменить и
    повторить»). Если передан edited_case — выполняется он (правка пользователя),
    иначе берётся кейс case_id из suite.json. Без обращения к LLM.

    Возвращает свежий TestResult этого кейса или None, если кейс/сьют не найдены.
    """
    src_dir = RUNS_DIR / source_run_id
    suite_path = src_dir / "suite.json"
    meta_path = src_dir / "meta.json"
    if not suite_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    spec_filename = meta.get("spec_filename") or "openapi.json"
    spec_path = src_dir / spec_filename
    if not spec_path.exists():
        return None

    spec: OpenAPISpec = load_spec(str(spec_path))
    suite = TestSuite.model_validate(json.loads(suite_path.read_text(encoding="utf-8")))

    if edited_case is not None:
        try:
            case = TestCase.model_validate(edited_case)
        except Exception as exc:
            raise ValueError(f"Некорректный формат кейса: {exc}") from exc
    else:
        case = next((c for c in suite.test_cases if c.id == case_id), None)
    if case is None:
        return None

    config = load_config()
    one_case_suite = TestSuite(
        test_cases=[case],
        spec_title=suite.spec_title,
        spec_version=suite.spec_version,
    )
    results = await run_test_suite_async(
        suite=one_case_suite,
        base_url=base_url or meta.get("base_url", ""),
        config=config,
        spec=spec,
    )
    return results[0] if results else None


def _compute_plan_coverage(
    plan_items: List[TestPlanItem], spec: OpenAPISpec
) -> Dict[str, Any]:
    """Считает покрытие плана: сколько операций спеки задействовано из общего числа."""
    from ai_analyzer import _iter_plan_item_ops  # локальный импорт во избежание циклов

    http_methods = {"get", "post", "put", "delete", "patch"}
    all_ops = {
        (m.upper(), p)
        for p, pi in spec.paths.items()
        if isinstance(pi, dict)
        for m in pi
        if m in http_methods
    }
    covered = set()
    for item in plan_items:
        for m, p, _op in _iter_plan_item_ops(item, spec):
            covered.add((m.upper(), p))
    covered &= all_ops
    uncovered = sorted(f"{m} {p}" for m, p in (all_ops - covered))
    return {"covered": len(covered), "total": len(all_ops), "uncovered": uncovered}


async def _await_plan_confirmation(
    state: RunState,
    spec: OpenAPISpec,
    config: AppConfig,
    loop: asyncio.AbstractEventLoop,
    progress_callback: Any,
    plan_items: List[TestPlanItem],
    plan_usage: Dict[str, int],
) -> tuple:
    """
    Ставит пайплайн на паузу после планирования и ждёт решения пользователя.

    Возвращает (plan_items, accumulated_plan_usage):
      - plan_items=None  → пользователь отменил прогон;
      - иначе            → отфильтрованный по excluded_ids список пунктов к генерации.

    Поддерживает цикл «перепланировать»: при action="replan" перезапускает планирование
    с обновлёнными user_instructions и снова ждёт подтверждения, суммируя usage.
    """
    while True:
        # Публикуем план и переходим в состояние ожидания подтверждения
        state.plan = [item.to_dict() for item in plan_items]
        state.status = "awaiting_confirmation"
        state.plan_decision = None
        state.plan_event.clear()
        _save_meta(state)

        coverage = _compute_plan_coverage(plan_items, spec)
        await _emit(state, RunEvent(
            type="plan_ready",
            message=(
                f"План готов: {len(plan_items)} тест-кейсов, "
                f"покрытие {coverage['covered']}/{coverage['total']} эндпоинтов. "
                f"Подтвердите генерацию."
            ),
            data={"plan": state.plan, "coverage": coverage},
        ))

        # Ждём, пока POST /api/runs/{id}/plan выставит решение и разбудит нас
        await state.plan_event.wait()
        decision = state.plan_decision or {"action": "generate"}
        action = decision.get("action", "generate")

        if action == "cancel":
            return None, plan_usage

        if action == "replan":
            extra = (decision.get("extra_instructions") or "").strip()
            if extra:
                existing = (config.ai_settings.user_instructions or "").strip()
                config.ai_settings.user_instructions = (
                    f"{existing}\n{extra}".strip() if existing else extra
                )
            state.status = "running"
            _save_meta(state)
            await _emit(state, RunEvent(
                type="stage",
                message="Перепланирование тест-сьюта",
                data={"stage": "planning"},
            ))
            plan_items, usage2 = await loop.run_in_executor(
                None,
                lambda: plan_test_suite_with_usage(
                    spec=spec,
                    settings=config.ai_settings,
                    api_key=None,
                    verbose=False,
                    progress_callback=progress_callback,
                ),
            )
            plan_usage = {
                "input_tokens":  plan_usage["input_tokens"] + usage2["input_tokens"],
                "output_tokens": plan_usage["output_tokens"] + usage2["output_tokens"],
                "calls":         plan_usage["calls"] + usage2["calls"],
            }
            continue

        # action == "generate": применяем исключения и выходим из цикла
        excluded = set(decision.get("excluded_ids") or [])
        if excluded:
            plan_items = [it for it in plan_items if it.id not in excluded]
        state.status = "running"
        _save_meta(state)
        return plan_items, plan_usage


def _make_sync_progress_callback(state: RunState, loop: asyncio.AbstractEventLoop):
    """
    Адаптер: синхронный callback, который безопасно планирует async-_emit в event loop.
    analyze_spec работает в thread executor и не может напрямую await'ить.
    """
    def callback(event_type: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        event = RunEvent(type=event_type, message=message, data=data)
        # run_coroutine_threadsafe безопасно вызывается из другого потока
        asyncio.run_coroutine_threadsafe(_emit(state, event), loop)
    return callback


def _save_meta(state: RunState) -> None:
    """Сохраняет метаданные прогона на диск (для восстановления после рестарта)."""
    run_dir = RUNS_DIR / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    meta_path.write_text(
        json.dumps(state.meta_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
