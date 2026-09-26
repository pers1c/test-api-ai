"""
FastAPI web-приложение для AI-тестировщика API.

Эндпоинты:
  GET  /                         — главная страница (статический HTML)
  GET  /api/runs                 — список всех прогонов
  POST /api/runs                 — создать новый прогон (multipart: spec + параметры)
  POST /api/estimate             — оценить стоимость прогона без запуска
  GET  /api/runs/{id}            — метаданные конкретного прогона
  GET  /api/runs/{id}/events     — SSE-стрим прогресса (live + replay истории)
  GET  /api/runs/{id}/report     — полный JSON-отчёт
  GET  /api/runs/{id}/report/download — отдать отчёт как файл для скачивания
  GET  /api/runs/{id}/suite      — сгенерированный тест-сьют (для отладки)

Запуск:
  export GPTUNNEL_API_KEY=...
  uvicorn web_app:app --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

load_dotenv(override=False)
from config import load_config
from cost_estimator import estimate_run
from run_manager import (
    RUNS_DIR,
    create_rerun,
    create_run,
    get_run,
    list_runs,
    rerun_single_case,
)
from spec_parser import load_spec


app = FastAPI(
    title="AI API Tester",
    description="Web-интерфейс для AI-генерации и выполнения API-тестов.",
)


# Статика и главная страница
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Статика (CSS/JS) монтируется под /static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Отдаёт единственную HTML-страницу SPA.

    На лету подставляет cache-busting query-параметр (?v=<mtime>) к ссылкам
    на app.css и app.js, чтобы браузер при каждом изменении этих файлов
    тянул свежую версию, а не кэш.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Файл {index_path} не найден. Убедитесь, что static/index.html создан.",
        )
    html = index_path.read_text(encoding="utf-8")

    # Cache-busting через mtime файлов: меняется содержимое — меняется URL
    for asset in ("app.css", "app.js"):
        asset_path = STATIC_DIR / asset
        if asset_path.exists():
            version = int(asset_path.stat().st_mtime)
            html = html.replace(
                f"/static/{asset}",
                f"/static/{asset}?v={version}",
            )

    response = HTMLResponse(html)
    # Сам HTML тоже не должен кэшироваться — иначе клиент не увидит обновлённые ?v=
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
async def health() -> dict:
    """Проверка работоспособности + наличия API-ключа в окружении."""
    return {
        "status": "ok",
        "gptunnel_key_present": bool(os.environ.get("GPTUNNEL_API_KEY")),
    }


# API прогонов
@app.get("/api/runs")
async def api_list_runs() -> JSONResponse:
    """Список всех прогонов — из памяти + с диска."""
    return JSONResponse({"runs": list_runs()})


@app.post("/api/runs")
async def api_create_run(
    spec: UploadFile = File(..., description="Файл OpenAPI-спецификации (JSON/YAML)"),
    base_url: str = Form(..., description="Базовый URL тестируемого API"),
    model: Optional[str] = Form(None, description="Модель LLM (необязательно)"),
    max_tests: Optional[int] = Form(None, description="Макс. число тест-кейсов"),
    include_negative: bool = Form(True, description="Включить негативные тесты"),
    include_edge_cases: bool = Form(True, description="Включить граничные кейсы"),
    auth_type: str = Form("none", description="Тип аутентификации для тестируемого API"),
    auth_token: Optional[str] = Form(None, description="Токен/ключ для тестируемого API"),
    auth_header_name: str = Form("Authorization", description="Имя заголовка для api_key"),
    user_instructions: Optional[str] = Form(
        None,
        description="Доп. инструкции для модели (опционально): что обязательно проверить",
    ),
) -> JSONResponse:
    """
    Создаёт новый прогон: принимает файл спеки и параметры, запускает фоновый воркер.
    Возвращает run_id, с которым клиент может подписаться на SSE и опросить статус.
    """
    if not os.environ.get("GPTUNNEL_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail=(
                "GPTUNNEL_API_KEY не задан на сервере. "
                "Задайте переменную окружения перед запуском uvicorn."
            ),
        )

    # Валидация входных данных
    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="base_url должен начинаться с http:// или https://",
        )

    spec_bytes = await spec.read()
    try:
        spec_content = spec_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Файл спецификации должен быть в кодировке UTF-8",
        )

    filename = spec.filename or "openapi.json"
    # Проверяем расширение
    if not filename.lower().endswith((".json", ".yaml", ".yml")):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только .json, .yaml и .yml файлы",
        )

    state = create_run(
        spec_content=spec_content,
        spec_filename=filename,
        base_url=base_url,
        model=model,
        max_tests=max_tests,
        include_negative=include_negative,
        include_edge_cases=include_edge_cases,
        auth_type=auth_type,
        auth_token=auth_token,
        auth_header_name=auth_header_name,
        user_instructions=user_instructions,
    )

    return JSONResponse({"run_id": state.run_id}, status_code=201)


@app.post("/api/runs/{run_id}/rerun")
async def api_run_rerun(run_id: str, payload: Optional[dict] = None) -> JSONResponse:
    """
    Перепрогон сохранённого сьюта исходного прогона БЕЗ обращения к LLM.
    Опционально принимает {"base_url": "..."} — прогнать против другой среды.
    Возвращает run_id нового прогона.
    """
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")

    base_url = (payload or {}).get("base_url")
    if base_url and not str(base_url).startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="base_url должен начинаться с http:// или https://",
        )

    state = create_rerun(run_id, base_url=base_url)
    if state is None:
        raise HTTPException(
            status_code=400,
            detail="У исходного прогона нет сохранённого сьюта (suite.json) для перепрогона",
        )
    return JSONResponse({"run_id": state.run_id}, status_code=201)


@app.post("/api/runs/{run_id}/rerun-case")
async def api_rerun_case(run_id: str, payload: dict) -> JSONResponse:
    """
    Перепрогон ОДНОГО кейса из сохранённого сьюта (повторить / изменить-и-повторить).
    Тело: {"case_id": str, "base_url"?: str, "case"?: {...правка TestCase...}}.
    Возвращает свежий TestResult кейса (без обращения к LLM).
    """
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")

    case_id = payload.get("case_id")
    base_url = payload.get("base_url")
    edited = payload.get("case")
    if not case_id and edited is None:
        raise HTTPException(status_code=400, detail="Нужен case_id или case")
    if base_url and not str(base_url).startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="base_url должен начинаться с http(s)://")

    try:
        result = await rerun_single_case(run_id, case_id, base_url, edited)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail="Кейс или сохранённый сьют не найдены")
    return JSONResponse(result.model_dump())


@app.post("/api/estimate")
async def api_estimate(
    spec: UploadFile = File(..., description="Файл OpenAPI-спецификации"),
    model: Optional[str] = Form(None, description="Модель (по умолчанию — из конфига)"),
    max_tests: Optional[int] = Form(None, description="Макс. число тестов"),
) -> JSONResponse:
    """
    Оценка стоимости прогона без запуска — для UI-превью.
    Парсит спеку, берёт цены из конфига, возвращает CostEstimate.to_dict().
    """
    spec_bytes = await spec.read()
    try:
        spec_content = spec_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Файл спецификации должен быть в UTF-8",
        )

    filename = spec.filename or "openapi.json"
    if not filename.lower().endswith((".json", ".yaml", ".yml")):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только .json, .yaml и .yml",
        )

    # Записываем во временный файл — load_spec работает с путями
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=Path(filename).suffix,
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(spec_content)
        tmp_path = tmp.name

    try:
        parsed = load_spec(tmp_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Ошибка парсинга спеки: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    config = load_config()
    resolved_model = model or config.ai_settings.model
    resolved_max_tests = max_tests if max_tests is not None else config.ai_settings.max_test_cases

    est = estimate_run(
        spec=parsed,
        model=resolved_model,
        max_tests=resolved_max_tests,
        pricing_config=config.pricing,
    )

    return JSONResponse({
        "estimate": est.to_dict(),
        "endpoints": parsed.get_endpoint_count(),
        "spec_title": parsed.title,
    })


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: str) -> JSONResponse:
    """Метаданные прогона и сводка, если он завершён."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return JSONResponse(state.meta_dict())


@app.get("/api/runs/{run_id}/events")
async def api_run_events(run_id: str) -> StreamingResponse:
    """
    SSE-поток событий прогона.

    При подключении сначала отдаём все накопленные события (replay),
    затем подписываемся на очередь для live-обновлений. Это позволяет
    клиенту подключиться в любой момент и не пропустить стадий.
    """
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Сначала реплей: отдаём все уже произошедшие события
        for event in list(state.events_history):
            if event.type == "_end":
                continue
            yield event.to_sse()

        # Если прогон уже завершён (завершился до подписки), шлём финальное событие и выходим
        if state.status in ("completed", "failed"):
            final_type = "complete" if state.status == "completed" else "error"
            final_msg = "Прогон уже был завершён" if state.status == "completed" else (state.error or "Ошибка")
            from run_manager import RunEvent  # локальный импорт во избежание циклов
            yield RunEvent(type=final_type, message=final_msg).to_sse()
            return

        # 2. Live: подписываемся на очередь событий
        while True:
            try:
                event = await asyncio.wait_for(state.queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Периодический keepalive, чтобы прокси/браузер не закрыли соединение
                yield ": keepalive\n\n"
                continue

            if event.type == "_end":
                break
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # отключаем буферизацию на nginx
        },
    )


@app.get("/api/runs/{run_id}/report")
async def api_run_report(run_id: str) -> JSONResponse:
    """Полный отчёт прогона (если завершён)."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")

    # Сначала пробуем из памяти
    if state.report is not None:
        return JSONResponse(state.report.model_dump())

    # Пробуем с диска
    report_path = RUNS_DIR / run_id / "report.json"
    if report_path.exists():
        return JSONResponse(json.loads(report_path.read_text(encoding="utf-8")))

    raise HTTPException(
        status_code=404,
        detail="Отчёт ещё не готов — прогон не завершён или завершился с ошибкой",
    )


@app.get("/api/runs/{run_id}/report/download")
async def api_run_report_download(run_id: str) -> FileResponse:
    """Отдаёт report.json как файл для скачивания."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")

    report_path = RUNS_DIR / run_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Отчёт ещё не сформирован")

    return FileResponse(
        path=str(report_path),
        media_type="application/json",
        filename=f"{run_id}_report.json",
    )


@app.get("/api/runs/{run_id}/suite")
async def api_run_suite(run_id: str) -> JSONResponse:
    """Возвращает сгенерированный тест-сьют (для отладки и просмотра)."""
    suite_path = RUNS_DIR / run_id / "suite.json"
    if not suite_path.exists():
        raise HTTPException(status_code=404, detail="Тест-сьют ещё не сгенерирован")
    return JSONResponse(json.loads(suite_path.read_text(encoding="utf-8")))


@app.get("/api/runs/{run_id}/plan")
async def api_run_plan(run_id: str) -> JSONResponse:
    """Возвращает запланированный тест-сьют (для переподключения к шагу подтверждения)."""
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    if state.plan is None:
        raise HTTPException(status_code=404, detail="План ещё не составлен")
    return JSONResponse({"plan": state.plan, "status": state.status})


@app.post("/api/runs/{run_id}/plan")
async def api_run_plan_decision(run_id: str, decision: dict) -> JSONResponse:
    """
    Решение пользователя по плану на шаге подтверждения.

    Тело JSON:
      {
        "action": "generate" | "cancel" | "replan",
        "excluded_ids": ["tc_003", ...],   // для generate — какие пункты не генерировать
        "extra_instructions": "..."         // для replan — доп. инструкции планировщику
      }
    """
    state = get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Прогон не найден")
    if state.status != "awaiting_confirmation":
        raise HTTPException(
            status_code=409,
            detail=f"Прогон не ожидает подтверждения плана (статус: {state.status})",
        )

    action = (decision or {}).get("action", "generate")
    if action not in ("generate", "cancel", "replan"):
        raise HTTPException(status_code=400, detail=f"Недопустимое действие: {action}")

    state.plan_decision = {
        "action": action,
        "excluded_ids": (decision or {}).get("excluded_ids") or [],
        "extra_instructions": (decision or {}).get("extra_instructions") or "",
    }
    # Будим фоновый воркер, ожидающий на plan_event
    state.plan_event.set()
    return JSONResponse({"ok": True, "action": action})
