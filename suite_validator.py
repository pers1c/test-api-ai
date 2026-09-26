"""
Детерминированный пост-валидатор сгенерированного тест-сьюта.

Работает БЕЗ обращения к LLM, между генерацией и выполнением. Ловит и устраняет
характерные ошибки генерации, из-за которых тест падает не по вине тестируемого API,
а по вине самого тест-кейса:

  R1. Неуникальные учётки между кейсами. Модель часто переиспользует один и тот же
      базовый логин (например "admin_user_{{run_nonce}}") в нескольких кейсах — внутри
      одного прогона run_nonce одинаков, поэтому второй register получает 400/409
      "username already taken". Чиним: вшиваем в username/email номер кейса.

  R2. 401-тест с заголовком Authorization. Шаг, который проверяет отказ без токена
      (expected 401), но тащит "Authorization: Bearer {{token}}". Убираем заголовок —
      в этом и смысл теста.

  R3. Использование {{token}} без шага логина. status_code/stateless кейс шлёт
      "Bearer {{token}}", но в кейсе нет register→login, который этот token извлекает →
      на сервер уходит литерал → 401 вместо 403/422. Чиним: добавляем в начало кейса
      шаги register+login (extract token), переводим кейс в contextual.

  R4. Недостижимый admin-позитив. Шаг на admin-only эндпоинт с ожидаемым НЕ-403 кодом
      (2xx happy-path или 422 валидация), когда схема регистрации не принимает поле role
      и admin-токен через API получить нельзя. Такой шаг всегда даст 403 → кейс
      отбраковываем целиком (это не баг API, а свойство спеки).

  R5. Позитивный логин без регистрации. POST /auth/login с ожиданием 2xx, но в кейсе
      нет предшествующего register с теми же учётками → 401. Чиним: добавляем register.

  R6. Ссылки на неопределённые переменные. После всех правок шаг всё ещё использует
      {{var}}, которую никакой предыдущий шаг не извлекает (и это не run_nonce) →
      кейс заведомо сломан → отбраковываем.

Возвращает новый TestSuite и список текстовых действий (для лога/UI).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from models import TestCase, TestStep, TestSuite
from spec_parser import OpenAPISpec
# Переиспользуем уже существующие эвристики анализатора, чтобы логика admin/role
# была единой во всём проекте.
from ai_analyzer import _is_admin_only, _register_accepts_role


_TPL = re.compile(r"\{\{(\w+)\}\}")
_CRED_KEYS = {"username", "login", "email"}
_DEFAULT_PASSWORD = "Passw0rd_2026"
_TOKEN_VAR = "token"


# ── низкоуровневые утилиты ────────────────────────────────────────────────────
def _vars_in(obj: Any) -> set:
    """Собирает имена {{переменных}} из строки / dict / list рекурсивно."""
    out: set = set()
    if isinstance(obj, str):
        out.update(_TPL.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _vars_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _vars_in(v)
    return out


def _step_used_vars(step: TestStep) -> set:
    """Все {{переменные}}, на которые ссылается шаг (endpoint/headers/body/query)."""
    used = set()
    used |= _vars_in(step.endpoint)
    used |= _vars_in(step.headers or {})
    used |= _vars_in(step.body or {})
    used |= _vars_in(step.query_params or {})
    return used


def _find_op(
    spec: OpenAPISpec, method: str, endpoint: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Находит (spec_path, operation) для конкретного эндпоинта шага.

    Сопоставляет по числу сегментов: статические сегменты должны совпадать, а
    шаблонные ({book_id} в спеке, литерал "1" или "{{book_id}}" в шаге) — совпадают
    с чем угодно. Так "/books/1" и "/books/{{book_id}}" оба матчатся на "/books/{book_id}".
    """
    ep = endpoint.split("?", 1)[0]
    if not ep.startswith("/"):
        ep = "/" + ep
    segs = [s for s in ep.strip("/").split("/") if s != ""]
    for path, item in spec.paths.items():
        if not isinstance(item, dict):
            continue
        op = item.get(method.lower())
        if not isinstance(op, dict):
            continue
        psegs = [s for s in path.strip("/").split("/") if s != ""]
        if len(psegs) != len(segs):
            continue
        ok = True
        for a, b in zip(psegs, segs):
            if a.startswith("{") and a.endswith("}"):
                continue
            if a != b:
                ok = False
                break
        if ok:
            return path, op
    return None, None


def _first_2xx(op: Optional[Dict[str, Any]], default: int) -> int:
    """Первый задекларированный 2xx-код операции (для register/login), иначе default."""
    if isinstance(op, dict):
        responses = op.get("responses") or {}
        if isinstance(responses, dict):
            codes = sorted(
                int(c) for c in responses
                if str(c).isdigit() and 200 <= int(c) < 300
            )
            if codes:
                return codes[0]
    return default


# ── R1: уникальные учётки ─────────────────────────────────────────────────────
def _tag_for(tc_id: str) -> str:
    """Короткий уникальный для кейса суффикс из его id (tc_021 → '021')."""
    m = re.search(r"(\d+)", tc_id or "")
    return m.group(1) if m else (tc_id or "x").replace("tc_", "")


def _canon_username(orig: str, tag: str) -> str:
    """Гарантирует, что логин содержит номер кейса (уникальность в прогоне) и run_nonce."""
    v = (orig or "").strip() or "user"
    if tag not in v:
        v = f"{v}_{tag}"
    if "{{run_nonce}}" not in v:
        v = v + "_{{run_nonce}}"
    return v


def _canon_email(orig: str, tag: str) -> str:
    """То же для email — вставляем уникальность в локальную часть до '@'."""
    v = (orig or "").strip()
    if "@" not in v:
        return _canon_username(v, tag) + "@example.com"
    local, domain = v.split("@", 1)
    if tag not in local:
        local = f"{local}_{tag}"
    if "{{run_nonce}}" not in local:
        local = local + "_{{run_nonce}}"
    return f"{local}@{domain}"


def _rewrite_creds(steps: List[TestStep], tag: str) -> bool:
    """
    Делает учётки уникальными для кейса. Один и тот же исходный логин в разных шагах
    кейса (register и login) маппится в одно и то же новое значение — пара остаётся
    согласованной. Возвращает True, если что-то поменялось.
    """
    cache: Dict[Tuple[str, str], str] = {}
    changed = False

    def _map(key: str, val: str) -> str:
        ck = (key.lower(), val)
        if ck in cache:
            return cache[ck]
        new = _canon_email(val, tag) if key.lower() == "email" else _canon_username(val, tag)
        cache[ck] = new
        return new

    def _fix(obj: Any) -> Any:
        nonlocal changed
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, str) and k.lower() in _CRED_KEYS:
                    nv = _map(k, v)
                    if nv != v:
                        changed = True
                    out[k] = nv
                else:
                    out[k] = _fix(v)
            return out
        if isinstance(obj, list):
            return [_fix(x) for x in obj]
        return obj

    for st in steps:
        if st.body is not None:
            st.body = _fix(st.body)
    return changed


# ── префикс register+login ────────────────────────────────────────────────────
def _case_creds(steps: List[TestStep], tag: str) -> Tuple[str, str]:
    """Достаёт username/password из кейса (или синтезирует) для авто-шагов register/login."""
    user: Optional[str] = None
    pwd: Optional[str] = None
    for st in steps:
        b = st.body if isinstance(st.body, dict) else {}
        for k, v in b.items():
            if k.lower() in ("username", "login") and isinstance(v, str) and user is None:
                user = v
            if k.lower() == "password" and isinstance(v, str) and pwd is None:
                pwd = v
    user = _canon_username(user or "user", tag)
    return user, (pwd or _DEFAULT_PASSWORD)


def _make_auth_steps(
    spec: OpenAPISpec,
    reg_path: str,
    login_path: str,
    user: str,
    pwd: str,
    want_admin: bool,
) -> List[TestStep]:
    """Строит шаги [register, login(extract token)] для подстановки в начало кейса."""
    _, reg_op = _find_op(spec, "POST", reg_path)
    _, login_op = _find_op(spec, "POST", login_path)
    reg_body: Dict[str, Any] = {"username": user, "password": pwd}
    if want_admin:
        reg_body["role"] = "admin"
    register = TestStep(
        description="Авто: регистрация пользователя для получения токена",
        endpoint=reg_path,
        method="POST",
        headers={"Content-Type": "application/json"},
        body=reg_body,
        expected_status=_first_2xx(reg_op, 201),
    )
    login = TestStep(
        description="Авто: логин и извлечение Bearer-токена",
        endpoint=login_path,
        method="POST",
        headers={"Content-Type": "application/json"},
        body={"username": user, "password": pwd},
        expected_status=_first_2xx(login_op, 200),
        extract={_TOKEN_VAR: "$.access_token"},
    )
    return [register, login]


# ── обработка одного кейса ────────────────────────────────────────────────────
def _fix_case(
    tc: TestCase,
    spec: OpenAPISpec,
    reg_path: Optional[str],
    login_path: Optional[str],
    accepts_role: bool,
) -> Tuple[Optional[TestCase], List[str]]:
    """
    Возвращает (исправленный_кейс | None, список_действий).
    None означает «кейс отбракован» (первая строка действий объясняет причину).
    """
    actions: List[str] = []
    # Работаем с глубокими копиями шагов
    steps: List[TestStep] = [TestStep.model_validate(s.model_dump()) for s in tc.steps]
    tag = _tag_for(tc.id)
    new_type = tc.type
    prepended = False

    # R1 — уникальные учётки
    if _rewrite_creds(steps, tag):
        actions.append(f"{tc.id}: учётные данные сделаны уникальными для кейса (R1)")

    # R2 — снять Authorization у чистого 401-теста
    for st in steps:
        if st.expected_status == 401 and st.headers and "Authorization" in st.headers:
            if _TOKEN_VAR in _vars_in(st.headers.get("Authorization", "")):
                st.headers = {k: v for k, v in st.headers.items() if k != "Authorization"}
                if st.depends_on_vars:
                    st.depends_on_vars = [v for v in st.depends_on_vars if v != _TOKEN_VAR] or None
                actions.append(f"{tc.id}: убран Authorization у 401-шага (R2)")

    # R4 — недостижимый admin (любой не-403 код на admin-only при отсутствии role)
    for st in steps:
        path, op = _find_op(spec, st.method, st.endpoint)
        if (
            op is not None
            and _is_admin_only(op)
            and not accepts_role
            and st.expected_status not in (401, 403)
        ):
            return None, [
                f"{tc.id}: отбракован — {st.method} {path} только для admin, а admin-токен "
                f"через этот API получить нельзя (схема регистрации без поля role); "
                f"ожидаемый {st.expected_status} недостижим (R4)"
            ]

    # Что извлекается и что используется
    extracted: set = set()
    for st in steps:
        if st.extract:
            extracted |= set(st.extract.keys())
    used: set = set()
    for st in steps:
        used |= _step_used_vars(st)
    missing = used - extracted - {"run_nonce"}

    # R3 — нужен token, но логина нет → добавить register+login
    if _TOKEN_VAR in missing:
        if reg_path and login_path:
            want_admin = False
            if accepts_role:
                for st in steps:
                    if _TOKEN_VAR in _step_used_vars(st):
                        _, o = _find_op(spec, st.method, st.endpoint)
                        if o is not None and _is_admin_only(o) and 200 <= st.expected_status < 300:
                            want_admin = True
            user, pwd = _case_creds(steps, tag)
            steps = _make_auth_steps(spec, reg_path, login_path, user, pwd, want_admin) + steps
            prepended = True
            for st in steps[2:]:
                if _TOKEN_VAR in _step_used_vars(st):
                    dep = set(st.depends_on_vars or []) | {_TOKEN_VAR}
                    st.depends_on_vars = sorted(dep)
            missing.discard(_TOKEN_VAR)
            actions.append(f"{tc.id}: добавлены авто-шаги register+login для токена (R3)")
        else:
            return None, [f"{tc.id}: отбракован — нужен токен, но в API нет register/login (R3)"]
    else:
        # R5 — позитивный логин без регистрации
        if reg_path and login_path:
            login_idx = _first_login_without_register(steps, spec, login_path, reg_path)
            if login_idx is not None:
                login_step = steps[login_idx]
                body = login_step.body if isinstance(login_step.body, dict) else {}
                user = _canon_username(
                    next((str(v) for k, v in body.items()
                          if k.lower() in ("username", "login")), "user"),
                    tag,
                )
                pwd = next(
                    (str(v) for k, v in body.items() if k.lower() == "password"),
                    _DEFAULT_PASSWORD,
                )
                _, reg_op = _find_op(spec, "POST", reg_path)
                register = TestStep(
                    description="Авто: регистрация пользователя перед логином",
                    endpoint=reg_path,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body={"username": user, "password": pwd},
                    expected_status=_first_2xx(reg_op, 201),
                )
                # синхронизируем учётку логина с регистрацией
                if isinstance(login_step.body, dict):
                    for k in list(login_step.body.keys()):
                        if k.lower() in ("username", "login"):
                            login_step.body[k] = user
                        if k.lower() == "password":
                            login_step.body[k] = pwd
                steps.insert(login_idx, register)
                prepended = True
                actions.append(f"{tc.id}: добавлен шаг register перед позитивным login (R5)")

    # R6 — остались неопределённые переменные
    if missing:
        return None, [
            f"{tc.id}: отбракован — используются неопределённые переменные "
            f"{sorted(missing)} (R6)"
        ]

    if prepended:
        new_type = "contextual"

    fixed = TestCase(
        id=tc.id,
        name=tc.name,
        description=tc.description,
        type=new_type,
        steps=steps,
        tags=tc.tags,
        priority=tc.priority,
    )
    return fixed, actions


def _first_login_without_register(
    steps: List[TestStep], spec: OpenAPISpec, login_path: str, reg_path: str
) -> Optional[int]:
    """Индекс первого позитивного login-шага, перед которым нет register; иначе None."""
    has_register_before = False
    for idx, st in enumerate(steps):
        path, _ = _find_op(spec, st.method, st.endpoint)
        if path == reg_path and st.method.upper() == "POST":
            has_register_before = True
        if (
            path == login_path
            and st.method.upper() == "POST"
            and 200 <= st.expected_status < 300
            and not has_register_before
        ):
            return idx
    return None


# ── публичная точка входа ─────────────────────────────────────────────────────
def _find_auth_paths(spec: OpenAPISpec) -> Tuple[Optional[str], Optional[str]]:
    """Находит пути register/login (POST) по имени пути."""
    reg_path: Optional[str] = None
    login_path: Optional[str] = None
    for path, item in spec.paths.items():
        if not isinstance(item, dict):
            continue
        has_post = any(str(k).lower() == "post" for k in item)
        if not has_post:
            continue
        low = path.lower()
        if reg_path is None and "register" in low:
            reg_path = path
        if login_path is None and "login" in low:
            login_path = path
    return reg_path, login_path


def validate_and_fix_suite(
    suite: TestSuite, spec: OpenAPISpec
) -> Tuple[TestSuite, List[str]]:
    """
    Прогоняет все кейсы через детерминированные правила R1–R6.

    Возвращает (новый_сьют, список_действий). Кейсы, которые нельзя починить,
    отбрасываются — их причины тоже попадают в список действий.
    """
    reg_path, login_path = _find_auth_paths(spec)
    accepts_role = _register_accepts_role(spec)

    kept: List[TestCase] = []
    actions: List[str] = []

    for tc in suite.test_cases:
        try:
            fixed, case_actions = _fix_case(tc, spec, reg_path, login_path, accepts_role)
        except Exception as exc:  # валидатор не имеет права ронять прогон
            kept.append(tc)
            actions.append(f"{tc.id}: пропущен валидатором из-за ошибки ({exc})")
            continue
        actions.extend(case_actions)
        if fixed is not None:
            kept.append(fixed)

    new_suite = TestSuite(
        test_cases=kept,
        generated_at=suite.generated_at,
        spec_title=suite.spec_title,
        spec_version=suite.spec_version,
    )
    return new_suite, actions
