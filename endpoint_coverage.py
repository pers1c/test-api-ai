"""
Вычисление матрицы покрытия: какие эндпоинты спеки задействованы какими тест-кейсами.

Важное отличие от наивного «коснулся = покрыл»: одно касание эндпоинта может быть
ВСПОМОГАТЕЛЬНЫМ (register/login ради получения токена), а не целевой проверкой. Поэтому
для каждой пары (кейс, эндпоинт) различаем:

  - role = "primary"  — эндпоинт является целью проверки;
  - role = "setup"    — эндпоинт задействован как auth-префикс (register/login),
                        и в кейсе есть другие, не-auth шаги.

И на уровне эндпоинта отмечаем дыры:
  - "не покрыт"            — ни один шаг его не дернул;
  - "только setup"        — виден лишь как auth-префикс, целевой проверки нет;
  - "нет happy-path"      — есть только негативные (не-2xx) primary-проверки;
  - "happy-path не проходит" — позитивный primary-тест есть, но он падает/пропущен.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models import (
    CoverageReport,
    EndpointCaseRef,
    EndpointCoverage,
    TestResult,
)
from spec_parser import OpenAPISpec


_HTTP_METHODS = {"get", "post", "put", "delete", "patch"}


def _enumerate_ops(spec: OpenAPISpec) -> List[Tuple[str, str]]:
    """Все (METHOD, path) операции спеки в порядке объявления."""
    ops: List[Tuple[str, str]] = []
    for path, item in spec.paths.items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method.lower() in _HTTP_METHODS and isinstance(op, dict):
                ops.append((method.upper(), path))
    return ops


def _match_op(
    ops: List[Tuple[str, str]], method: str, endpoint: str
) -> Optional[Tuple[str, str]]:
    """
    Сопоставляет конкретный путь шага с шаблоном операции спеки по сегментам.
    Шаблонные сегменты спеки ({id}) и подставленные значения шага (литерал "1",
    "{{var}}") совпадают с чем угодно. "/books/1" → "/books/{book_id}".
    """
    ep = endpoint.split("?", 1)[0]
    if not ep.startswith("/"):
        ep = "/" + ep
    segs = [s for s in ep.strip("/").split("/") if s != ""]
    for M, P in ops:
        if M != method.upper():
            continue
        psegs = [s for s in P.strip("/").split("/") if s != ""]
        if len(psegs) != len(segs):
            continue
        if all(a.startswith("{") and a.endswith("}") or a == b
               for a, b in zip(psegs, segs)):
            return (M, P)
    return None


def _is_auth_path(path: str) -> bool:
    """Путь register/login — кандидат на роль 'setup' (auth-префикс)."""
    low = path.lower()
    return "register" in low or "login" in low


def _outcome(passed_all: bool, any_failed: bool, all_skipped: bool) -> str:
    if all_skipped:
        return "skipped"
    if passed_all and not any_failed:
        return "passed"
    if any_failed and not passed_all:
        return "failed"
    return "mixed"


def compute_coverage(results: List[TestResult], spec: OpenAPISpec) -> CoverageReport:
    """Строит CoverageReport из результатов прогона и спеки."""
    ops = _enumerate_ops(spec)
    auth_paths = {p for (_m, p) in ops if _is_auth_path(p)}

    # endpoint(op) -> список ссылок на кейсы
    by_endpoint: Dict[Tuple[str, str], List[EndpointCaseRef]] = {op: [] for op in ops}

    for tc in results:
        # Сначала сматчим каждый шаг к операции спеки
        matched = [
            (_match_op(ops, st.method, st.endpoint), st) for st in tc.steps_results
        ]
        # Есть ли в кейсе НЕ-auth целевые шаги — тогда auth-шаги считаем setup'ом
        non_auth_present = any(
            mp is not None and mp[1] not in auth_paths for mp, _ in matched
        )

        # Агрегируем по операции в пределах кейса (один шаг или несколько)
        per_op: Dict[Tuple[str, str], Dict] = {}
        for mp, st in matched:
            if mp is None:
                continue
            is_auth = mp[1] in auth_paths
            role = "setup" if (is_auth and non_auth_present) else "primary"
            acc = per_op.setdefault(mp, {
                "role": role,
                "statuses": set(),
                "passed_all": True,
                "any_failed": False,
                "all_skipped": True,
            })
            # primary перебивает setup, если эндпоинт встречается в обеих ролях
            if role == "primary":
                acc["role"] = "primary"
            acc["statuses"].add(st.expected_status)
            if not st.skipped:
                acc["all_skipped"] = False
                if not st.passed:
                    acc["any_failed"] = True
                    acc["passed_all"] = False

        for mp, acc in per_op.items():
            by_endpoint[mp].append(EndpointCaseRef(
                test_case_id=tc.test_case_id,
                test_case_name=tc.test_case_name,
                role=acc["role"],
                expected_statuses=sorted(acc["statuses"]),
                outcome=_outcome(acc["passed_all"], acc["any_failed"], acc["all_skipped"]),
            ))

    # Сборка EndpointCoverage с флагами
    endpoints: List[EndpointCoverage] = []
    covered = tested = happy = flagged = 0

    # Чтобы определить has_happy_path, нужно знать pos_passed по primary-кейсам.
    # Пересоберём быстрый индекс (case_id, op) -> pos_passed из результатов.
    pos_passed_index: Dict[Tuple[str, str, str], bool] = {}
    for tc in results:
        for st in tc.steps_results:
            mp = _match_op(ops, st.method, st.endpoint)
            if mp is None:
                continue
            key = (tc.test_case_id, mp[0], mp[1])
            if (not st.skipped) and st.passed and 200 <= st.expected_status < 300:
                pos_passed_index[key] = True
            pos_passed_index.setdefault(key, pos_passed_index.get(key, False))

    for (m, p) in ops:
        refs = by_endpoint[(m, p)]
        is_covered = len(refs) > 0
        primary_refs = [r for r in refs if r.role == "primary"]
        is_tested = len(primary_refs) > 0

        has_pos_primary = any(
            any(200 <= s < 300 for s in r.expected_statuses) for r in primary_refs
        )
        has_pos_pass = any(
            pos_passed_index.get((r.test_case_id, m, p), False) for r in primary_refs
        )
        has_neg_primary = any(
            any(not (200 <= s < 300) for s in r.expected_statuses) for r in primary_refs
        )
        negative_only = is_tested and has_neg_primary and not has_pos_primary

        flags: List[str] = []
        if not is_covered:
            flags.append("не покрыт")
        elif not is_tested:
            flags.append("только setup (auth-префикс)")
        else:
            if not has_pos_primary:
                flags.append("нет happy-path (только негатив)")
            elif not has_pos_pass:
                flags.append("happy-path не проходит")

        ec = EndpointCoverage(
            method=m,
            path=p,
            covered=is_covered,
            tested=is_tested,
            has_happy_path=bool(has_pos_pass),
            negative_only=bool(negative_only),
            case_count=len(refs),
            cases=refs,
            flags=flags,
        )
        endpoints.append(ec)

        covered += 1 if is_covered else 0
        tested += 1 if is_tested else 0
        happy += 1 if has_pos_pass else 0
        flagged += 1 if flags else 0

    return CoverageReport(
        total_endpoints=len(ops),
        covered=covered,
        tested=tested,
        happy_path=happy,
        flagged=flagged,
        endpoints=endpoints,
    )
