"""
Лёгкая структурная валидация тела ответа против объявленной в спеке схемы.

Намеренно НЕ используем строгий jsonschema: OpenAPI 3.0 и 3.1 по-разному трактуют
nullable/exclusiveMinimum, и строгий валидатор даёт ложные «провалы» на валидных
ответах. Здесь проверяются только высокосигнальные нарушения, которые почти всегда
означают реальный баг сериализации:

  - неверный тип значения (declared type vs фактический);
  - отсутствие обязательного (required) поля объекта;
  - значение вне enum;
  - null там, где поле не nullable.

Намеренно НЕ проверяются: лишние поля (OpenAPI по умолчанию их допускает), format,
числовые границы (min/max) — они шумные и редко нарушаются в ответах.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from spec_parser import OpenAPISpec

_HTTP_METHODS = {"get", "post", "put", "delete", "patch"}
_MAX_DEPTH = 25
_MAX_ARRAY_ITEMS = 50  # не проверяем гигантские списки целиком


def _resolve_ref(root: Dict[str, Any], ref: str) -> Optional[Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    cur: Any = root
    for part in ref.lstrip("#/").split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _deref(schema: Any, root: Dict[str, Any], _seen: Optional[set] = None) -> Any:
    """Разворачивает цепочку $ref (с защитой от циклов)."""
    seen = _seen or set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            return {}
        seen.add(ref)
        resolved = _resolve_ref(root, ref)
        if resolved is None:
            return {}
        schema = resolved
    return schema


def _match_path_item(spec: OpenAPISpec, method: str, endpoint: str):
    """Находит operation спеки по конкретному пути шага (сегментное сопоставление)."""
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
        if all(a.startswith("{") and a.endswith("}") or a == b
               for a, b in zip(psegs, segs)):
            return op
    return None


def response_schema(
    spec: OpenAPISpec, method: str, endpoint: str, status: int
) -> Optional[Dict[str, Any]]:
    """
    Возвращает (неразвёрнутую) схему тела ответа для status, либо None.
    Поддерживает OpenAPI 3 (responses[code].content[*].schema) и Swagger 2
    (responses[code].schema). Пустую схему {} трактуем как «нет ограничений» → None.
    """
    op = _match_path_item(spec, method, endpoint)
    if not isinstance(op, dict):
        return None
    responses = op.get("responses")
    if not isinstance(responses, dict):
        return None
    resp = responses.get(str(status)) or responses.get(status)
    if not isinstance(resp, dict):
        return None
    schema: Optional[Any] = None
    content = resp.get("content")
    if isinstance(content, dict) and content:
        first = next(iter(content.values()))
        if isinstance(first, dict):
            schema = first.get("schema")
    if schema is None:
        schema = resp.get("schema")  # Swagger 2
    if not isinstance(schema, dict) or not schema:
        return None
    return schema


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _type_ok(value: Any, declared: str) -> bool:
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, list)
    if declared == "null":
        return value is None
    return True  # неизвестный тип — не придираемся


def validate(
    body: Any,
    schema: Any,
    root: Dict[str, Any],
    path: str = "$",
    depth: int = 0,
) -> List[str]:
    """Возвращает список нарушений (пустой = тело валидно)."""
    errors: List[str] = []
    if depth > _MAX_DEPTH:
        return errors
    schema = _deref(schema, root)
    if not isinstance(schema, dict) or not schema:
        return errors

    # Комбинаторы: anyOf/oneOf — валидно, если подходит хотя бы под одну ветку
    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list) and branches:
        for sub in branches:
            if not validate(body, sub, root, path, depth + 1):
                return errors  # нашли подходящую ветку
        errors.append(f"{path}: не соответствует ни одной из допустимых схем (anyOf/oneOf)")
        return errors
    if isinstance(schema.get("allOf"), list):
        for sub in schema["allOf"]:
            errors.extend(validate(body, sub, root, path, depth + 1))

    declared = schema.get("type")
    types = declared if isinstance(declared, list) else ([declared] if declared else [])
    nullable = schema.get("nullable") is True or "null" in types

    if body is None:
        if nullable or not types:
            return errors
        errors.append(f"{path}: ожидался тип {'/'.join(types)}, получен null")
        return errors

    if types:
        non_null = [t for t in types if t != "null"]
        if non_null and not any(_type_ok(body, t) for t in non_null):
            errors.append(
                f"{path}: ожидался тип {'/'.join(non_null)}, получен {_json_type(body)}"
            )
            return errors  # при несовпадении типа глубже не идём

    if "enum" in schema and isinstance(schema["enum"], list):
        if body not in schema["enum"]:
            errors.append(f"{path}: значение {body!r} вне допустимого enum {schema['enum']}")

    if isinstance(body, dict):
        props = schema.get("properties")
        required = schema.get("required") or []
        if isinstance(required, list):
            for req in required:
                if req not in body:
                    errors.append(f"{path}.{req}: отсутствует обязательное поле")
        if isinstance(props, dict):
            for key, sub in props.items():
                if key in body:
                    errors.extend(validate(body[key], sub, root, f"{path}.{key}", depth + 1))

    if isinstance(body, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, elem in enumerate(body[:_MAX_ARRAY_ITEMS]):
                errors.extend(validate(elem, items, root, f"{path}[{i}]", depth + 1))

    return errors


def validate_response_body(
    spec: OpenAPISpec, method: str, endpoint: str, status: int, body: Any
) -> List[str]:
    """
    Высокоуровневая обёртка: находит схему ответа для (method, endpoint, status) и
    валидирует тело. Пустой список — нет схемы или тело валидно.
    """
    schema = response_schema(spec, method, endpoint, status)
    if schema is None:
        return []
    return validate(body, schema, spec.raw)
