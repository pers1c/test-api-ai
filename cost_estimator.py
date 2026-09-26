"""
Оценка и расчёт стоимости LLM-генерации.

Два режима:
  - ESTIMATE (до запуска): прикидываем объём токенов по размеру спеки и max_tests,
    умножаем на цену модели — получаем ожидаемую стоимость в рублях.
  - ACTUAL (после запуска): складываем prompt_tokens/completion_tokens из всех ответов
    GPTunnel, считаем реальную стоимость.

Оценочные коэффициенты выведены из реальной структуры промптов:
  - Planner:  ~800 токенов системный промпт и правила + ~30 на эндпоинт + few-shot (~400)
  - Generator: ~900 + релевантные схемы (зависит от эндпоинта) + few-shot (~500), на каждый тест

Погрешность оценки — около ±30% (у LLM недетерминированное поведение,
но порядок величины будет верный).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from models import PricingConfig, ModelPrice
from spec_parser import OpenAPISpec


# ============================================================
# Дефолтные цены моделей в GPTunnel
# Цены актуальны на апрель 2026, в рублях за 1 МИЛЛИОН токенов.
# Могут быть переопределены через pricing.models в config.yaml.
# ============================================================
DEFAULT_PRICES: Dict[str, ModelPrice] = {
    # OpenAI (цены GPTunnel, актуальны на май 2026)
    "gpt-4o": ModelPrice(input_per_1m=1350.0, output_per_1m=2700.0),
    "gpt-4o-mini": ModelPrice(input_per_1m=120.0, output_per_1m=2700.0),
    "o1": ModelPrice(input_per_1m=4500.0, output_per_1m=18000.0),
    "o1-mini": ModelPrice(input_per_1m=1200.0, output_per_1m=4800.0),

    # Anthropic
    "claude-opus-4-6": ModelPrice(input_per_1m=1500.0, output_per_1m=7500.0),
    "claude-sonnet-4-6": ModelPrice(input_per_1m=1200.0,  output_per_1m=4500.0),
    "claude-haiku-4-5": ModelPrice(input_per_1m=200.0,   output_per_1m=1000.0),

    # Google
    "gemini-2.5-pro": ModelPrice(input_per_1m=350.0,  output_per_1m=1500.0),
    "gemini-2.5-flash": ModelPrice(input_per_1m=60.0,   output_per_1m=180.0),
}


# ============================================================
# Коэффициенты оценки токенов (калибровка по реальным промптам)
# ============================================================

# Planner — один вызов
_PLANNER_BASE_INPUT = 800     # системный промпт + правила + структура
_PLANNER_PER_ENDPOINT = 30      # строка сводки на каждый эндпоинт
_PLANNER_FEW_SHOT = 400     # пример
_PLANNER_OUTPUT_PER_CASE = 70   # один элемент плана — ~70 токенов

# Generator — на каждый тест-кейс
_GEN_BASE_INPUT = 900       # системный промпт + правила + few-shot разметка
_GEN_FEW_SHOT = 500       # пример
_GEN_SPEC_FRAGMENT = 600       # релевантные фрагменты спеки (оценка среднего)
_GEN_PLAN_ITEM = 100       # описание пункта плана

# Выход генератора зависит от типа теста (в среднем)
_GEN_OUTPUT_STATELESS   = 300
_GEN_OUTPUT_STATUS_CODE = 500
_GEN_OUTPUT_CONTEXTUAL  = 800

# Распределение типов тестов (настроено в промпте планировщика)
_DISTRIBUTION = {
    "stateless":    0.40,
    "status_code":  0.35,
    "contextual":   0.25,
}


@dataclass
class CostEstimate:
    """Результат оценки или факт стоимости прогона."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_rub: float
    model: str
    price_known: bool           # False если цены модели нет — стоимость = 0
    breakdown: Optional[Dict[str, float]] = None  # детализация по этапам (для estimate)

    def to_dict(self) -> dict:
        d = {
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens":  self.total_tokens,
            "cost_rub":      round(self.cost_rub, 4),
            "model":         self.model,
            "price_known":   self.price_known,
        }
        if self.breakdown is not None:
            d["breakdown"] = {k: round(v, 4) for k, v in self.breakdown.items()}
        return d


def resolve_price(
    model: str,
    pricing_config: Optional[PricingConfig] = None,
) -> Optional[ModelPrice]:
    """
    Ищет цену модели: сначала в конфиге, затем в дефолтах.
    Возвращает None если модель неизвестна.
    """
    # 1. Конфиг имеет приоритет
    if pricing_config and pricing_config.models:
        if model in pricing_config.models:
            return pricing_config.models[model]

    # 2. Дефолты
    if model in DEFAULT_PRICES:
        return DEFAULT_PRICES[model]

    # 3. Частичное совпадение — ищем по префиксу
    # (например "gpt-4o-2024-11-20" → "gpt-4o")
    for known_name, price in DEFAULT_PRICES.items():
        if model.startswith(known_name):
            return price

    return None


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    price: Optional[ModelPrice],
) -> float:
    """Стоимость в рублях для заданного числа токенов и цены."""
    if price is None:
        return 0.0
    return (
        input_tokens  * price.input_per_1m  / 1_000_000
        + output_tokens * price.output_per_1m / 1_000_000
    )


def estimate_run(
    spec: OpenAPISpec,
    model: str,
    max_tests: int,
    pricing_config: Optional[PricingConfig] = None,
) -> CostEstimate:
    """
    Оценивает ожидаемую стоимость прогона на основе спеки и параметров.

    Подход: делаем физически обоснованный расчёт вместо угадывания.
    Planner даёт план размером N тестов (от max_tests, но не более endpoints*3),
    потом Generator запускается N раз с разными входами.
    """
    endpoint_count = spec.get_endpoint_count()

    # Сколько тест-кейсов реалистично сгенерируется
    # (модель может сгенерировать меньше max_tests, если API маленький)
    realistic_count = min(max_tests, max(endpoint_count * 2, 5))

    # --- Planner (один вызов) ---
    planner_input = (
        _PLANNER_BASE_INPUT
        + _PLANNER_PER_ENDPOINT * endpoint_count
        + _PLANNER_FEW_SHOT
    )
    planner_output = _PLANNER_OUTPUT_PER_CASE * realistic_count

    # --- Generator (N вызовов) ---
    # Средний output по распределению типов
    avg_gen_output = (
        _DISTRIBUTION["stateless"]   * _GEN_OUTPUT_STATELESS
        + _DISTRIBUTION["status_code"] * _GEN_OUTPUT_STATUS_CODE
        + _DISTRIBUTION["contextual"]  * _GEN_OUTPUT_CONTEXTUAL
    )

    gen_input_per_call = (
        _GEN_BASE_INPUT
        + _GEN_FEW_SHOT
        + _GEN_SPEC_FRAGMENT
        + _GEN_PLAN_ITEM
    )

    gen_total_input  = gen_input_per_call * realistic_count
    gen_total_output = int(avg_gen_output * realistic_count)

    # --- Итого ---
    total_input  = planner_input + gen_total_input
    total_output = planner_output + gen_total_output

    price = resolve_price(model, pricing_config)
    total_cost = calculate_cost(total_input, total_output, price)

    # Разбивка стоимости по этапам (полезно для анализа)
    breakdown: Optional[Dict[str, float]] = None
    if price is not None:
        breakdown = {
            "planner_rub":   calculate_cost(planner_input,  planner_output,  price),
            "generator_rub": calculate_cost(gen_total_input, gen_total_output, price),
        }

    return CostEstimate(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        cost_rub=total_cost,
        model=model,
        price_known=price is not None,
        breakdown=breakdown,
    )


def compute_actual_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    pricing_config: Optional[PricingConfig] = None,
) -> CostEstimate:
    """
    Считает фактическую стоимость по накопленным токенам (из ответов GPTunnel).
    """
    price = resolve_price(model, pricing_config)
    cost  = calculate_cost(input_tokens, output_tokens, price)
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_rub=cost,
        model=model,
        price_known=price is not None,
    )


def list_known_models(pricing_config: Optional[PricingConfig] = None) -> List[str]:
    """Список моделей с известными ценами — для подсказок в UI."""
    names = set(DEFAULT_PRICES.keys())
    if pricing_config and pricing_config.models:
        names.update(pricing_config.models.keys())
    return sorted(names)
