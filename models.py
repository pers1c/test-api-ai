from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# Модели тест-кейсов (генерируются LLM)
class TestStep(BaseModel):
    """Один HTTP-запрос в рамках тест-кейса."""

    model_config = ConfigDict(extra="ignore")

    description: str
    endpoint: str                                        # например "/users/{id}" или "/users/{{user_id}}"
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None               # тело запроса (может содержать шаблоны {{var}})
    query_params: Optional[Dict[str, Any]] = None
    expected_status: int
    expected_response_schema: Optional[Dict[str, Any]] = None  # фрагмент JSON Schema для валидации
    extract: Optional[Dict[str, str]] = None            # {"имя_переменной": "$.json.path"} — для передачи значений следующим шагам
    depends_on_vars: Optional[List[str]] = None         # имена переменных, которые этот шаг ожидает из контекста


class TestCase(BaseModel):
    """Полный тест-кейс, состоящий из одного или нескольких шагов."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    description: str
    type: Literal["stateless", "contextual", "status_code"]
    steps: List[TestStep]
    tags: Optional[List[str]] = None
    priority: Optional[Literal["low", "medium", "high"]] = None


class TestSuite(BaseModel):
    """Корневой контейнер, возвращаемый LLM после анализа спецификации."""

    model_config = ConfigDict(extra="ignore")

    test_cases: List[TestCase]
    generated_at: Optional[str] = None
    spec_title: Optional[str] = None
    spec_version: Optional[str] = None
    # Пункты плана, которые НЕ попали в test_cases (провал генерации после ретраев
    # или отбраковка пост-валидатором). Заполняется генератором, чтобы потерянные
    # кейсы были видны, а не исчезали молча. Элемент: {id, name, type, stage, reason}.
    failed_generations: Optional[List[Dict[str, Any]]] = None

# Модели результатов (формируются после выполнения тестов)
class StepResult(BaseModel):
    """Результат выполнения одного HTTP-запроса."""

    step_description: str
    endpoint: str
    method: str
    request_url: str
    request_headers: Dict[str, str] = Field(default_factory=dict)
    request_body: Optional[Any] = None
    actual_status: Optional[int] = None
    expected_status: int
    passed: bool
    response_body: Optional[Any] = None
    response_headers: Optional[Dict[str, str]] = None
    duration_ms: float
    error_message: Optional[str] = None
    extracted_values: Optional[Dict[str, str]] = None
    skipped: bool = False
    # Нарушения схемы тела ответа (если включена валидация и они найдены).
    # Непустой список означает, что статус-код совпал, но тело не соответствует
    # объявленной в спеке схеме ответа — шаг считается проваленным.
    schema_errors: Optional[List[str]] = None


class TestResult(BaseModel):
    """Агрегированный результат выполнения одного тест-кейса."""

    test_case_id: str
    test_case_name: str
    type: str
    status: Literal["passed", "failed", "error", "skipped"]
    steps_results: List[StepResult] = Field(default_factory=list)
    duration_ms: float
    error_message: Optional[str] = None


class ReportSummary(BaseModel):
    """Общая статистика по всему прогону тестов."""

    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_ms: float
    timestamp: str


# Модели покрытия эндпоинтов тест-кейсами
class EndpointCaseRef(BaseModel):
    """Связь одного тест-кейса с одним эндпоинтом (агрегировано по шагам кейса)."""

    test_case_id: str
    test_case_name: str
    # primary — эндпоинт является целью проверки; setup — задействован как
    # вспомогательный (auth-префикс register/login для получения токена).
    role: Literal["primary", "setup"]
    expected_statuses: List[int] = Field(default_factory=list)
    # Итог касания эндпоинта этим кейсом: passed/failed/skipped/mixed
    outcome: Literal["passed", "failed", "skipped", "mixed"]


class EndpointCoverage(BaseModel):
    """Покрытие одной операции спеки всеми тест-кейсами прогона."""

    method: str
    path: str
    covered: bool          # задействован хотя бы одним шагом (любая роль)
    tested: bool           # задействован как primary хотя бы в одном кейсе
    has_happy_path: bool   # есть проходящий primary-тест с 2xx
    negative_only: bool    # покрыт как primary, но только негативными (не-2xx) проверками
    case_count: int        # сколько кейсов касаются эндпоинта (любая роль)
    cases: List[EndpointCaseRef] = Field(default_factory=list)
    # Человекочитаемые пометки о дырах покрытия (пусто = всё хорошо)
    flags: List[str] = Field(default_factory=list)


class CoverageReport(BaseModel):
    """Сводная матрица покрытия эндпоинтов тест-кейсами."""

    total_endpoints: int
    covered: int           # покрыто хотя бы как-то
    tested: int            # покрыто как primary
    happy_path: int        # есть проходящий 2xx primary
    flagged: int           # эндпоинтов с непустыми flags (дыры покрытия)
    endpoints: List[EndpointCoverage] = Field(default_factory=list)


class TestReport(BaseModel):
    """Полный отчёт о прогоне тестов, сохраняемый в JSON."""

    summary: ReportSummary
    test_cases: List[TestResult]
    base_url: str
    spec_file: str
    config_file: Optional[str] = None
    cost: Optional[Dict[str, Any]] = None  # {estimated, actual} — см. cost_estimator.CostEstimate
    coverage: Optional[CoverageReport] = None  # матрица покрытия эндпоинтов (см. coverage.py)

# Модели конфигурации
class AuthConfig(BaseModel):
    """Настройки аутентификации для тестируемого API."""

    type: Literal["bearer", "api_key", "none"] = "none"
    token: Optional[str] = None
    header_name: str = "Authorization"          # имя заголовка для типа api_key
    query_param_name: Optional[str] = None      # для API-ключей в query-параметрах (редко)


class TestSettings(BaseModel):
    """Настройки HTTP-выполнения запросов."""

    timeout: float = 30.0
    max_retries: int = 1
    delay_between_requests: float = 0.5
    # Проверять тело ответа против объявленной в спеке схемы (type/required/enum/nullable).
    # Если статус-код совпал, но тело не соответствует схеме — шаг проваливается.
    validate_response_body: bool = True


class AISettings(BaseModel):
    """Настройки генерации тест-кейсов через LLM (GPTunnel, OpenAI-совместимый API)."""

    # Модель по умолчанию. Примеры доступных моделей в GPTunnel:
    #   "gpt-4o", "gpt-4o-mini", "o1", "claude-opus-4-6", "gemini-2.5-pro"
    model: str = "gpt-4o"
    max_tokens: int = 16000

    # --- Параметры генерации тестов ---
    # Жёсткий верхний предел числа тест-кейсов. Реальное число рассчитывается
    # из структуры API (см. _compute_test_targets в ai_analyzer.py) и обычно
    # составляет ~3–5 кейсов на эндпоинт. Этот параметр служит защитой от
    # неконтролируемого роста стоимости генерации, а не целевым значением.
    max_test_cases: int = 200
    include_negative_tests: bool = True
    include_edge_cases: bool = True

    # Свободные инструкции от пользователя, которые передаются и планировщику, и
    # генератору. Полезно для уточнений после первого прогона ("проверь rate-limit
    # на /search", "обязательно покрой негативами /admin/*", и т.п.).
    # None или пустая строка означает отсутствие дополнительных инструкций.
    user_instructions: Optional[str] = None


class AppConfig(BaseModel):
    """Корневая конфигурация приложения."""

    auth: AuthConfig = Field(default_factory=AuthConfig)
    test_settings: TestSettings = Field(default_factory=TestSettings)
    ai_settings: AISettings = Field(default_factory=AISettings)
    pricing: "PricingConfig" = Field(default_factory=lambda: PricingConfig())


# Модели ценообразования
class ModelPrice(BaseModel):
    """Цена одной модели LLM: рубли за 1 миллион токенов, раздельно вход/выход."""

    input_per_1m: float = Field(
        ..., description="Стоимость входных токенов, руб. за 1M",
    )
    output_per_1m: float = Field(
        ..., description="Стоимость выходных токенов, руб. за 1M",
    )


class PricingConfig(BaseModel):
    """
    Пользовательские цены моделей, переопределяют встроенные дефолты.

    Ключ — имя модели как оно передаётся в GPTunnel API (например "gpt-4o").
    Можно добавлять новые модели, которых нет в дефолтах.
    """

    models: Dict[str, ModelPrice] = Field(default_factory=dict)


# Поле cost в TestReport обновляется после добавления модели Cost (ниже).
class Cost(BaseModel):
    """Данные о стоимости прогона — оценка до и/или факт после."""

    estimated: Optional[Dict[str, Any]] = None  # CostEstimate.to_dict()
    actual:    Optional[Dict[str, Any]] = None  # CostEstimate.to_dict()
