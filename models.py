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


class TestReport(BaseModel):
    """Полный отчёт о прогоне тестов, сохраняемый в JSON."""

    summary: ReportSummary
    test_cases: List[TestResult]
    base_url: str
    spec_file: str
    config_file: Optional[str] = None
    cost: Optional[Dict[str, Any]] = None  # {estimated, actual} — см. cost_estimator.CostEstimate

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


class AISettings(BaseModel):
    """Настройки генерации тест-кейсов через LLM (GPTunnel, OpenAI-совместимый API)."""

    # Модель по умолчанию. Примеры доступных моделей в GPTunnel:
    #   "gpt-4o", "gpt-4o-mini", "o1", "claude-opus-4-6", "gemini-2.5-pro"
    model: str = "gpt-4o"
    max_tokens: int = 16000

    # --- Параметры генерации тестов ---
    max_test_cases: int = 20
    include_negative_tests: bool = True
    include_edge_cases: bool = True


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
