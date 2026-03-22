# AI-тестировщик API

Python CLI-инструмент для автоматической генерации и выполнения тест-кейсов для любого REST API,
описанного спецификацией OpenAPI. Использует LLM (GPTunnel / Anthropic Claude) для интеллектуального
анализа спецификации и построения реалистичных сценариев.

## Возможности

- Парсинг спецификаций OpenAPI 3.x и Swagger 2.x (JSON и YAML)
- Поддержка двух LLM-провайдеров: **GPTunnel** (OpenAI-совместимый) и **Anthropic** (нативный API)
- Три типа генерируемых тестов:
  - **stateless** — независимые тесты отдельных эндпоинтов, без контекста
  - **contextual** — многошаговые бизнес-сценарии (регистрация → вход → создание → проверка → удаление)
  - **status_code** — точная проверка HTTP-кодов (401, 403, 404, 422 и др.)
- Выполнение тестов через асинхронные HTTP-запросы (httpx)
- Извлечение значений из ответов (токены, ID) и подстановка в последующие шаги через JSONPath
- Пропуск зависимых шагов при падении предшественника
- Сохранение подробного JSON-отчёта и форматированный вывод таблицы в терминал

---

## Требования

- Python 3.11+
- API-ключ провайдера LLM:
  - GPTunnel: переменная окружения `GPTUNNEL_API_KEY`
  - Anthropic: переменная окружения `ANTHROPIC_API_KEY`

---

## Установка

```bash
cd api_tester
pip install -r requirements.txt
```

---

## Быстрый старт

```bash
# GPTunnel (по умолчанию, модель gpt-4o)
export GPTUNNEL_API_KEY="ваш-ключ"
python main.py --spec openapi.yaml --base-url https://api.example.com

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --spec openapi.yaml --base-url https://api.example.com \
               --provider anthropic --model claude-sonnet-4-6

# Все параметры
python main.py \
  --spec openapi.yaml \
  --base-url https://api.example.com \
  --config config.yaml \
  --output reports/results.json \
  --verbose
```

---

## Конфигурация

Скопируйте пример конфига и заполните значения:

```bash
cp config.yaml.example config.yaml
```

Поддерживается подстановка переменных окружения через синтаксис `${ИМЯ_ПЕРЕМЕННОЙ}`.
Файл `.env` в текущей директории загружается автоматически.

```yaml
auth:
  type: bearer          # bearer | api_key | none
  token: "${API_TOKEN}"

test_settings:
  timeout: 30
  delay_between_requests: 0.5

ai_settings:
  provider: gptunnel    # gptunnel | anthropic
  model: gpt-4o
  max_test_cases: 20
  include_negative_tests: true
  include_edge_cases: true
```

---

## Справка по CLI

```
python main.py [ПАРАМЕТРЫ]

Обязательные:
  --spec FILE          Путь к файлу OpenAPI-спецификации (.json, .yaml, .yml)
  --base-url URL       Базовый URL тестируемого API-сервера

Необязательные:
  --config FILE        Путь к конфиг-файлу (по умолчанию: config.yaml)
  --output FILE        Путь для JSON-отчёта (по умолчанию: results.json)
  --verbose            Подробный вывод: запросы, ответы, извлечённые значения
  --api-key KEY        API-ключ провайдера LLM (переопределяет переменную окружения)
  --provider PROVIDER  LLM-провайдер: gptunnel | anthropic (переопределяет конфиг)
  --model MODEL        Модель LLM, например gpt-4o или claude-sonnet-4-6 (переопределяет конфиг)
  --max-tests N        Максимальное количество генерируемых тест-кейсов
  --no-negative        Пропустить негативные тесты (невалидные данные)
  --no-edge-cases      Пропустить граничные тесты
```

### Примеры моделей

| Провайдер  | Модель                      | Описание                      |
|------------|-----------------------------|-------------------------------|
| gptunnel   | `gpt-4o` (по умолчанию)     | Быстро, экономно              |
| gptunnel   | `gpt-4o-mini`               | Дешевле, подходит для простых API |
| gptunnel   | `claude-opus-4-6`           | Claude через GPTunnel-прокси   |
| anthropic  | `claude-sonnet-4-6`         | Баланс качества и скорости    |
| anthropic  | `claude-opus-4-6`           | Максимальное качество         |

---

## Формат JSON-отчёта

```json
{
  "summary": {
    "total": 20,
    "passed": 16,
    "failed": 3,
    "errors": 1,
    "skipped": 0,
    "duration_ms": 12450.5,
    "timestamp": "2026-03-06T18:00:00+00:00"
  },
  "base_url": "https://api.example.com",
  "spec_file": "/path/to/openapi.yaml",
  "test_cases": [
    {
      "test_case_id": "tc_001",
      "test_case_name": "Регистрация нового пользователя",
      "type": "stateless",
      "status": "passed",
      "duration_ms": 320.5,
      "steps_results": [
        {
          "step_description": "POST /users с валидными данными",
          "endpoint": "/users",
          "method": "POST",
          "request_url": "https://api.example.com/users",
          "request_body": {"name": "Иван Иванов", "email": "ivan@example.com"},
          "expected_status": 201,
          "actual_status": 201,
          "passed": true,
          "duration_ms": 320.5,
          "extracted_values": {"user_id": "42"}
        }
      ]
    }
  ]
}
```

---

## Как работают контекстные тесты

Контекстные тесты имитируют реальные пользовательские сценарии. Модель генерирует шаги с:

1. **`extract`** — JSONPath-выражения для извлечения значений из ответа:
   ```json
   "extract": {"token": "$.access_token", "user_id": "$.data.id"}
   ```

2. **`{{variable}}`** — шаблоны для подстановки извлечённых значений в последующие шаги:
   ```json
   "headers": {"Authorization": "Bearer {{token}}"},
   "endpoint": "/users/{{user_id}}"
   ```

3. **`depends_on_vars`** — если требуемая переменная не извлечена (предыдущий шаг упал),
   зависимый шаг автоматически пропускается со статусом `skipped`:
   ```json
   "depends_on_vars": ["token", "user_id"]
   ```

---

## Коды завершения

| Код | Значение                                     |
|-----|----------------------------------------------|
| 0   | Все тесты прошли успешно                     |
| 1   | Один или несколько тестов провалились/ошибка |
| 1   | Ошибка конфигурации или спецификации         |

---

## Структура проекта

```
api_tester/
├── main.py              Точка входа CLI, оркестрация пайплайна
├── config.py            Загрузка YAML-конфига и переменных окружения
├── models.py            Pydantic v2 модели данных
├── spec_parser.py       Парсер OpenAPI JSON/YAML
├── ai_analyzer.py       Интеграция с LLM (GPTunnel / Anthropic)
├── context_manager.py   Извлечение JSONPath и подстановка шаблонов
├── test_runner.py       Асинхронный HTTP-исполнитель тестов (httpx)
├── reporter.py          Формирование JSON-отчёта и вывод в консоль
├── requirements.txt
└── config.yaml.example
```****