# AI-тестировщик API

Python CLI-инструмент, который использует **Claude claude-opus-4-6** (с адаптивным мышлением) для автоматической
генерации и выполнения тест-кейсов для любого REST API, описанного спецификацией OpenAPI.

## Возможности

- Парсинг спецификаций OpenAPI 3.x и Swagger 2.x (JSON и YAML)
- Отправка спецификации в Claude AI для интеллектуальной генерации тестов
- Генерация трёх типов тестов:
  - **Stateless (без контекста)** — независимые тесты отдельных эндпоинтов
  - **Contextual (контекстные)** — многошаговые сценарии (например: регистрация -> вход -> создание заказа -> проверка -> удаление)
  - **Status code (коды ответа)** — точная проверка HTTP-кодов для конкретных сценариев
- Выполнение тестов против реального сервера через асинхронные HTTP-запросы
- Извлечение значений из ответов (токены, ID) и подстановка в последующие шаги
- Сохранение подробного JSON-отчёта с результатами по каждому шагу
- Вывод форматированной таблицы результатов в терминале

---

## Требования

- Python 3.11+
- Ключ API Anthropic (задаётся через переменную окружения `ANTHROPIC_API_KEY`)

---

## Установка

```bash
cd api_tester
pip install -r requirements.txt
```

---

## Быстрый старт

```bash
# Задать ключ API Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Запустить тесты по OpenAPI-спецификации
python main.py \
  --spec openapi.yaml \
  --base-url https://api.example.com

# Запуск со всеми параметрами
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

Конфиг поддерживает подстановку переменных окружения через синтаксис `${ИМЯ_ПЕРЕМЕННОЙ}`.
Также автоматически загружается файл `.env` из текущей директории.

```yaml
auth:
  type: bearer
  token: "${API_TOKEN}"

test_settings:
  timeout: 30
  delay_between_requests: 0.5

ai_settings:
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
  --output FILE        Путь для сохранения JSON-отчёта (по умолчанию: results.json)
  --verbose            Подробный вывод: запросы, ответы, извлечённые значения
  --anthropic-api-key  Переопределить переменную ANTHROPIC_API_KEY
  --max-tests N        Переопределить максимальное количество тест-кейсов
  --no-negative        Пропустить негативные тесты (невалидные данные)
  --no-edge-cases      Пропустить граничные тесты
```

---

## Формат JSON-отчёта

Файл отчёта (`results.json`) имеет следующую структуру:

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

Контекстные тесты имитируют реальные пользовательские сценарии. Claude генерирует шаги с:

1. **`extract`** — JSONPath-выражения для извлечения значений из ответа:
   ```json
   "extract": {"token": "$.access_token", "user_id": "$.data.id"}
   ```

2. **`{{variable}}`** шаблоны — подстановка извлечённых значений в последующие шаги:
   ```json
   "headers": {"Authorization": "Bearer {{token}}"},
   "endpoint": "/users/{{user_id}}"
   ```

3. **`depends_on_vars`** — если требуемая переменная не была извлечена (из-за падения предыдущего шага),
   зависимый шаг автоматически пропускается с пометкой `skipped`.

---

## Коды выхода

| Код | Значение                               |
|-----|----------------------------------------|
| 0   | Все тесты прошли успешно               |
| 1   | Один или несколько тестов упали/ошибка |
| 1   | Ошибка конфигурации или спецификации   |

---

## Структура проекта

```
api_tester/
├── main.py            Точка входа CLI, оркестрация пайплайна
├── config.py          Загрузка YAML-конфига и переменных окружения
├── models.py          Pydantic v2 модели данных
├── spec_parser.py     Парсер OpenAPI JSON/YAML
├── ai_analyzer.py     Интеграция с Claude AI (генерация тестов)
├── context_manager.py Извлечение JSONPath и подстановка шаблонов
├── test_runner.py     Асинхронный HTTP-исполнитель тестов (httpx)
├── reporter.py        Формирование JSON-отчёта и вывод в консоль
├── requirements.txt
└── config.yaml.example
```
