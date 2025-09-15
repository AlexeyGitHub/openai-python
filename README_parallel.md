# Параллельная фильтрация диалогов

Этот проект содержит несколько версий скрипта для фильтрации диалогов с различными подходами к распараллеливанию.

## Файлы

1. **`parallel_filter.py`** - Асинхронная версия с использованием `asyncio` и `aiohttp`
2. **`multiprocess_filter.py`** - Версия с мультипроцессингом
3. **`benchmark_filters.py`** - Скрипт для сравнения производительности
4. Ваш оригинальный скрипт (для сравнения)

## Основные улучшения

### 1. Асинхронная версия (`parallel_filter.py`)

**Преимущества:**
- Эффективно для I/O-bound задач (API вызовы)
- Низкое потребление памяти
- Хорошо масштабируется для HTTP API

**Новые параметры:**
- `--max-concurrent N` - максимальное количество одновременных вызовов LLM (по умолчанию 10)
- `--batch-size N` - размер батча для обработки (по умолчанию 100)

**Пример использования:**
```bash
python parallel_filter.py \
    --input dialogs.jsonl \
    --output filtered.jsonl \
    --backend openai \
    --model-path gpt-3.5-turbo \
    --api-key YOUR_API_KEY \
    --api-base https://api.openai.com/v1 \
    --max-concurrent 15 \
    --batch-size 50
```

### 2. Мультипроцессинг версия (`multiprocess_filter.py`)

**Преимущества:**
- Истинный параллелизм (обходит GIL)
- Хорошо для CPU-bound задач
- Стабильная производительность

**Новые параметры:**
- `--num-processes N` - количество процессов (по умолчанию 4)
- `--batch-size N` - размер батча для обработки (по умолчанию 100)

**Пример использования:**
```bash
python multiprocess_filter.py \
    --input dialogs.jsonl \
    --output filtered.jsonl \
    --backend openai \
    --model-path gpt-3.5-turbo \
    --api-key YOUR_API_KEY \
    --api-base https://api.openai.com/v1 \
    --num-processes 8 \
    --batch-size 100
```

### 3. Для sglang

Обе версии поддерживают sglang:

```bash
python parallel_filter.py \
    --input dialogs.jsonl \
    --output filtered.jsonl \
    --backend sglang \
    --model-path /path/to/model \
    --tp 2 \
    --dp-size 1 \
    --max-concurrent 8
```

## Бенчмарк

### Создание тестового датасета

```bash
python benchmark_filters.py \
    --input test_data.jsonl \
    --create-test-data 1000
```

### Запуск сравнения

```bash
python benchmark_filters.py \
    --input test_data.jsonl \
    --backend openai \
    --model-path gpt-3.5-turbo \
    --api-key YOUR_API_KEY \
    --api-base https://api.openai.com/v1
```

Результат покажет сравнение производительности всех методов.

## Рекомендации по выбору метода

### Для OpenAI/OpenRouter API:
- **Асинхронная версия** (`parallel_filter.py`) - лучший выбор
- Рекомендуемые настройки: `--max-concurrent 10-20`, `--batch-size 50-100`

### Для sglang (локальный сервер):
- **Мультипроцессинг** (`multiprocess_filter.py`) может быть эффективнее
- Рекомендуемые настройки: `--num-processes 4-8`, `--batch-size 100`

### Для больших датасетов:
- Используйте больший `--batch-size` (200-500)
- Мониторьте использование памяти
- Для API с лимитами - уменьшите `--max-concurrent`

## Особенности реализации

### Обработка ошибок
- Автоматическое восстановление после ошибок API
- Логирование проблемных диалогов
- Сохранение частичных результатов

### Мониторинг прогресса
- Real-time отображение прогресса
- Подсчет скорости обработки
- Логирование времени выполнения батчей

### Оптимизации
- Батчевая обработка для уменьшения overhead
- Семафоры для контроля нагрузки
- Переиспользование HTTP соединений (aiohttp)

## Требования

Дополнительные зависимости для новых версий:

```bash
pip install aiohttp tqdm
```

Все остальные зависимости остаются теми же, что и в оригинальном скрипте.

## Производительность

Ожидаемое ускорение (по сравнению с оригинальным последовательным подходом):

- **Асинхронная версия**: 5-15x для API вызовов
- **Мультипроцессинг**: 2-8x в зависимости от количества процессов
- **Реальное ускорение** зависит от:
  - Скорости API/модели
  - Размера диалогов
  - Пропускной способности сети
  - Аппаратных ресурсов

## Отладка

Для отладки используйте флаг `--debug`:

```bash
python parallel_filter.py --debug --input test.jsonl --output out.jsonl ...
```

Это включит подробное логирование времени выполнения и ошибок.