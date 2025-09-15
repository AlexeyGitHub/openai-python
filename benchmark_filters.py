#!/usr/bin/env python3
"""
Скрипт для бенчмарка различных подходов к распараллеливанию фильтрации диалогов
"""

import argparse
import time
import subprocess
import json
import os
from typing import List, Dict

def run_filter_script(script_path: str, args: Dict[str, str], timeout: int = 3600) -> Dict[str, float]:
    """Запуск скрипта фильтрации и измерение времени"""
    
    # Формируем команду
    cmd = ["python", script_path]
    for key, value in args.items():
        if key.startswith("--"):
            cmd.append(key)
            if value is not None:
                cmd.append(str(value))
        else:
            cmd.append(f"--{key}")
            if value is not None:
                cmd.append(str(value))
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode == 0:
            execution_time = end_time - start_time
            
            # Подсчитываем количество обработанных диалогов
            output_file = args.get("output") or args.get("--output")
            if output_file and os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    processed_count = sum(1 for _ in f)
            else:
                processed_count = 0
            
            return {
                "execution_time": execution_time,
                "processed_count": processed_count,
                "dialogs_per_second": processed_count / execution_time if execution_time > 0 else 0,
                "success": True,
                "error": None
            }
        else:
            return {
                "execution_time": end_time - start_time,
                "processed_count": 0,
                "dialogs_per_second": 0,
                "success": False,
                "error": result.stderr
            }
    
    except subprocess.TimeoutExpired:
        return {
            "execution_time": timeout,
            "processed_count": 0,
            "dialogs_per_second": 0,
            "success": False,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "execution_time": 0,
            "processed_count": 0,
            "dialogs_per_second": 0,
            "success": False,
            "error": str(e)
        }

def create_test_dataset(output_path: str, num_dialogs: int = 100):
    """Создание тестового датасета для бенчмарка"""
    
    sample_dialog = {
        "instruction": [
            {"role": "user", "content": "Как рассчитать НДС для товара стоимостью 1000 рублей?"}
        ],
        "output": {
            "role": "assistant", 
            "content": "Для расчета НДС используется формула: НДС = сумма * 20/120. Для товара стоимостью 1000 рублей: НДС = 1000 * 20/120 = 166.67 рублей."
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_dialogs):
            # Немного варьируем содержимое
            dialog = sample_dialog.copy()
            dialog["instruction"][0]["content"] = f"Вопрос {i+1}: " + dialog["instruction"][0]["content"]
            f.write(json.dumps(dialog, ensure_ascii=False) + '\n')
    
    print(f"Created test dataset with {num_dialogs} dialogs: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark different filtering approaches")
    parser.add_argument("--input", required=True, help="Input test dataset")
    parser.add_argument("--create-test-data", type=int, help="Create test dataset with N dialogs")
    parser.add_argument("--backend", default="openai", choices=["sglang", "openai", "openrouter"])
    parser.add_argument("--model-path", default="gpt-3.5-turbo")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    # Создаем тестовый датасет если нужно
    if args.create_test_data:
        create_test_dataset(args.input, args.create_test_data)
        return
    
    # Базовые аргументы для всех скриптов
    base_args = {
        "input": args.input,
        "backend": args.backend,
        "model-path": args.model_path,
        "api-key": args.api_key,
        "api-base": args.api_base,
    }
    
    # Конфигурации для тестирования
    test_configs = [
        {
            "name": "Original (Sequential)",
            "script": "filter.py",  # Ваш оригинальный скрипт
            "args": {**base_args, "output": "output_original.jsonl"}
        },
        {
            "name": "Async (10 concurrent)",
            "script": "parallel_filter.py",
            "args": {**base_args, "output": "output_async_10.jsonl", "max-concurrent": 10}
        },
        {
            "name": "Async (20 concurrent)",
            "script": "parallel_filter.py", 
            "args": {**base_args, "output": "output_async_20.jsonl", "max-concurrent": 20}
        },
        {
            "name": "Multiprocess (4 processes)",
            "script": "multiprocess_filter.py",
            "args": {**base_args, "output": "output_mp_4.jsonl", "num-processes": 4}
        },
        {
            "name": "Multiprocess (8 processes)",
            "script": "multiprocess_filter.py",
            "args": {**base_args, "output": "output_mp_8.jsonl", "num-processes": 8}
        }
    ]
    
    # Запускаем бенчмарки
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        result = run_filter_script(config["script"], config["args"], args.timeout)
        result["name"] = config["name"]
        results.append(result)
        
        if result["success"]:
            print(f"✅ Success!")
            print(f"   Time: {result['execution_time']:.2f}s")
            print(f"   Processed: {result['processed_count']} dialogs")
            print(f"   Speed: {result['dialogs_per_second']:.2f} dialogs/sec")
        else:
            print(f"❌ Failed!")
            print(f"   Error: {result['error']}")
    
    # Выводим сводную таблицу
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<25} {'Time (s)':<10} {'Dialogs':<8} {'Speed (d/s)':<12} {'Status':<8}")
    print(f"{'-'*80}")
    
    for result in results:
        status = "✅ OK" if result["success"] else "❌ FAIL"
        print(f"{result['name']:<25} {result['execution_time']:<10.2f} "
              f"{result['processed_count']:<8} {result['dialogs_per_second']:<12.2f} {status:<8}")
    
    # Находим самый быстрый метод
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        fastest = max(successful_results, key=lambda x: x["dialogs_per_second"])
        print(f"\n🏆 Fastest method: {fastest['name']} "
              f"({fastest['dialogs_per_second']:.2f} dialogs/sec)")
    
    # Сохраняем результаты в JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: benchmark_results.json")

if __name__ == "__main__":
    main()