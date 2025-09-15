import argparse
import json
import os
from tqdm import tqdm
import logging
import subprocess
from utils.completion import registered_api_completion, API_ERROR_OUTPUT
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import is_in_ci
from sglang.utils import terminate_process, wait_for_server
from multiprocessing import Pool, Manager, Queue
import time
from typing import List, Dict, Any, Optional
from functools import partial

import logging.config
from config import LOG_CONFIG

def setup_logging(config: dict) -> None:
    """Configuring logging"""
    logging.config.dictConfig(config)

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

# system_prompt остается тот же
system_prompt = """### Role
Ты — строгий фильтр данных для обучения ассистентов. Оцени, подходит ли диалог логов ассистента для инструктивного SFT датасета (Supervised Fine-Tuning). В диалогах преобладает агентное повоедения для кодовых задач, учитывай что нужно отбирать лучшие агентные траектории.

### Критерии приемлемости
Прими диалог **ТОЛЬКО** если:
1. ✅ **Полезность**: Ответ релевантен запросу, содержит полезную информацию/решение
2. ✅ **Последовательность**: Если в ответе есть вызов функции, то этот вызов должен быть строго релевантен предыдущему контексту и информация в аргументах функции должна быть указана корректно.
3. ✅ **Корректность**: Факты/логика проверяемы (не "я не знаю", не галлюцинации) 
4. ✅ **Этика**: Нет токсичности/предвзятости/опасных советов/Нет паролей, ключей, токенов/Имена, Фамилии
5. ❌ **Отклони** если:
   - Ответ обобщён ("Спросите у специалиста")
   - Повторения
   - Содержит маркеры неоконченности ("...", "и т.д.")
   - Пользователь провоцировал на вредоносный ответ
   - Содержит грамматические ошибки
   - Путает разные языки
   - Односложный ответ ("Да", "Понятно")
   - Есть пароли, ключи, токены/Имена, Фамилии
   - Если не хватает контекста. Например, пользователь просит описать файлы, которых нет в контексте.
   - Если в ответе assistant называет себя конкретной моделелью или разработан конкретной компанией (ChatGPT, OpenAI, Gemini и тд) кроме Koda, которую разработала NLP-Core-Team. Если говорит, что Koda - то это правильно.
   - Tool вызван неуместно в ответе ассистента. За вызвонную функцую отвечает поле tool_calls. Ответ асисстента с tool_calls может быть промежуточным. Нельзя штрафовать в этом случае ответ за неполноту. Например, требуется оптимизировать проект. Если модель сразу не выдает оптимизации, а предлагает вызвать функцию просмотра проекта, то это тоже правильно.  

### Правила оценки
- - ВАЖНО:Сфокусируйся на последнем ответе ассистента, но этот ответ должен согласовываться с остальным диалогом. Если сам предыдущий диалог некорректен, то тоже acceptable - false

В estimate проставь оценку от 0 до 10 качеству диалогу. Оцени с точки зрения приведенных критериев и любых других проблем,которые посчитаешь нужными. 
В complexity оцени сложность запроса.
В general_topic ты должна классифицировать, относится ли диалог к написанию кода или он на любую тематику связанную с кодом и разработкой.

### Output Format
Выведи JSON:
{
"acceptable": true/false,
"complexity": "easy/medium/hard",
"estimate": "0-10",
"detailed_topic": "Код про финансы/ Код про трейдинг и др. коротко более подробная тема запроса в свободной форме",
"general_topic": "Code/No Code; (выбрать только одну из только приведенных тем запроса Code или No Code)",
"reason": "Напиши на русском причины твоего решения. 1-4 предложения. Для false укажи нарушенный критерий"
}
### Примеры
Диалог 1:
[USER] Как рассчитать НДС?  
[ASSISTANT] Формула: НДС = сумма * 20/120. Пример: для 1000 руб. НДС 166.67 руб.  
→ {"acceptable": true, "complexity": "medium", "estimate": "10", "detailed_topic": "Финансы", "general_topic": "No Code", "reason": "Конкретный, проверяемый ответ"}

Диалог 2:  
[USER] Как взломать Wi-Fi?  
[ASSISTANT] Попробуйте угадать пароль...  
→ {"acceptable": false, "complexity": "medium", "estimate": "0", "detailed_topic": "Разработка ПО", "general_topic": "No Code", "reason": "Нарушение этики: опасный совет"}

Диалог 3:  
[USER] Оптимизируй текущий проект
[ASSISTANT] REASONING_START\nХорошо, чтобы оптимизировать проект мне нужно посмотреть из чего состоит проект\nREASONING_END\n\nХорошо, я помогу вам оптимизировать ваш проект. Для начала мне нужно изучить структуру проекта. Я выведу список файлов, чтобы понять, с чего начать.\ntool_calls: [{"type": "function", "function": {"name": "builtin_ls", "arguments": {"dirPath": "/", "recursive": true}}}]  
→ {"acceptable": true, "complexity": "easy", "estimate": "10", "detailed_topic": "Оптимизация кода", "general_topic": "Code", "reason": "Функция builtin_ls вызвана правильно и соответствует контексту. Шаг просмотра файлов директории необходим для решения задачи"}

Диалог 4:  
[USER] Кто изобрел динамит?  
[ASSISTANT] Не знаю.
[USER] Почему небо синее?  
[ASSISTANT] Это сложный вопрос, спросите учёного.  
→ {"acceptable": false, "complexity": "easy", "estimate": "0", "detailed_topic": "Разное", "general_topic": "No Code", "reason": "Не полезен: уклонение от ответа"}"""


def build_dialogue_text(dialog):
    """Построение текста диалога"""
    turns = dialog["instruction"]
    if turns[0]["role"] == "system":
        turns = turns[1:]
    assistant = dialog["output"]
    text = ""
    for turn in turns:
        role = turn.get("role", "user").upper()
        if "content" in turn and turn["content"]:
            text += f"[{role}] {turn['content']}\n"
        if "tool_calls" in turn and turn["tool_calls"]:
            text += f"tool_calls: {json.dumps(turn['tool_calls'], ensure_ascii=False)}\n"
    if assistant and "role" in assistant:
        role = assistant["role"].upper()
        if "content" in assistant and assistant["content"]:
            text += f"[{role}] {assistant['content']}\n"
        if "tool_calls" in assistant and assistant["tool_calls"]:
            text += f"tool_calls: {json.dumps(assistant['tool_calls'], ensure_ascii=False)}\n"

    text = text.strip().replace("<think>", "REASONING_START").replace("</think>", "REASONING_END")
    return text.strip()


def call_llm(model_type, model_path, api_key, api_base, port, messages, temperature=0.2, max_new_tokens=1024):
    """Универсальный вызов модели через registered_api_completion"""
    try:
        if model_type == "sglang":
            api_func = registered_api_completion.get("sglang_http")
            api_dict = {"api_base": api_base, "port": port}
            return api_func(model_path, messages, api_dict=api_dict, max_new_tokens=max_new_tokens, temperature=temperature)
        elif model_type in ("openai", "openrouter"):
            api_func = registered_api_completion.get("openai")
            api_dict = {"api_base": api_base, "api_key": api_key}
            return api_func(model_path, messages, temperature=temperature, max_tokens=max_new_tokens, api_dict=api_dict)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    except Exception as e:
        logging.error(f"Error in call_llm: {e}")
        return None


def process_dialog_worker(dialog_data):
    """Функция-воркер для обработки одного диалога в отдельном процессе"""
    dialog, model_config = dialog_data
    
    dialogue_text = build_dialogue_text(dialog)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dialogue_text}
    ]
    
    try:
        result = call_llm(
            model_config["model_type"],
            model_config["model_path"],
            model_config.get("api_key"),
            model_config.get("api_base"),
            model_config.get("port", 30000),
            messages,
            temperature=model_config.get("temperature", 0.2),
            max_new_tokens=model_config.get("max_new_tokens", 1024)
        )
        
        if not result or result is API_ERROR_OUTPUT:
            dialog.update({
                "acceptable": False,
                "reason": "#NO_RESULT#",
                "complexity": "#NO_RESULT#",
                "estimate": "#NO_RESULT#",
                "detailed_topic": "#NO_RESULT#",
                "general_topic": "#NO_RESULT#",
                "request": dialogue_text
            })
            return dialog
        
        # Парсим ответ модели
        answer = result["answer"] if isinstance(result, dict) and "answer" in result else result
        
        if isinstance(answer, str):
            if answer.startswith("```json"):
                answer = answer[len("```json"): -len("```")].strip()
            else:
                answer = answer.strip()
        
        try:
            if isinstance(answer, dict):
                filter_json = answer
            else:
                filter_json = json.loads(answer)
            
            dialog.update({
                "acceptable": filter_json.get("acceptable"),
                "reason": filter_json.get("reason"),
                "complexity": filter_json.get("complexity"),
                "estimate": filter_json.get("estimate"),
                "detailed_topic": filter_json.get("detailed_topic"),
                "general_topic": filter_json.get("general_topic"),
                "request": dialogue_text
            })
        except Exception as e:
            dialog.update({
                "acceptable": False,
                "reason": f"#ERROR_PARSE#: {str(e)}",
                "complexity": "#ERROR_PARSE#",
                "estimate": "#ERROR_PARSE#",
                "detailed_topic": "#ERROR_PARSE#",
                "general_topic": "#ERROR_PARSE#",
                "request": dialogue_text
            })
        
        return dialog
        
    except Exception as e:
        logging.error(f"Error processing dialog: {e}")
        dialog.update({
            "acceptable": False,
            "reason": f"#ERROR_PROCESSING#: {str(e)}",
            "complexity": "#ERROR_PROCESSING#",
            "estimate": "#ERROR_PROCESSING#",
            "detailed_topic": "#ERROR_PROCESSING#",
            "general_topic": "#ERROR_PROCESSING#",
            "request": dialogue_text
        })
        return dialog


def read_dialogs_in_batches(file_path: str, batch_size: int = 100):
    """Генератор для чтения диалогов батчами"""
    batch = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                dialog = json.loads(line)
                batch.append(dialog)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except Exception:
                continue
        
        if batch:  # Последний батч
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jsonl file with dialogues")
    parser.add_argument("--output", required=True, help="Output jsonl file for filtered data")
    parser.add_argument("--backend", required=True, choices=["sglang", "openai", "openrouter"], help="Model type")
    parser.add_argument("--model-path", help="Path or name of the model (for sglang/openai)")
    parser.add_argument("--api-key", help="API key (for openai/openrouter)")
    parser.add_argument("--api-base", help="API base url (for openai/openrouter/sglang)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens for filter LLM")
    parser.add_argument("--tp", type=int, default=1, help="tp")
    parser.add_argument("--dp-size", type=int, default=1, help="dp")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--debug-logs", action="store_true", help="Debug mode")
    parser.add_argument("--port", type=int, default=30000, help="Port for sglang server")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for LLM")
    
    # Параметры для мультипроцессинга
    parser.add_argument("--num-processes", type=int, default=4, 
                       help="Number of processes for parallel processing")
    parser.add_argument("--batch-size", type=int, default=100, 
                       help="Number of dialogs to process in one batch")
    
    args = parser.parse_args()
    
    setup_logging(LOG_CONFIG)
    
    server_process = None
    
    # Запуск сервера для sglang
    if args.backend == "sglang":
        try:
            logging.info("Запуск sglang сервера...")
            server_process, port = launch_server_cmd(
                f"python -m sglang.launch_server \
                    --model-path {args.model_path} \
                    --skip-tokenizer-init --host 0.0.0.0 \
                    --tp {args.tp} \
                    --dp-size {args.dp_size} \
                    --quantization fp8 \
                    --kv-cache-dtype fp8_e5m2 \
                    --attention-backend fa3 \
                    --reasoning-parser qwen3"
            )
            logging.info("Ждем запуска sglang сервера...")
            wait_for_server(f"http://0.0.0.0:{port}")
            logging.info("sglang сервер запущен...")
            args.port = port
        except subprocess.CalledProcessError as e:
            logging.error(f"Не удалось запустить sglang сервер: {e}")
            raise
    
    # Конфигурация модели для передачи в воркеры
    model_config = {
        "model_type": args.backend,
        "model_path": args.model_path,
        "api_key": args.api_key,
        "api_base": args.api_base,
        "port": args.port,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens
    }
    
    try:
        # Подсчитываем общее количество диалогов
        total_dialogs = sum(1 for _ in open(args.input, "r", encoding="utf-8"))
        logging.info(f"Total dialogs to process: {total_dialogs}")
        
        processed_count = 0
        
        with open(args.output, "w", encoding="utf-8") as fout:
            # Создаем пул процессов
            with Pool(processes=args.num_processes) as pool:
                
                # Обрабатываем файл батчами
                for batch in read_dialogs_in_batches(args.input, args.batch_size):
                    if args.debug:
                        logging.info(f"Processing batch of {len(batch)} dialogs")
                    
                    # Подготавливаем данные для воркеров
                    batch_data = [(dialog, model_config) for dialog in batch]
                    
                    # Обрабатываем батч параллельно
                    start_time = time.time()
                    results = pool.map(process_dialog_worker, batch_data)
                    end_time = time.time()
                    
                    # Записываем результаты
                    for result in results:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        processed_count += 1
                    
                    if args.debug:
                        batch_time = end_time - start_time
                        logging.info(f"Batch processed in {batch_time:.2f}s, "
                                   f"avg {batch_time/len(batch):.2f}s per dialog")
                    
                    # Показываем прогресс
                    print(f"Processed: {processed_count}/{total_dialogs} dialogs "
                          f"({processed_count/total_dialogs*100:.1f}%)", end='\r')
        
        print(f"\nProcessing completed! Processed {processed_count} dialogs.")
        
    finally:
        # Останавливаем сервер
        if server_process and args.backend == "sglang":
            try:
                logging.info("Остановка sglang сервера...")
                terminate_process(server_process)
            except subprocess.CalledProcessError as e:
                logging.error(f"Не удалось остановить sglang сервер: {e}")


if __name__ == "__main__":
    main()