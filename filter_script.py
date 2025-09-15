import argparse
import json
import os
from tqdm import tqdm
import logging
import subprocess
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
from utils.completion import registered_api_completion, API_ERROR_OUTPUT
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import is_in_ci
from sglang.utils import terminate_process, wait_for_server

import logging.config
from config import LOG_CONFIG

def setup_logging(config: dict) -> None:
    """Configuring logging"""
    logging.config.dictConfig(config)

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

# system_prompt для фильтрации
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
    # Универсальный вызов модели через registered_api_completion
    if model_type == "sglang":
        api_func = registered_api_completion.get("sglang_http")
        api_dict = {"api_base": api_base, "port": port}
        return api_func(model_path, messages, api_dict=api_dict, max_new_tokens=max_new_tokens, temperature=temperature)
    elif model_type in ("openai", "openrouter"):
        api_func = registered_api_completion.get("openai")
        api_dict = {"api_base": api_base, "api_key": api_key}
        return api_func(model_path, messages, temperature=0.0, max_tokens=max_new_tokens, api_dict=api_dict)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def process_dialog(dialog_data):
    """
    Worker function to process a single dialog.
    This function will be executed in parallel by multiple processes.
    """
    dialog, args = dialog_data
    
    try:
        dialogue_text = build_dialogue_text(dialog)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dialogue_text}
        ]
        
        result = call_llm(
            args.backend,
            args.model_path,
            args.api_key,
            args.api_base,
            args.port,
            messages,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )
        
        if not result or result is API_ERROR_OUTPUT:
            dialog["acceptable"] = False
            dialog["reason"] = "#API_ERROR#"
            dialog["complexity"] = "#API_ERROR#"
            dialog["estimate"] = "#API_ERROR#"
            dialog["detailed_topic"] = "#API_ERROR#"
            dialog["general_topic"] = "#API_ERROR#"
            dialog["request"] = dialogue_text
            return dialog
        
        # Парсим ответ модели (ожидаем JSON)
        answer = result["answer"] if isinstance(result, dict) and "answer" in result else result
        
        if isinstance(answer, dict):
            filter_json = answer
        else:
            if isinstance(answer, str):
                if answer.startswith("```json"):
                    answer = answer[len("```json"): -len("```")].strip()
                else:
                    answer = answer.strip()
            
            try:
                filter_json = json.loads(answer)
            except Exception as e:
                dialog["acceptable"] = False
                dialog["reason"] = "#ERROR_PARSE#"
                dialog["complexity"] = "#ERROR_PARSE#"
                dialog["estimate"] = "#ERROR_PARSE#"
                dialog["detailed_topic"] = "#ERROR_PARSE#"
                dialog["general_topic"] = "#ERROR_PARSE#"
                dialog["request"] = dialogue_text
                return dialog
        
        try:
            dialog["acceptable"] = filter_json.get("acceptable")
            dialog["reason"] = filter_json.get("reason")
            dialog["complexity"] = filter_json.get("complexity")
            dialog["estimate"] = filter_json.get("estimate")
            dialog["detailed_topic"] = filter_json.get("detailed_topic")
            dialog["general_topic"] = filter_json.get("general_topic")
            dialog["request"] = dialogue_text
        except Exception:
            dialog["acceptable"] = False
            dialog["reason"] = "#ERROR_PARSE_KEYS#"
            dialog["complexity"] = "#ERROR_PARSE_KEYS#"
            dialog["estimate"] = "#ERROR_PARSE_KEYS#"
            dialog["detailed_topic"] = "#ERROR_PARSE_KEYS#"
            dialog["general_topic"] = "#ERROR_PARSE_KEYS#"
            dialog["request"] = dialogue_text
        
        return dialog
        
    except Exception as e:
        dialog["acceptable"] = False
        dialog["reason"] = f"#PROCESSING_ERROR: {str(e)}#"
        dialog["complexity"] = "#PROCESSING_ERROR#"
        dialog["estimate"] = "#PROCESSING_ERROR#"
        dialog["detailed_topic"] = "#PROCESSING_ERROR#"
        dialog["general_topic"] = "#PROCESSING_ERROR#"
        dialog["request"] = build_dialogue_text(dialog) if "dialog" in locals() else "#ERROR#"
        return dialog

def load_dialogs(input_file):
    """Load all dialogs from input file."""
    dialogs = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                dialog = json.loads(line)
                dialogs.append(dialog)
            except Exception:
                continue
    return dialogs

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
    parser.add_argument("--num-processes", type=int, default=4, help="Number of parallel processes for API calls")
    args = parser.parse_args()

    setup_logging(LOG_CONFIG)

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
        except subprocess.CalledProcessError as e:
            logging.error(f"Не удалось запустить sglang сервер: {e}")
            raise

        logging.info("Ждем запуска sglang сервера...")
        wait_for_server(f"http://0.0.0.0:{port}")
        logging.info("sglang сервер запущен...")
        args.port = port 
    
    # Load all dialogs
    logging.info(f"Loading dialogs from {args.input}...")
    dialogs = load_dialogs(args.input)
    logging.info(f"Loaded {len(dialogs)} dialogs")
    
    # Prepare data for multiprocessing
    dialog_data = [(dialog, args) for dialog in dialogs]
    
    # Process dialogs in parallel
    logging.info(f"Processing dialogs with {args.num_processes} processes...")
    
    try:
        with Pool(processes=args.num_processes) as pool:
            # Use imap for progress tracking
            processed_dialogs = list(tqdm(
                pool.imap(process_dialog, dialog_data),
                total=len(dialog_data),
                desc="Filtering"
            ))
        
        # Write results to output file
        logging.info(f"Writing results to {args.output}...")
        with open(args.output, "w", encoding="utf-8") as fout:
            for dialog in processed_dialogs:
                fout.write(json.dumps(dialog, ensure_ascii=False) + "\n")
        
        logging.info(f"Processing completed. Results written to {args.output}")
        
    except Exception as e:
        logging.error(f"Error during multiprocessing: {e}")
        raise
    
    finally:
        if args.backend == "sglang":
            try:
                logging.info("Остановка sglang сервера...")
                terminate_process(server_process)
            except subprocess.CalledProcessError as e:
                logging.error(f"Не удалось остановить sglang сервер: {e}")

if __name__ == "__main__":
    main()