import argparse
import json
import os
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import logging
import subprocess
from utils.completion import registered_api_completion, API_ERROR_OUTPUT
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import is_in_ci
from sglang.utils import terminate_process, wait_for_server
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict, Any, Optional

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


class ParallelLLMCaller:
    """Класс для параллельных вызовов LLM с различными бэкендами"""
    
    def __init__(self, model_type: str, model_path: str, api_key: Optional[str] = None, 
                 api_base: Optional[str] = None, port: int = 30000, 
                 max_concurrent: int = 10, temperature: float = 0.2, 
                 max_new_tokens: int = 1024):
        self.model_type = model_type
        self.model_path = model_path
        self.api_key = api_key
        self.api_base = api_base
        self.port = port
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Для синхронных вызовов (например, sglang через registered_api_completion)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def call_llm_async(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Асинхронный вызов LLM"""
        async with self.semaphore:
            try:
                if self.model_type in ("openai", "openrouter"):
                    return await self._call_openai_async(messages)
                elif self.model_type == "sglang":
                    # Для sglang используем ThreadPoolExecutor для синхронных вызовов
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.executor, self._call_sglang_sync, messages
                    )
                else:
                    raise ValueError(f"Unknown model_type: {self.model_type}")
            except Exception as e:
                logging.error(f"Error calling LLM: {e}")
                return None
    
    def _call_sglang_sync(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Синхронный вызов sglang"""
        try:
            api_func = registered_api_completion.get("sglang_http")
            api_dict = {"api_base": self.api_base, "port": self.port}
            result = api_func(
                self.model_path, messages, api_dict=api_dict, 
                max_new_tokens=self.max_new_tokens, temperature=self.temperature
            )
            return result if result and result is not API_ERROR_OUTPUT else None
        except Exception as e:
            logging.error(f"Error in sglang call: {e}")
            return None
    
    async def _call_openai_async(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Асинхронный вызов OpenAI/OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base}/chat/completions", 
                    headers=headers, 
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "answer": result["choices"][0]["message"]["content"]
                        }
                    else:
                        logging.error(f"API call failed with status {response.status}")
                        return None
            except Exception as e:
                logging.error(f"Error in OpenAI API call: {e}")
                return None


async def process_dialog_batch(llm_caller: ParallelLLMCaller, 
                              dialog_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Обработка батча диалогов"""
    tasks = []
    
    for dialog in dialog_batch:
        dialogue_text = build_dialogue_text(dialog)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dialogue_text}
        ]
        
        task = asyncio.create_task(
            process_single_dialog(llm_caller, dialog, messages, dialogue_text)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Обработка исключений
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Error processing dialog: {result}")
            # Создаем диалог с ошибкой
            error_dialog = {
                "acceptable": False,
                "reason": f"#ERROR_PROCESSING#: {str(result)}",
                "complexity": "#ERROR_PROCESSING#",
                "estimate": "#ERROR_PROCESSING#",
                "detailed_topic": "#ERROR_PROCESSING#",
                "general_topic": "#ERROR_PROCESSING#",
                "request": "#ERROR_PROCESSING#"
            }
            processed_results.append(error_dialog)
        else:
            processed_results.append(result)
    
    return processed_results


async def process_single_dialog(llm_caller: ParallelLLMCaller, 
                               dialog: Dict[str, Any], 
                               messages: List[Dict[str, Any]], 
                               dialogue_text: str) -> Dict[str, Any]:
    """Обработка одного диалога"""
    result = await llm_caller.call_llm_async(messages)
    
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


def read_dialogs_in_chunks(file_path: str, chunk_size: int = 1000):
    """Генератор для чтения диалогов порциями"""
    chunk = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                dialog = json.loads(line)
                chunk.append(dialog)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            except Exception:
                continue
        
        if chunk:  # Последняя порция
            yield chunk


async def main_async(args):
    """Основная асинхронная функция"""
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
    
    # Создаем объект для параллельных вызовов
    llm_caller = ParallelLLMCaller(
        model_type=args.backend,
        model_path=args.model_path,
        api_key=args.api_key,
        api_base=args.api_base,
        port=args.port,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    try:
        with open(args.output, "w", encoding="utf-8") as fout:
            # Подсчитываем общее количество диалогов для прогресс-бара
            total_dialogs = sum(1 for _ in open(args.input, "r", encoding="utf-8"))
            
            # Обрабатываем диалоги порциями
            processed_count = 0
            
            async for chunk in async_read_dialogs_in_chunks(args.input, args.batch_size):
                if args.debug:
                    logging.info(f"Processing batch of {len(chunk)} dialogs")
                
                # Обрабатываем батч
                processed_dialogs = await process_dialog_batch(llm_caller, chunk)
                
                # Записываем результаты
                for dialog in processed_dialogs:
                    fout.write(json.dumps(dialog, ensure_ascii=False) + "\n")
                    processed_count += 1
                
                if args.debug:
                    logging.info(f"Processed {processed_count}/{total_dialogs} dialogs")
                
                # Показываем прогресс
                print(f"Processed: {processed_count}/{total_dialogs} dialogs", end='\r')
    
    finally:
        # Останавливаем сервер
        if server_process and args.backend == "sglang":
            try:
                logging.info("Остановка sglang сервера...")
                terminate_process(server_process)
            except subprocess.CalledProcessError as e:
                logging.error(f"Не удалось остановить sglang сервер: {e}")


async def async_read_dialogs_in_chunks(file_path: str, chunk_size: int = 1000):
    """Асинхронный генератор для чтения диалогов порциями"""
    chunk = []
    
    def read_chunk():
        nonlocal chunk
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    dialog = json.loads(line)
                    chunk.append(dialog)
                    if len(chunk) >= chunk_size:
                        result = chunk[:]
                        chunk = []
                        return result
                except Exception:
                    continue
        
        if chunk:
            result = chunk[:]
            chunk = []
            return result
        return None
    
    loop = asyncio.get_event_loop()
    
    while True:
        batch = await loop.run_in_executor(None, read_chunk)
        if batch is None:
            break
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
    
    # Новые параметры для распараллеливания
    parser.add_argument("--max-concurrent", type=int, default=10, 
                       help="Maximum number of concurrent LLM calls")
    parser.add_argument("--batch-size", type=int, default=100, 
                       help="Number of dialogs to process in one batch")
    
    args = parser.parse_args()
    
    # Запускаем асинхронную версию
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()