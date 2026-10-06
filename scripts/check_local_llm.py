#!/usr/bin/env python3
"""Interactive, logged local Ollama chat, plus three reproducible checks."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
OUTPUT = Path(__file__).resolve().parents[1] / ".local/local-llm/report.json"
OPENER = build_opener(ProxyHandler({}))


def api(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=300) as response:
        return json.load(response)


def run_checks():
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "model": MODEL,
        "version": api("/api/version"),
        "installed_models": api("/api/tags"),
        "results": [],
    }
    cases = [
        ("simple", 'Какой город является столицей Франции? Название города на русском языке в поле capital.', {"type": "object", "properties": {"capital": {"type": "string"}}, "required": ["capital"], "additionalProperties": False}),
        ("structured", 'Извлеки данные из текста: «Анна оплатила 1250 рублей 5 октября 2026 года». Верни только JSON с ключами name, date (YYYY-MM-DD), amount (число), currency (RUB).', "json"),
        ("reasoning", 'В магазине купили 17 книг по 240 рублей. На книги действует скидка 15%. Доставка стоит 350 рублей, скидка на неё не действует. Верни JSON: subtotal, discount, after_discount, delivery, total (все числа), explanation (краткое объяснение по-русски).', "json"),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for level, prompt, output_format in cases:
        prompt += "\n/no_think"
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {"num_ctx": 2048, "num_predict": 512, "temperature": 0},
        }
        if output_format:
            payload["format"] = output_format
        print(f"Running {level}...", flush=True)
        print("[HTTP → Ollama] POST " + BASE_URL + "/api/chat")
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        started = time.monotonic()
        response = api("/api/chat", payload)
        print("[HTTP ← Ollama] Полный ответ API:")
        print(json.dumps(response, ensure_ascii=False, indent=2), flush=True)
        answer = response.get("message", {}).get("content", "")
        complete = response.get("done") is True and response.get("done_reason") == "stop"
        passed = bool(answer.strip()) and complete
        try:
            parsed = json.loads(answer)
            expected = ({"capital": "Париж"} if level == "simple" else
                        {"name": "Анна", "date": "2026-10-05", "amount": 1250, "currency": "RUB"}
                        if level == "structured" else
                        {"subtotal": 4080, "discount": 612, "after_discount": 3468, "delivery": 350, "total": 3818})
            passed = passed and isinstance(parsed, dict) and all(parsed.get(k) == v for k, v in expected.items())
        except (ValueError, TypeError):
            passed = False
        report["results"].append({"level": level, "request": payload, "seconds": round(time.monotonic() - started, 3), "passed": passed, "response": response})
        report["running_models"] = api("/api/ps")
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'}: {answer}", flush=True)
    print(f"Report: {OUTPUT}")
    return 0 if all(result["passed"] for result in report["results"]) else 1


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Число должно быть больше нуля")
    return number


def stream_reply(payload, log_path):
    print("\n[HTTP → Ollama] POST " + BASE_URL + "/api/chat")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    request = Request(BASE_URL + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    started = time.monotonic()
    record = {"created_at": datetime.now(timezone.utc).isoformat(),
              "url": BASE_URL + "/api/chat", "request": payload, "chunks": []}
    thinking, content, last_channel, final = "", "", None, None
    try:
        with OPENER.open(request, timeout=300) as response:
            print(f"[HTTP ← Ollama] Статус {response.status}; поток ответа:", flush=True)
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                record["chunks"].append(chunk)
                if chunk.get("error"):
                    raise ValueError(chunk["error"])
                message = chunk.get("message", {})
                for channel in ("thinking", "content"):
                    fragment = message.get(channel, "")
                    if not fragment:
                        continue
                    if last_channel != channel:
                        label = "Рассуждения модели" if channel == "thinking" else "Ответ модели / исходный текст"
                        print(f"\n[{label}]", flush=True)
                        last_channel = channel
                    print(fragment, end="", flush=True)
                    if channel == "thinking":
                        thinking += fragment
                    else:
                        content += fragment
                if chunk.get("done"):
                    final = chunk
                    break
        if final is None:
            raise ValueError("Поток оборвался без done=true; ответ не считается завершённым")
        print(f"\n\n[Итог] Модель: {final.get('model')}; остановка: {final.get('done_reason')}; "
              f"токенов: {final.get('eval_count')}; время: {time.monotonic() - started:.2f} с", flush=True)
        if final.get("done_reason") != "stop":
            print("[Предупреждение] Генерация ограничена или остановлена: ответ может быть неполным.")
        if not thinking:
            print("[Рассуждения] Отдельное поле thinking не пришло. Текст content показан без изменений.")
        if not content.strip():
            print("[Ответ отсутствует] Модель не выдала финальный текст. Увеличьте --max-tokens или используйте /think off.")
            return None
        return {"role": "assistant", "content": content}
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record.update({"seconds": round(time.monotonic() - started, 3),
                       "done": final is not None, "thinking": thinking, "content": content})
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat(args):
    version = api("/api/version")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = OUTPUT.parent / f"chat-{stamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Локальный чат: {MODEL}, Ollama {version.get('version')}, {BASE_URL}")
    print("Вопросы вводите после «Вы >». /clear — очистить историю; /exit — выйти.")
    print("/think on и /think off — включить или выключить рассуждения в следующих запросах.")
    print("Показываются реальные запросы, текст рассуждений из API и ответы. Ответы не подставляются.")
    print("Если модель включит <think> в content, эти теги также будут видны.")
    print(f"Журнал запросов и исходных частей ответа: {log_path}")
    print(f"Рассуждения: {'выключены' if args.no_think else 'включены'}; лимит ответа: {args.max_tokens} токенов.")
    history = []
    while True:
        if args.prompt is not None:
            prompt = args.prompt
            print(f"\nВы > {prompt}")
        else:
            try:
                prompt = input("\nВы > ").strip()
            except EOFError:
                print("\nЧат завершён.")
                return 0
        if prompt == "/exit":
            return 0
        if prompt in ("/think on", "/think off"):
            args.no_think = prompt == "/think off"
            print(f"Рассуждения {'выключены' if args.no_think else 'включены'} для следующих запросов.")
            if args.prompt is not None:
                return 0
            continue
        if prompt == "/clear":
            history.clear()
            print("История очищена; прежние запросы остаются в журнале.")
            if args.prompt is not None:
                return 0
            continue
        if not prompt.strip():
            if args.prompt is not None:
                raise ValueError("Вопрос не должен быть пустым")
            continue
        messages = history + [{"role": "user", "content": prompt}]
        payload = {"model": MODEL, "messages": messages, "stream": True,
                   "think": not args.no_think, "keep_alive": "30m",
                   "options": {"num_ctx": 2048, "num_predict": args.max_tokens, "temperature": 0.6}}
        reply = stream_reply(payload, log_path)
        history = messages + ([reply] if reply else [])
        if args.prompt is not None:
            return 0 if reply else 1


def main():
    parser = argparse.ArgumentParser(description="Локальный чат с Qwen3 4B: запросы, рассуждения и ответы в консоли")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Три контрольных запроса вместо интерактивного чата")
    mode.add_argument("--prompt", help="Один произвольный вопрос вместо интерактивного ввода")
    parser.add_argument("--no-think", action="store_true", help="Запросить отключение рассуждений в чате")
    parser.add_argument("--max-tokens", type=positive_int, default=1024, help="Лимит генерации в чате (по умолчанию 1024)")
    args = parser.parse_args()
    return run_checks() if args.check else chat(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        print(f"Local Ollama check failed: {error}", file=sys.stderr)
        print("Проверьте, что Ollama запущена (ollama serve) и модель скачана (ollama pull qwen3:4b).", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nЧат остановлен пользователем.")
        sys.exit(130)
