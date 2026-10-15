# HTTP API и данные

Источники контрактов — `internal/models/models.go`, `internal/handlers/handlers.go` и регистрации маршрутов в `main.go`. Этот документ описывает реализованный интерфейс; схемы MCP получают через живой `tools/list`.

## Маршруты

| Маршрут | Методы | Назначение |
|---|---|---|
| `/api/agent/chat` | GET, POST, PATCH, DELETE | Состояние, сообщение, явная команда, очистка диалога |
| `/api/agent/models` | GET | Каталоги моделей основного чата |
| `/api/mcp/tools` | GET | Каталог MCP-серверов и инструментов |
| `/api/mcp/call` | POST | Диагностический вызов MCP |
| `/api/chat` | POST | Ответ с выбранными ограничениями формата |
| `/api/reasoning` | POST | Сравнение способов построения ответа |
| `/api/model-versions` | POST | Сравнение моделей на фиксированном запросе |
| `/api/agent/token-demo` | GET, POST | Сценарий измерения контекста |
| `/api/agent/strategy-demo` | GET, POST | Сравнение стратегий истории |
| `/api/agent/branching-demo` | GET, POST | Сценарий ветвления диалога |

## Сообщение агенту

```json
{
  "message": "/help Как устроен проект?",
  "recentMessages": 4,
  "strategy": "sliding_window",
  "model": "deepseek-flash",
  "ragEnabled": false,
  "ragOptions": {"candidateLimit": 8, "resultLimit": 3, "minSimilarity": 0.35, "rewrite": true, "rerank": true},
  "settings": {"temperature": 0.2, "maxTokens": 2048}
}
```

`/help` без вопроса возвращает инструкцию без генерации. `/help <вопрос>` и `/structure` включают RAG независимо от `ragEnabled` только для текущего вызова. Последующие обычные сообщения используют переданный `ragEnabled`; сохраняемого режима помощи нет. `/reindex` запускает индексатор, `/structure` задаёт вопрос о каталогах. `/diff` без аргументов показывает текущую git-ветку через MCP; ответ содержит `toolExecutions` и источник `MCP:project_git_branch`, а `ragModelCallSkipped=true`. `/help` и `/structure` Git MCP не вызывают. Остальные неизвестные slash-команды отклоняются без вызова модели. Разрешён один JSON-объект, неизвестные поля отклоняются. Идентификаторы моделей и настройки проверяются backend.

Cookie `agent_user` отделяет пользовательскую память, `agent_session` — диалог. Cookie HttpOnly и SameSite=Lax. Эти cookie служат локальной идентификацией состояния, а не полноценной системой авторизации.

## Ответ

`AgentResponse` содержит `answer`, `messages`, `model`, `settings`, `strategy`, `recentMessages`, `memory`, `taskMemory`, `task` и `tokens`. При RAG добавляются `ragEnabled`, `ragSources`, `ragTrace`, `ragAbstained`, `ragReason`, `ragModelCallSkipped`. `requestMessages` показывает контекст запроса; `toolExecutions` — реальные вызовы MCP; `localMetrics` — метрики Ollama.

Документальный `RAGSource` содержит `source`, `section`, `chunkId`, `quote`, `score` и оценки поиска. MCP-источник отмечается отдельным именем `MCP:project_git_branch` или другого вызванного инструмента. В ответе `/diff` MCP-источник описывает фактически прочитанную ветку; поиск документов и вызов модели не выполняются.

## Команды состояния

PATCH принимает `ContextCommand` с `action` и соответствующими аргументами. Примеры действий: `set_recent_messages`, `set_strategy`, `set_model`, `set_generation_settings`, `create_profile`, `set_active_profile`, `checkpoint`, `create_branch`, `switch_branch`, `save_memory`, `delete_memory`, `configure_task`, `approve_plan`, `pause_task`, `resume_task`, `set_planner_mode`, `save_invariant`.

```json
{"action":"set_recent_messages","recentMessages":4}
```

Переключение ветки диалога через PATCH не переключает git-ветку. DELETE очищает диалог согласно правилам памяти; GET не выполняет генерацию.

## Вызов Project MCP

```json
{"name":"project_git_branch","arguments":{}}
```

Ответ `/api/mcp/call` содержит `name`, `isError` и `content`. Content — сериализованный результат с `branch`, `detached`, `commit`. Ключи и параметры внешних подключений браузеру не возвращаются.

## Результат переиндексации

Для POST `/api/agent/chat` с `message: "/reindex"` ответ после успешной операции содержит `reindex`: `files`, `fixedChunks`, `structureChunks`, `embeddingModel`, `durationMs`. Эти значения получены из фактической сборки. Генерация пропускается (`ragModelCallSkipped=true`); при ошибке поле `reindex` отсутствует и текст явно сообщает о неуспехе. Лимит HTTP-запроса переиндексации — десять минут, независимо от выбранной модели.

## Данные на диске

| Путь | Содержание |
|---|---|
| `.local/agent-history.json` | Пользователи, диалоги, настройки, память и состояние задач |
| `.local/rag/index-structure.json` | Файлы с хешами и структурные чанки с векторами |
| `.local/rag/index-fixed.json` | Альтернативное оконное разбиение |
| `.local/rag/report.json` | Состав корпуса и статистика индексации |
| `.local/weather-observations.json` | Измерения и расписание погоды |
| `.local/rag-eval/` | Результаты отдельной оценки ответов |

Файл истории хранит внутреннее состояние и не является публичным REST-контрактом для ручного редактирования. Для тестов используйте отдельные временные каталоги и сессии.
