// Command evaluateragchat runs two 15-turn scenarios through the real local
// Ollama retriever and DeepSeek model, saving a result after every turn.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-challenge-app/internal/agent"
	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

type scenario struct {
	ID, Title string
	Turns     []string
}
type turnResult struct {
	Number          int                  `json:"number"`
	Question        string               `json:"question"`
	Answer          string               `json:"answer"`
	Sources         []models.RAGSource   `json:"sources"`
	Abstained       bool                 `json:"abstained"`
	WindowMessages  int                  `json:"windowMessages"`
	RequestMessages []models.ChatMessage `json:"requestMessages,omitempty"`
	Memory          models.TaskMemory    `json:"taskMemory"`
	RAGTrace        *models.RAGTrace     `json:"ragTrace,omitempty"`
	Checks          map[string]bool      `json:"checks"`
	Error           string               `json:"error,omitempty"`
}
type scenarioResult struct {
	ID       string       `json:"id"`
	Title    string       `json:"title"`
	Turns    []turnResult `json:"turns"`
	Complete bool         `json:"complete"`
}
type report struct {
	GeneratedAt time.Time        `json:"generatedAt"`
	Model       string           `json:"model"`
	IndexHash   string           `json:"indexHash"`
	Window      int              `json:"windowMessages"`
	Scenarios   []scenarioResult `json:"scenarios"`
}

func main() {
	root := flag.String("root", ".", "project directory")
	indexDir := flag.String("index", ".local/rag", "RAG index directory")
	outDir := flag.String("output", ".local/rag-chat-eval", "result directory")
	ollamaURL := flag.String("ollama-url", envOr("OLLAMA_URL", "http://127.0.0.1:11434"), "local Ollama URL")
	window := flag.Int("window", 4, "messages included in the prompt (2-40)")
	flag.Parse()
	if *window < 2 || *window > 40 {
		fatalIf(fmt.Errorf("window must be between 2 and 40"))
	}
	rootAbs, err := filepath.Abs(*root)
	fatalIf(err)
	key := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY"))
	if key == "" {
		key = readEnvFile(filepath.Join(rootAbs, ".env"), "DEEPSEEK_API_KEY")
	}
	if key == "" {
		fatalIf(fmt.Errorf("DEEPSEEK_API_KEY is missing from environment and %s/.env", rootAbs))
	}
	resolve := func(path string) string {
		if filepath.IsAbs(path) {
			return path
		}
		return filepath.Join(rootAbs, path)
	}
	indexPath := filepath.Join(resolve(*indexDir), "index-structure.json")
	indexData, err := os.ReadFile(indexPath)
	fatalIf(err)
	var index ragindex.Index
	fatalIf(json.Unmarshal(indexData, &index))
	chunks := make(map[string]ragindex.Chunk, len(index.Chunks))
	for _, chunk := range index.Chunks {
		chunks[chunk.ChunkID] = chunk
	}
	stamp := time.Now().UTC().Format("20060102T150405Z")
	statePath := filepath.Join(resolve(*outDir), "session-state-"+stamp+".json")
	store := agent.NewJSONStore(statePath)
	a, err := agent.NewPersistent(deepseek.NewClient(key, 120*time.Second), store)
	fatalIf(err)
	a.SetRetriever(&rag.Searcher{Root: rootAbs, IndexDir: resolve(*indexDir), Model: "embeddinggemma", Embedder: ragindex.Ollama{URL: *ollamaURL, Model: "embeddinggemma"}})
	r := report{GeneratedAt: time.Now().UTC(), Model: models.DeepSeekFlashModel, IndexHash: hash(indexData), Window: *window}
	out := resolve(*outDir)
	fatalIf(os.MkdirAll(out, 0700))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
	defer cancel()
	for _, item := range scenarios() {
		result := scenarioResult{ID: item.ID + "-" + stamp, Title: item.Title, Turns: []turnResult{}}
		for i, question := range item.Turns {
			response, callErr := a.RespondWithUserOptionsRAGConfigured(ctx, "rag-chat-eval", result.ID, question, *window, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
			turn := turnResult{Number: i + 1, Question: question, Answer: response.Answer, Sources: response.RAGSources, Abstained: response.RAGAbstained, WindowMessages: response.RecentMessages, RequestMessages: response.RequestMessages, Memory: response.TaskMemory, RAGTrace: response.RAGTrace, Checks: map[string]bool{}}
			if callErr != nil {
				turn.Error = callErr.Error()
			}
			turn.Checks["quotesExistInIndexedChunks"] = len(turn.Sources) > 0 && quotesExist(turn.Sources, chunks)
			turn.Checks["goalRetained"] = turn.Memory.Goal != "" && turn.Memory.GoalTurn == 1
			turn.Checks["memorySurvivesWindow"] = !containsUserText(turn.RequestMessages, item.Turns[0]) && strings.Contains(formatRequest(turn.RequestMessages), turn.Memory.Goal)
			if i == len(item.Turns)-1 {
				turn.Checks["updatedConstraintRetained"] = hasNoteText(turn.Memory.Constraints, expectedConstraint(item.ID))
				turn.Checks["termRetained"] = hasNoteText(turn.Memory.Terms, expectedTerm(item.ID))
			}
			result.Turns = append(result.Turns, turn)
			r.Scenarios = replaceScenario(r.Scenarios, result)
			result.Complete = len(result.Turns) == len(item.Turns)
			r.Scenarios = replaceScenario(r.Scenarios, result)
			fatalIf(writeReport(out, r))
			fmt.Printf("%s: ход %d/%d сохранён (источники: %d, ошибка: %t)\n", item.ID, i+1, len(item.Turns), len(turn.Sources), turn.Error != "")
			if callErr != nil {
				break
			}
		}
	}
}

func scenarios() []scenario {
	return []scenario{
		{ID: "safe-launch", Title: "Безопасный локальный запуск", Turns: []string{
			"Цель: подготовить инструкцию безопасного локального запуска AI Challenge App.\nУточнение: я новичок и работаю на macOS.\nОграничения: без Docker; без облачного развёртывания; до 5 действий в итоговой инструкции.\nТермин: ключ = DEEPSEEK_API_KEY.\nКакая версия Go нужна для запуска?",
			"Как подготовить локальный файл .env из шаблона?",
			"Куда записать ключ, чтобы он не оказался в браузере?",
			"Если ключ одновременно задан в окружении процесса и в .env, какой из них используется?",
			"В каком заголовке сервер отправляет ключ DeepSeek?",
			"Как запустить приложение и по какому адресу открыть чат?",
			"Какой маршрут принимает сообщения обычного чата?",
			"Ограничения: без Docker; без облачного развёртывания; до 3 действий в итоговой инструкции.\nУточнение: Ollama уже установлена и запущена.\nКакую локальную модель эмбеддингов использует поиск?",
			"В каком файле сохраняется история диалогов между перезапусками?",
			"Почему короткое окно сообщений не должно удалять историю обычного чата?",
			"Какие документы индексируются и почему файл .env не должен входить в базу?",
			"Что делать, если после изменения исходников поиск сообщает об устаревшем индексе?",
			"Как проверить, что к ответу приложены настоящий файл, раздел и точная цитата?",
			"Напомни мою цель, платформу, значение слова «ключ» и актуальный лимит действий. Отдельно объясни по документации, где хранится история.",
			"Собери итоговую инструкцию для моей задачи с учётом всех уточнений. Соблюди актуальный лимит действий и приложи источники.",
		}},
		{ID: "rag-memory", Title: "Архитектура RAG и памяти задачи", Turns: []string{
			"Цель: объяснить архитектуру чата с RAG и памятью задачи для учебного видео.\nУточнение: слушатель понимает HTTP, но впервые знакомится с RAG.\nОграничения: без смены модели; без внешних векторных сервисов; краткий итоговый ответ.\nТермин: память задачи = отдельно сохранённые цель, уточнения, ограничения и термины пользователя.\nКакие этапы проходит новый вопрос в чате с включённым RAG?",
			"Чем сохранённая история отличается от сообщений, отправляемых модели в текущем запросе?",
			"Какие две стратегии разбиения документов есть в индексаторе?",
			"Какой размер фиксированных фрагментов и их перекрытие?",
			"Какие метаданные позволяют показать источник найденного фрагмента?",
			"Сколько кандидатов берётся до фильтрации и сколько фрагментов остаётся по умолчанию?",
			"Для чего нужны порог similarity, rewrite и reranking?",
			"Уточнение: мне нужна демонстрация памяти с окном 4 сообщения.\nОграничения: без смены модели; без внешних векторных сервисов; подробный итоговый ответ.\nКак память задачи помогает после того, как раннее сообщение вышло за окно?",
			"Как выбранные профиль, стратегия контекста и память пользователя учитываются вместе с найденными документами?",
			"Как контекст задачи помогает искать документы для короткого уточняющего вопроса?",
			"Если подходящего фрагмента не найдено, что должен ответить ассистент и что покажет блок источников?",
			"Что произойдёт, если модель вернёт выдуманный chunk_id или цитату, отсутствующую в документе?",
			"Гарантирует ли дословное совпадение цитаты, что она по смыслу доказывает утверждение?",
			"Вспомни мою цель, аудиторию, согласованное значение «память задачи» и актуальные ограничения. Затем объясни по документации, как сохраняется история общего чата.",
			"Подготовь итоговое объяснение для моего видео: путь вопроса, поиск, память, проверка цитат и сохранение истории. Учти актуальную просьбу о подробном ответе и приложи источники.",
		}},
	}
}

func expectedConstraint(id string) string {
	if id == "safe-launch" {
		return "до 3 действий в итоговой инструкции"
	}
	return "подробный итоговый ответ"
}
func expectedTerm(id string) string {
	if id == "safe-launch" {
		return "ключ = DEEPSEEK_API_KEY"
	}
	return "память задачи = отдельно сохранённые цель, уточнения, ограничения и термины пользователя"
}
func hasNoteText(items []models.MemoryNote, expected string) bool {
	for _, item := range items {
		if strings.Contains(strings.ToLower(item.Text), strings.ToLower(expected)) {
			return true
		}
	}
	return false
}
func quotesExist(sources []models.RAGSource, chunks map[string]ragindex.Chunk) bool {
	for _, source := range sources {
		chunk, ok := chunks[source.ChunkID]
		if !ok || chunk.Source != source.Source || !strings.Contains(chunk.Text, source.Quote) {
			return false
		}
	}
	return true
}
func containsUserText(messages []models.ChatMessage, text string) bool {
	for _, m := range messages {
		if m.Role == "user" && m.Content == text {
			return true
		}
	}
	return false
}
func formatRequest(messages []models.ChatMessage) string {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString(m.Content)
		b.WriteByte('\n')
	}
	return b.String()
}
func replaceScenario(items []scenarioResult, value scenarioResult) []scenarioResult {
	for i := range items {
		if items[i].ID == value.ID {
			items[i] = value
			return items
		}
	}
	return append(items, value)
}
func writeReport(dir string, value report) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(dir, "results.json"), data, 0600); err != nil {
		return err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Live RAG task-memory evaluation\n\nGenerated: %s\n\nModel: `%s` · window: %d messages · index SHA-256: `%s`\n", value.GeneratedAt.Format(time.RFC3339), value.Model, value.Window, value.IndexHash)
	for _, s := range value.Scenarios {
		fmt.Fprintf(&b, "\n## %s\n\n", s.Title)
		for _, t := range s.Turns {
			fmt.Fprintf(&b, "### %d. %s\n\n%s\n\nSources: %d · quote validation: %t · goal retained: %t · memory beyond window: %t\n", t.Number, t.Question, t.Answer, len(t.Sources), t.Checks["quotesExistInIndexedChunks"], t.Checks["goalRetained"], t.Checks["memorySurvivesWindow"])
			for _, source := range t.Sources {
				fmt.Fprintf(&b, "- `%s` · %s · `%s` — %q\n", source.Source, source.Section, source.ChunkID, source.Quote)
			}
			if t.Error != "" {
				fmt.Fprintf(&b, "\nERROR: %s\n", t.Error)
			}
			b.WriteString("\n")
		}
	}
	return os.WriteFile(filepath.Join(dir, "results.md"), []byte(b.String()), 0600)
}
func readEnvFile(path, name string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "export "))
		key, value, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(key) == name {
			return strings.Trim(strings.TrimSpace(value), "\"'")
		}
	}
	return ""
}
func envOr(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}
func hash(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func fatalIf(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
