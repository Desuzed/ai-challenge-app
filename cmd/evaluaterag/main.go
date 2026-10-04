// Command evaluaterag runs the ten reviewed control questions through the live
// Ollama retrieval and DeepSeek answer path and writes a reviewable report.
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
	"sort"
	"strings"
	"time"

	"ai-challenge-app/internal/agent"
	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

type testCase struct {
	Question        string     `json:"question"`
	ExpectedSources []string   `json:"expectedSources"`
	ExpectedTerms   []string   `json:"expectedTerms"`
	ExpectedAny     [][]string `json:"expectedAny,omitempty"`
}
type check struct {
	HasExpectedSource    bool   `json:"hasExpectedSource"`
	QuotesAreExact       bool   `json:"quotesAreExact"`
	SemanticProxy        bool   `json:"semanticProxy"`
	ManualSemanticReview string `json:"manualSemanticReview"`
}
type row struct {
	Number   int                `json:"number"`
	Question string             `json:"question"`
	Answer   string             `json:"answer"`
	Sources  []models.RAGSource `json:"sources"`
	Checks   check              `json:"checks"`
	Error    string             `json:"error,omitempty"`
}
type report struct {
	GeneratedAt time.Time `json:"generatedAt"`
	Model       string    `json:"model"`
	Fingerprint string    `json:"fingerprint"`
	Questions   []row     `json:"questions"`
}

var cases = []testCase{
	{"Где приложение хранит историю диалога агента между перезапусками?", []string{"README.md"}, []string{".local/agent-history.json"}, nil},
	{"Что имеет приоритет для DEEPSEEK_API_KEY: окружение процесса или .env?", []string{"main.go"}, []string{"окруж", ".env"}, nil},
	{"Какой маршрут браузер использует для сообщения основному агенту?", []string{"README.md", "internal/handlers/handlers.go"}, []string{"/api/agent/chat", "post"}, nil},
	{"Какие два способа разбиения документов реализованы в индексаторе?", []string{"README.md"}, []string{"1200", "150", "заголов"}, nil},
	{"Где сохраняются два индекса и отчёт?", []string{"README.md"}, []string{".local/rag/", "index-fixed.json", "index-structure.json", "report.json"}, nil},
	{"Какую информацию об источнике, файле, разделе и идентификаторе хранит каждый чанк локального индекса?", []string{"README.md"}, []string{"source", "title", "section", "chunk_id"}, nil},
	{"Какие четыре MCP-сервера собираются в единый каталог при старте приложения?", []string{"main.go"}, []string{"яндекс", "github", "погод", "новост"}, nil},
	{"Как соединены MCP-сервер и клиент внутри приложения?", []string{"README.md"}, []string{"in-memory", "transport"}, nil},
	{"Какой период сбора погоды используется при отсутствии настройки интервала?", []string{"main.go"}, nil, [][]string{{"одна минут", "1 минут"}}},
	{"Куда уходит API-ключ DeepSeek и виден ли он браузеру?", []string{"README.md"}, []string{"authorization", "браузер", "сервер"}, nil},
}

func main() {
	root := flag.String("root", ".", "project directory")
	indexDir := flag.String("index", ".local/rag", "RAG index directory")
	outDir := flag.String("output", ".local/rag-eval", "report directory")
	ollamaURL := flag.String("ollama-url", envOr("OLLAMA_URL", "http://127.0.0.1:11434"), "Ollama URL")
	resume := flag.Bool("resume", false, "reuse completed answers from output/results.json")
	flag.Parse()
	rootAbs, err := filepath.Abs(*root)
	fatalIf(err)
	key := os.Getenv("DEEPSEEK_API_KEY")
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
	client := deepseek.NewClient(key, 120*time.Second)
	a := agent.New(client)
	a.SetRetriever(&rag.Searcher{Root: rootAbs, IndexDir: resolve(*indexDir), Model: "embeddinggemma", Embedder: ragindex.Ollama{URL: *ollamaURL, Model: "embeddinggemma"}})
	fingerprint := evaluationFingerprint(filepath.Join(resolve(*indexDir), "index-structure.json"))
	r := report{GeneratedAt: time.Now().UTC(), Model: models.DeepSeekFlashModel, Fingerprint: fingerprint, Questions: make([]row, 0, len(cases))}
	out := resolve(*outDir)
	fatalIf(os.MkdirAll(out, 0700))
	verifiedChunks := loadChunks(filepath.Join(resolve(*indexDir), "index-structure.json"))
	if *resume {
		if data, err := os.ReadFile(filepath.Join(out, "results.json")); err == nil {
			var previous report
			if json.Unmarshal(data, &previous) == nil && previous.Fingerprint == fingerprint && previous.Model == r.Model && validSavedQuestions(previous.Questions) {
				r = previous
			} else {
				fmt.Fprintln(os.Stderr, "Previous results do not match the current index, model, or question set; starting a fresh evaluation.")
			}
		}
	}
	completed := map[int]row{}
	for _, item := range r.Questions {
		completed[item.Number] = item
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	for i, tc := range cases {
		if previous, ok := completed[i+1]; ok && previous.Error == "" {
			continue
		}
		response, err := a.RespondWithUserOptionsRAGConfigured(ctx, "rag-eval", fmt.Sprintf("q-%02d", i+1), tc.Question, 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
		item := row{Number: i + 1, Question: tc.Question}
		item.Answer, item.Sources = response.Answer, response.RAGSources
		if err != nil {
			item.Error = err.Error()
		}
		item.Checks = check{QuotesAreExact: true, ManualSemanticReview: "pending"}
		for _, s := range response.RAGSources {
			if contains(tc.ExpectedSources, s.Source) {
				item.Checks.HasExpectedSource = true
			}
			chunk, ok := verifiedChunks[s.ChunkID]
			if !ok || chunk.Source != s.Source || chunk.Section != s.Section || s.Quote == "" || !strings.Contains(chunk.Text, s.Quote) {
				item.Checks.QuotesAreExact = false
			}
		}
		if len(response.RAGSources) == 0 {
			item.Checks.QuotesAreExact = false
		}
		answerOnly, _, _ := strings.Cut(response.Answer, "\n\nИсточники и цитаты\n")
		item.Checks.SemanticProxy = semanticProxy(answerOnly, tc)
		// This check only measures expected keyword coverage; it cannot prove entailment.
		r.Questions = upsertRow(r.Questions, item)
		sort.Slice(r.Questions, func(i, j int) bool { return r.Questions[i].Number < r.Questions[j].Number })
		// Checkpoint after every question so a transient API failure does not
		// discard completed live requests.
		persist(out, r)
		fmt.Printf("%02d/10 %s\n", i+1, map[bool]string{true: "ok", false: "review"}[item.Error == "" && item.Checks.HasExpectedSource && item.Checks.QuotesAreExact])
	}
	persist(out, r)
	fmt.Printf("Saved %s and %s\n", filepath.Join(out, "results.json"), filepath.Join(out, "results.md"))
}

func semanticProxy(answer string, tc testCase) bool {
	answer, _, _ = strings.Cut(answer, "\n\nИсточники и цитаты\n")
	lower := strings.ToLower(answer)
	for _, term := range tc.ExpectedTerms {
		if !strings.Contains(lower, strings.ToLower(term)) {
			return false
		}
	}
	for _, alternatives := range tc.ExpectedAny {
		found := false
		for _, term := range alternatives {
			if strings.Contains(lower, strings.ToLower(term)) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func evaluationFingerprint(indexPath string) string {
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return "missing-index"
	}
	questions, _ := json.Marshal(cases)
	hash := sha256.New()
	hash.Write(data)
	hash.Write([]byte(models.DeepSeekFlashModel))
	hash.Write(questions)
	return hex.EncodeToString(hash.Sum(nil))
}
func validSavedQuestions(rows []row) bool {
	if len(rows) > len(cases) {
		return false
	}
	seen := make(map[int]bool, len(rows))
	for _, item := range rows {
		if item.Number < 1 || item.Number > len(cases) || seen[item.Number] || item.Question != cases[item.Number-1].Question {
			return false
		}
		seen[item.Number] = true
	}
	return true
}
func upsertRow(rows []row, item row) []row {
	for i := range rows {
		if rows[i].Number == item.Number {
			rows[i] = item
			return rows
		}
	}
	return append(rows, item)
}

func loadChunks(path string) map[string]ragindex.Chunk {
	data, err := os.ReadFile(path)
	if err != nil {
		return map[string]ragindex.Chunk{}
	}
	var index ragindex.Index
	if json.Unmarshal(data, &index) != nil {
		return map[string]ragindex.Chunk{}
	}
	chunks := map[string]ragindex.Chunk{}
	for _, chunk := range index.Chunks {
		chunks[chunk.ChunkID] = chunk
	}
	return chunks
}
func persist(out string, r report) {
	data, err := json.MarshalIndent(r, "", "  ")
	fatalIf(err)
	fatalIf(os.WriteFile(filepath.Join(out, "results.json"), data, 0600))
	fatalIf(os.WriteFile(filepath.Join(out, "results.md"), []byte(markdown(r)), 0600))
}

func contains(list []string, value string) bool {
	for _, item := range list {
		if value == item {
			return true
		}
	}
	return false
}
func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
func readEnvFile(path, key string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok && strings.TrimSpace(k) == key {
			return strings.Trim(strings.TrimSpace(v), "\"'")
		}
	}
	return ""
}
func fatalIf(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func markdown(r report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# RAG evaluation\n\nGenerated: %s · Model: %s\n\n", r.GeneratedAt.Format(time.RFC3339), r.Model)
	b.WriteString("| # | Question | Expected source | Exact quote | Semantic proxy | Manual semantic review |\n|---:|---|---|---|---|---|\n")
	for i, item := range r.Questions {
		fmt.Fprintf(&b, "| %d | %s | %t | %t | %t | %s |\n", i+1, item.Question, item.Checks.HasExpectedSource, item.Checks.QuotesAreExact, item.Checks.SemanticProxy, item.Checks.ManualSemanticReview)
	}
	b.WriteString("\n## Answers and evidence\n")
	for _, item := range r.Questions {
		fmt.Fprintf(&b, "\n### %d. %s\n\n%s\n", item.Number, item.Question, item.Answer)
		for _, source := range item.Sources {
			fmt.Fprintf(&b, "\n- `%s` · `%s` · `%s`: > %s\n", source.Source, source.Section, source.ChunkID, strings.ReplaceAll(source.Quote, "\n", " "))
		}
		if item.Error != "" {
			fmt.Fprintf(&b, "\nError: %s\n", item.Error)
		}
	}
	b.WriteString("\nSemantic proxy checks expected keyword coverage only. They do not prove that a claim follows from its quote; review the answers manually.\n")
	return b.String()
}
