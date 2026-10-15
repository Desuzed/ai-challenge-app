package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

func helpFixtureAnswer(t *testing.T, a *Agent) string {
	t.Helper()
	r, err := a.retriever.SearchWithOptions(context.Background(), "history", rag.DefaultSearchOptions())
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf(`{"answer":"","claims":[{"text":"История хранится в JSON.","evidence":[{"chunkId":%q,"quote":"The private project uses a local JSON file for agent history."}]}]}`, r.Matches[0].Chunk.ChunkID)
}

func TestHelpOnlyHandlesExplicitCommands(t *testing.T) {
	for _, command := range []string{"/help", "/help off", "/help Где хранится история?", "/structure"} {
		t.Run(command, func(t *testing.T) {
			a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
			client.answer = helpFixtureAnswer(t, a)
			runtime := &helpRuntime{}
			a.SetToolRuntime(runtime)
			response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", command, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
			if err != nil {
				t.Fatal(err)
			}
			wantCalls := 1
			if command == "/help" || command == "/help off" {
				wantCalls = 0
				if !response.RAGModelCallSkipped || response.RAGEnabled || !strings.Contains(response.Answer, "/help <вопрос>") {
					t.Fatalf("usage=%#v", response)
				}
			} else if !response.RAGEnabled || response.RAGAbstained {
				t.Fatalf("help=%#v", response)
			}
			if len(client.requests) != wantCalls || len(runtime.calls) != 0 {
				t.Fatalf("model=%d want=%d tools=%d", len(client.requests), wantCalls, len(runtime.calls))
			}
			client.answer = "Обычный ответ"
			plain, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "Где хранится история?", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
			if err != nil || plain.RAGEnabled || plain.Answer != "Обычный ответ" || len(plain.ToolExecutions) != 0 || len(client.requests) != wantCalls+1 || len(runtime.calls) != 0 {
				t.Fatalf("plain=%#v err=%v model=%d tools=%d", plain, err, len(client.requests), len(runtime.calls))
			}
		})
	}
}

func TestLegacyHelpModeIsIgnoredAfterRestartWithoutLosingHistory(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, "Обычный ответ", models.ModelUsage{})
	path := filepath.Join(t.TempDir(), "history.json")
	a.store = NewJSONStore(path)
	if _, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/help", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false); err != nil {
		t.Fatal(err)
	}
	before := a.StateForUser("u", "s")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var legacy struct {
		Sessions map[string]map[string]any `json:"sessions"`
		Users    map[string]any            `json:"users"`
	}
	if err := json.Unmarshal(data, &legacy); err != nil {
		t.Fatal(err)
	}
	legacy.Sessions["s"]["helpEnabled"] = true
	legacy.Sessions["s"]["helpTopic"] = "История проекта"
	data, err = json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	restored, err := NewPersistent(client, a.store)
	if err != nil {
		t.Fatal(err)
	}
	restored.SetRetriever(a.retriever)
	runtime := &helpRuntime{}
	restored.SetToolRuntime(runtime)
	if got := restored.StateForUser("u", "s"); !reflect.DeepEqual(got.Messages, before.Messages) || got.Model != before.Model || got.Strategy != before.Strategy || !reflect.DeepEqual(got.Settings, before.Settings) {
		t.Fatal("restoring legacy help state changed history or settings")
	}
	plain, err := restored.RespondWithUserOptionsRAG(context.Background(), "u", "s", "Где хранится история?", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil || plain.RAGEnabled || plain.Answer != "Обычный ответ" || len(runtime.calls) != 0 || len(client.requests) != 1 {
		t.Fatalf("plain=%#v err=%v tools=%d model=%d", plain, err, len(runtime.calls), len(client.requests))
	}
	data, err = os.ReadFile(path)
	if err != nil || strings.Contains(string(data), `"helpEnabled"`) || strings.Contains(string(data), `"helpTopic"`) {
		t.Fatalf("legacy mode fields were persisted again: %v", err)
	}
}

func TestUnknownSlashCommandDoesNotCallModelOrSearch(t *testing.T) {
	client := &fakeCompleter{answer: "unexpected"}
	a := New(client)
	response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/not-a-command", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true)
	if err != nil || len(client.requests) != 0 || !response.RAGModelCallSkipped || !strings.Contains(response.Answer, "не реализована") {
		t.Fatalf("response=%#v err=%v", response, err)
	}
}

func TestDiffReadsBranchWithoutModelOrRAG(t *testing.T) {
	for _, test := range []struct{ result, want string }{
		{`{"branch":"feature/current","detached":false}`, "Текущая git-ветка: feature/current (MCP)."},
		{`{"branch":"","detached":true,"commit":"abc123"}`, "Текущая git-ветка: detached HEAD (abc123) (MCP)."},
	} {
		t.Run(test.want, func(t *testing.T) {
			client := &fakeCompleter{answer: "unexpected"}
			a := New(client)
			runtime := &helpRuntime{result: test.result}
			a.SetToolRuntime(runtime)
			before := a.StateForUser("u", "s")
			r, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/diff", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true)
			if err != nil || r.Answer != test.want || !r.RAGModelCallSkipped || r.RAGEnabled || len(client.requests) != 0 || len(r.RequestMessages) != 0 {
				t.Fatalf("response=%#v err=%v modelCalls=%d", r, err, len(client.requests))
			}
			if runtime.definitionsRequested || !reflect.DeepEqual(runtime.calls, []string{"project_git_branch"}) || len(r.ToolExecutions) != 1 || len(r.RAGSources) != 1 || r.RAGSources[0].Source != "MCP:project_git_branch" {
				t.Fatalf("runtime=%#v response=%#v", runtime, r)
			}
			if !reflect.DeepEqual(r.Task, before.Task) || !reflect.DeepEqual(r.TaskMemory, before.TaskMemory) {
				t.Fatal("diff changed planner or task memory")
			}
		})
	}
}

func TestDiffFailureDoesNotInventBranchOrInvokeModel(t *testing.T) {
	client := &fakeCompleter{answer: "unexpected"}
	a := New(client)
	a.SetToolRuntime(&helpRuntime{failure: errors.New("git unavailable")})
	r, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/diff", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil || !strings.Contains(r.Answer, "Не удалось получить") || strings.Contains(r.Answer, "Текущая git-ветка:") || len(r.ToolExecutions) != 0 || len(r.RAGSources) != 0 || len(client.requests) != 0 {
		t.Fatalf("response=%#v err=%v", r, err)
	}
}

func TestDiffRejectsArgumentsAndRespectsTaskPause(t *testing.T) {
	client := &fakeCompleter{answer: "unexpected"}
	a := New(client)
	runtime := &helpRuntime{}
	a.SetToolRuntime(runtime)
	r, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/diff other", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil || !strings.Contains(r.Answer, "без аргументов") || len(runtime.calls) != 0 {
		t.Fatalf("arguments: response=%#v err=%v", r, err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Check git context", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "pause_task"}); err != nil {
		t.Fatal(err)
	}
	r, err = a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/diff", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil || !strings.Contains(r.Answer, "на паузе") || len(runtime.calls) != 0 || len(client.requests) != 0 {
		t.Fatalf("paused: response=%#v err=%v", r, err)
	}
}

func TestHelpWorksWithoutMCP(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	client.answer = helpFixtureAnswer(t, a)
	for _, command := range []string{"/help Где хранится история?", "/structure"} {
		r, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", command, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
		if err != nil || r.RAGAbstained || len(r.ToolExecutions) != 0 || strings.Contains(r.Answer, "Текущая git-ветка:") {
			t.Fatalf("response=%#v err=%v", r, err)
		}
	}
}

func TestReindexActuallyRefreshesStaleDocumentsWithoutLLM(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	a.SetIndexer(&ragindex.Rebuilder{Root: a.retriever.Root, Output: a.retriever.IndexDir, Model: "test-model", Sources: []string{"guide.md"}, Embedder: ragTestEmbedder{}})
	if err := os.WriteFile(filepath.Join(a.retriever.Root, "guide.md"), []byte("# Storage\nUpdated documentation stores history in JSON."), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := a.retriever.IndexInfo(); err != rag.ErrIndexStale {
		t.Fatalf("before=%v", err)
	}
	response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/reindex", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true)
	if err != nil || response.Reindex == nil || response.Reindex.Files != 1 || len(client.requests) != 0 {
		t.Fatalf("response=%#v err=%v calls=%d", response, err, len(client.requests))
	}
	if _, err := a.retriever.IndexInfo(); err != nil {
		t.Fatalf("after=%v", err)
	}
	data, err := os.ReadFile(filepath.Join(a.retriever.IndexDir, "index-structure.json"))
	if err != nil || !strings.Contains(string(data), "Updated documentation") {
		t.Fatalf("updated index err=%v", err)
	}
}

type failingIndexer struct{}

func (failingIndexer) Rebuild(context.Context) (ragindex.Report, error) {
	return ragindex.Report{}, errors.New("embedding service unavailable")
}

func TestFailedReindexDoesNotClaimSuccess(t *testing.T) {
	a := New(&fakeCompleter{})
	a.SetIndexer(failingIndexer{})
	r, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/reindex", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil || r.Reindex != nil || !strings.Contains(r.Answer, "не выполнена") || strings.Contains(r.Answer, "Индексация завершена") {
		t.Fatalf("response=%#v err=%v", r, err)
	}
}

type helpRepairClient struct {
	answers []models.ModelCompletion
	calls   int
}

func (c *helpRepairClient) CompleteMessages(context.Context, []models.ChatMessage, models.GenerationSettings) (models.ModelCompletion, error) {
	i := c.calls
	c.calls++
	if i >= len(c.answers) {
		return models.ModelCompletion{}, errors.New("unexpected third completion")
	}
	return c.answers[i], nil
}

func TestHelpRetriesBadEvidenceAndCountsBothRequests(t *testing.T) {
	a, _ := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	client := &helpRepairClient{answers: []models.ModelCompletion{
		{Answer: `{"answer":"","claims":[{"text":"invented","evidence":[{"chunkId":"missing","quote":"fabricated"}]}]}`, Usage: models.ModelUsage{InputTokens: 10, OutputTokens: 3, TotalTokens: 13}},
		{Answer: helpFixtureAnswer(t, a), FinishReason: "stop", Usage: models.ModelUsage{InputTokens: 11, OutputTokens: 4, TotalTokens: 15}},
	}}
	a.client = client
	a.SetToolRuntime(&helpRuntime{})
	response, err := a.RespondWithUserOptionsRAGConfiguredSettings(context.Background(), "u", "s", "/help Где хранится история?", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false, models.RAGOptions{}, models.GenerationSettings{MaxTokens: 2048})
	if err != nil || response.RAGAbstained || client.calls != 2 || response.RAGTrace.CompletionAttempts != 2 || response.RAGTrace.CompletionRetryReason != "invalid_evidence" || response.RAGTrace.OutputTokenBudget != 2048 || response.Tokens.RequestTokens != 21 || response.Tokens.ResponseTokens != 7 {
		t.Fatalf("response=%#v err=%v calls=%d", response, err, client.calls)
	}
}
