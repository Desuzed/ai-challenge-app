package agent

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/rag"
)

type helpRuntime struct {
	calls                []string
	definitionsRequested bool
	result               string
	failure              error
}

func (r *helpRuntime) ToolsForModel(context.Context) ([]models.ToolDefinition, error) {
	r.definitionsRequested = true
	return nil, fmt.Errorf("help must not expose model tools")
}
func (r *helpRuntime) CallForModel(_ context.Context, name string, _ map[string]any) (string, bool, error) {
	r.calls = append(r.calls, name)
	if name != "project_git_branch" {
		return "", true, fmt.Errorf("unexpected write or other tool: %s", name)
	}
	if r.failure != nil {
		return "", false, r.failure
	}
	if r.result != "" {
		return r.result, false, nil
	}
	return `{"branch":"feature/current","detached":false}`, false, nil
}

func TestHelpForcesRAGWithoutMCPAndPreservesTask(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	runtime := &helpRuntime{}
	a.SetToolRuntime(runtime)
	before, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "enabled"})
	if err != nil {
		t.Fatal(err)
	}
	matches, err := a.retriever.SearchWithOptions(context.Background(), "history", rag.DefaultSearchOptions())
	if err != nil {
		t.Fatal(err)
	}
	chunk := matches.Matches[0].Chunk
	client.answer = fmt.Sprintf(`{"claims":[{"text":"История проекта хранится в JSON.","evidence":[{"chunkId":%q,"quote":%q}]}]}`, chunk.ChunkID, "The private project uses a local JSON file for agent history.")
	for _, message := range []string{"/structure", "/help Где хранится история проекта?"} {
		response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", message, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
		if err != nil {
			t.Fatal(err)
		}
		if !response.RAGEnabled || response.RAGAbstained || len(response.RAGSources) != 1 || len(response.ToolExecutions) != 0 || strings.Contains(response.Answer, "Текущая git-ветка:") {
			t.Fatalf("response=%#v", response)
		}
		if response.RAGTrace.OriginalQuery == message || strings.HasPrefix(response.RAGTrace.OriginalQuery, "/help") {
			t.Fatalf("command leaked into retrieval: %s", response.RAGTrace.OriginalQuery)
		}
		if !reflect.DeepEqual(response.Task, before.Task) || !reflect.DeepEqual(response.TaskMemory, before.TaskMemory) {
			t.Fatal("help changed the planner or task memory")
		}
		if strings.Contains(fmt.Sprint(response.RequestMessages), "feature/current") {
			t.Fatal("help sent live git context to the model")
		}
	}
	if runtime.definitionsRequested || len(runtime.calls) != 0 {
		t.Fatalf("tools=%#v", runtime)
	}
}

func TestHelpRejectsStaleIndexBeforeMCPOrModel(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	runtime := &helpRuntime{}
	a.SetToolRuntime(runtime)
	if err := os.WriteFile(filepath.Join(a.retriever.Root, "guide.md"), []byte("Changed documentation"), 0600); err != nil {
		t.Fatal(err)
	}
	_, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/help Где хранится история?", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != rag.ErrIndexStale || len(runtime.calls) != 0 || len(client.requests) != 0 {
		t.Fatalf("err=%v tools=%#v requests=%d", err, runtime.calls, len(client.requests))
	}
}

func TestHelpOverviewIntentDoesNotSwallowSpecificQuestions(t *testing.T) {
	for _, question := range []string{"расскажи о проекте", " Расскажи   о проекте! ", "Что это за проект?", "обзор проекта"} {
		if !projectOverviewQuestion(question) {
			t.Fatalf("overview not recognised: %q", question)
		}
	}
	for _, question := range []string{"Где хранится история проекта?", "Расскажи о проекте Туман-773 и цвете его обложки", "Какие MCP-серверы подключены?"} {
		if projectOverviewQuestion(question) {
			t.Fatalf("specific question replaced with overview: %q", question)
		}
	}
}

func TestHelpUsesCompactContractAndPreservesSmallTokenSetting(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	client.answer = helpFixtureAnswer(t, a)
	a.SetToolRuntime(&helpRuntime{})
	r, err := a.RespondWithUserOptionsRAGConfiguredSettings(context.Background(), "u", "s", "/help расскажи о проекте", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false, models.RAGOptions{}, models.GenerationSettings{MaxTokens: 512})
	if err != nil || r.RAGAbstained || len(client.requests) != 1 || r.Settings.MaxTokens != 512 || r.RAGTrace.OutputTokenBudget != 512 {
		t.Fatalf("response=%#v err=%v", r, err)
	}
	if !strings.Contains(client.requests[0][0].Content, "ровно один короткий claim") || !strings.Contains(fmt.Sprint(client.requests[0]), "расскажи о проекте") {
		t.Fatal("small-budget help did not retain question and compact instructions")
	}
}

func TestHelpRefusesInventedCitationWithoutGitPrefix(t *testing.T) {
	a, _ := makeRAGFixture(t, []float64{1, 0}, `{"claims":[{"text":"Invented fact","evidence":[{"chunkId":"missing","quote":"Invented quote"}]}]}`, models.ModelUsage{})
	a.SetToolRuntime(&helpRuntime{})
	response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "/help Где хранится история?", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil {
		t.Fatal(err)
	}
	if !response.RAGAbstained || response.RAGReason != "invalid_evidence" || strings.Contains(response.Answer, "Invented fact") || len(response.RAGSources) != 0 || strings.Contains(response.Answer, "Текущая git-ветка:") {
		t.Fatalf("response=%#v", response)
	}
}
