package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

func TestTaskMemoryReplacesCurrentConstraintAndTermWithProvenance(t *testing.T) {
	var memory models.TaskMemory
	memory = updateTaskMemory(memory, "Цель: безопасный запуск\nОграничения: до 5 действий; без Docker\nТермин: ключ = старое значение", 1)
	memory = updateTaskMemory(memory, "Ограничения: до 3 действий; без Docker\nТермин: ключ = DEEPSEEK_API_KEY", 8)
	if len(memory.Constraints) != 2 || memory.Constraints[0].Text != "до 3 действий" || memory.Constraints[1].Text != "без Docker" {
		t.Fatalf("current constraints = %#v", memory.Constraints)
	}
	if len(memory.Terms) != 1 || memory.Terms[0].Text != "ключ = DEEPSEEK_API_KEY" || memory.Terms[0].Turn != 8 {
		t.Fatalf("current terms = %#v", memory.Terms)
	}
	if prompt := taskMemoryPrompt(memory); !strings.Contains(prompt, "имеет приоритет") || !strings.Contains(prompt, "Обязательные task/state/global invariants имеют приоритет") || !strings.Contains(prompt, "до 3 действий") {
		t.Fatalf("precedence not explicit in prompt: %s", prompt)
	}
}

func TestRetrievalKeepsCurrentQuestionPrimaryAndAddsMemoryOnlyForShortAnaphora(t *testing.T) {
	memory := models.TaskMemory{
		Goal:  "объяснить архитектуру RAG с памятью",
		Terms: []models.MemoryNote{{Text: "память задачи = цель, уточнения и ограничения"}},
	}
	question := "Цель: объяснить архитектуру RAG с памятью\nОграничения: без внешних сервисов\nКакой размер фиксированных фрагментов и их перекрытие?"
	primary := retrievalQuestion(question)
	if primary != "Какой размер фиксированных фрагментов и их перекрытие?" {
		t.Fatalf("primary query was contaminated by memory fields: %q", primary)
	}
	if got := retrievalSupplementalQuery(question, memory); got != "" {
		t.Fatalf("specific question received supplemental memory query: %q", got)
	}
	followup := "А это где?"
	if got := retrievalSupplementalQuery(followup, memory); !strings.Contains(got, memory.Goal) || !strings.Contains(got, memory.Terms[0].Text) {
		t.Fatalf("short anaphoric follow-up omitted task memory: %q", got)
	}
	if got := retrievalSupplementalQuery("Их перекрытие сколько?", memory); got != "" {
		t.Fatalf("specific follow-up was treated as anaphoric memory query: %q", got)
	}
}

func TestDetailedSummaryRetrievalDropsPresentationScaffoldingButKeepsTopic(t *testing.T) {
	question := "Подготовь итоговое объяснение для моего видео: путь вопроса, поиск, память, проверка цитат и сохранение истории. Учти актуальную просьбу о подробном ответе и приложи источники."
	want := "путь вопроса поиск память проверка цитат и сохранение истории"
	if got := retrievalQuestion(question); got != want {
		t.Fatalf("detailed retrieval query=%q, want topical terms %q", got, want)
	}
}

type scenarioRAGEmbedder struct{}

func (scenarioRAGEmbedder) Embed(context.Context, []string) ([][]float64, error) {
	return [][]float64{{1, 0}}, nil
}

func newScenarioAgent(t *testing.T, store Store, includePlanner bool) (*Agent, *fakeCompleter, string, string) {
	t.Helper()
	root := t.TempDir()
	quote := "The server listens locally on port 8080."
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte("# Local server\n"+quote+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	docs, err := ragindex.Load(root, []string{"guide.md"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ragindex.ChunkDocuments(docs, "structure")
	if err != nil || len(chunks) != 1 {
		t.Fatalf("chunks=%d err=%v", len(chunks), err)
	}
	chunks[0].Vector = []float64{1, 0}
	indexData, err := json.Marshal(ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks})
	if err != nil {
		t.Fatal(err)
	}
	indexDir := filepath.Join(root, "index")
	if err := os.Mkdir(indexDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(indexDir, "index-structure.json"), indexData, 0600); err != nil {
		t.Fatal(err)
	}
	answer := fmt.Sprintf(`{"answer":"The local service uses this endpoint.","claims":[{"text":"The server listens locally on port 8080.","evidence":[{"chunkId":%q,"quote":%q}]}]}`, chunks[0].ChunkID, quote)
	if includePlanner {
		answer = strings.TrimSuffix(answer, "}") + `,"plan":{"phase":"planning","specification":"A task plan grounded by the repository.","currentStep":"Review the evidence","expectedAction":"Continue","openQuestions":[],"decisions":[],"nextSteps":["Continue"]}}`
	}
	client := &fakeCompleter{answer: answer}
	var a *Agent
	if store == nil {
		a = New(client)
	} else {
		var err error
		a, err = NewPersistent(client, store)
		if err != nil {
			t.Fatal(err)
		}
	}
	a.SetRetriever(&rag.Searcher{Root: root, IndexDir: indexDir, Model: "test-model", Embedder: scenarioRAGEmbedder{}})
	return a, client, chunks[0].ChunkID, quote
}

func TestTwoOfflineFifteenTurnRAGTaskMemoryScenariosWithWindowFour(t *testing.T) {
	scenarios := []struct {
		name, first, update, expectedConstraint, term string
	}{
		{"safe-launch", "Цель: подготовить безопасную инструкцию запуска на macOS\nОграничения: до 5 действий; без Docker\nТермин: ключ = старое значение\nГде слушает сервер?", "Ограничения: до 3 действий; без Docker\nТермин: ключ = DEEPSEEK_API_KEY\nКакой порт?", "до 3 действий", "ключ = DEEPSEEK_API_KEY"},
		{"rag-architecture", "Цель: объяснить RAG и память задачи\nОграничения: короткое объяснение; без внешних векторных сервисов\nТермин: память задачи = отдельная память\nГде слушает сервер?", "Ограничения: подробное объяснение; без внешних векторных сервисов\nТермин: память задачи = цель, уточнения, ограничения и термины\nКакой порт?", "подробное объяснение", "память задачи = цель, уточнения, ограничения и термины"},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			store := NewJSONStore(filepath.Join(t.TempDir(), "state.json"))
			a, client, _, quote := newScenarioAgent(t, store, false)
			for turn := 1; turn <= 15; turn++ {
				question := fmt.Sprintf("Как ответить на уточнение %d?", turn)
				if turn == 1 {
					question = scenario.first
				} else if turn == 8 {
					question = scenario.update
				}
				response, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), scenario.name, scenario.name, question, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
				if err != nil {
					t.Fatalf("turn %d: %v", turn, err)
				}
				if len(response.RAGSources) != 1 || response.RAGSources[0].Quote != quote || !strings.Contains(response.Answer, quote) {
					t.Fatalf("turn %d missing validated source: %#v", turn, response)
				}
				wantShortTerm := turn * 2
				if wantShortTerm > 4 {
					wantShortTerm = 4
				}
				if response.RecentMessages != 4 || len(response.Memory.ShortTerm) != wantShortTerm {
					t.Fatalf("turn %d window/short-term = %d/%d", turn, response.RecentMessages, len(response.Memory.ShortTerm))
				}
			}
			state := a.StateForUser(scenario.name, scenario.name)
			if len(state.Messages) != 30 || len(a.History(scenario.name)) != 30 {
				t.Fatalf("full archive lost messages: %d", len(state.Messages))
			}
			if len(state.TaskMemory.Constraints) != 2 || state.TaskMemory.Constraints[0].Text != scenario.expectedConstraint || state.TaskMemory.Constraints[0].Turn != 8 {
				t.Fatalf("updated constraint not retained: %#v", state.TaskMemory.Constraints)
			}
			if len(state.TaskMemory.Terms) != 1 || state.TaskMemory.Terms[0].Text != scenario.term || state.TaskMemory.Terms[0].Turn != 8 {
				t.Fatalf("term not replaced: %#v", state.TaskMemory.Terms)
			}
			lastRequest := client.requests[len(client.requests)-1]
			joined := fmt.Sprint(lastRequest)
			if strings.Contains(joined, scenario.first) || !strings.Contains(joined, state.TaskMemory.Goal) || !strings.Contains(joined, scenario.expectedConstraint) {
				t.Fatalf("memory did not survive window: %s", joined)
			}
			restarted, _, _, _ := newScenarioAgent(t, store, false)
			restored := restarted.StateForUser(scenario.name, scenario.name)
			if len(restored.Messages) != 30 || restored.TaskMemory.Goal != state.TaskMemory.Goal || restored.TaskMemory.Terms[0].Text != scenario.term {
				t.Fatalf("restart lost state: %#v", restored)
			}
			if err := restarted.ClearForUser(scenario.name, scenario.name); err != nil {
				t.Fatal(err)
			}
			cleared := restarted.StateForUser(scenario.name, scenario.name)
			if len(cleared.Messages) != 0 || hasTaskMemory(cleared.TaskMemory) || cleared.RecentMessages != 4 || cleared.Strategy != models.StrategySlidingWindow {
				t.Fatalf("clear did not reset task state: %#v", cleared)
			}
		})
	}
}

func TestRAGPreservesFactsProfilePlannerAndPausedLifecycle(t *testing.T) {
	a, client, _, _ := newScenarioAgent(t, nil, true)
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "set_strategy", Strategy: models.StrategyFacts}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "create_profile", Profile: models.UserProfile{Name: "reviewer", Perspective: "software reviewer", Style: "concise", Constraints: "до 5 действий"}}); err != nil {
		t.Fatal(err)
	}
	state := a.StateForUser("u", "s")
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "set_active_profile", ProfileID: state.Profiles[0].ID}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "enabled"}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "save_invariant", Invariant: models.Invariant{Scope: "global", Rule: "Never expose a secret key."}}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Grounded RAG task", Phases: []string{"planning", "execution", "validation"}}}); err != nil {
		t.Fatal(err)
	}
	question := "Цель: подготовить план\nОграничения: до 5 действий\nГде слушает сервер?"
	response, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", question, 4, models.StrategyFacts, models.DeepSeekFlashModel, true, models.RAGOptions{})
	if err != nil {
		t.Fatal(err)
	}
	request := fmt.Sprint(response.RequestMessages)
	for _, marker := range []string{"software reviewer", "Исторические sticky facts", "Актуальная память задачи", "Grounded RAG task", "Never expose a secret key.", "claims", "Факты из документов"} {
		if marker == "Факты из документов" {
			continue
		}
		if !strings.Contains(request, marker) {
			t.Errorf("request omitted compatibility state %q: %s", marker, request)
		}
	}
	if client.requests[len(client.requests)-1][0].Content == "" || response.Strategy != models.StrategyFacts || response.Profile.Name != "reviewer" || response.Task.Goal != "Grounded RAG task" {
		t.Fatalf("existing settings were bypassed: %#v", response)
	}
	if len(response.RAGSources) != 1 || response.Task.Specification == "" {
		t.Fatalf("planner envelope not grounded/applied: %#v", response)
	}
	client.answer = `{"answer":"Сначала согласуйте последовательность действий.","claims":[],"plan":{"phase":"planning","specification":"Предложение на основе цели пользователя","currentStep":"Уточнить объём","expectedAction":"Подтвердить","openQuestions":[],"decisions":[],"nextSteps":["Продолжить"]}}`
	proposal, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", "Подтверждаю план", 4, models.StrategyFacts, models.DeepSeekFlashModel, true, models.RAGOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !proposal.Task.PlanApproved || proposal.Task.Phase != "execution" || len(proposal.RAGSources) != 0 || !strings.Contains(proposal.Answer, "Предложение по плану задачи") || !strings.Contains(proposal.Answer, "Источники:") {
		t.Fatalf("claims=[] should preserve server approval and user-grounded plan: %#v", proposal)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "pause_task"}); err != nil {
		t.Fatal(err)
	}
	beforeCalls := len(client.requests)
	paused, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", "Какой порт у сервера?", 4, models.StrategyFacts, models.DeepSeekFlashModel, true, models.RAGOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != beforeCalls || paused.Task.Status != models.TaskPaused || paused.RAGTrace == nil || paused.RAGTrace.FilteredCount == 0 || len(paused.RAGSources) != 0 || !strings.Contains(paused.Answer, "из-за паузы") {
		t.Fatalf("paused request advanced or lacked honest RAG trace: %#v", paused)
	}
}

func TestTaskMemoryFollowsActiveBranchAndStrategySwitch(t *testing.T) {
	a := New(&fakeCompleter{answer: "Ответ"})
	if _, err := a.RespondWithStrategy(context.Background(), "s", "Цель: первая ветка", 4, models.StrategySlidingWindow); err != nil {
		t.Fatal(err)
	}
	branching, err := a.ApplyContextCommand("s", models.ContextCommand{Action: "set_strategy", Strategy: models.StrategyBranching})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommand("s", models.ContextCommand{Action: "checkpoint", Name: "first"}); err != nil {
		t.Fatal(err)
	}
	state := a.State("s")
	if len(state.Checkpoints) != 1 {
		t.Fatalf("checkpoints=%#v", state.Checkpoints)
	}
	created, err := a.ApplyContextCommand("s", models.ContextCommand{Action: "create_branch", CheckpointID: state.Checkpoints[0].ID, Name: "second"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.RespondWithStrategy(context.Background(), "s", "Цель: вторая ветка", 4, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommand("s", models.ContextCommand{Action: "switch_branch", BranchID: branching.ActiveBranchID}); err != nil {
		t.Fatal(err)
	}
	if got := a.State("s").TaskMemory.Goal; got != "первая ветка" {
		t.Fatalf("root branch memory=%q", got)
	}
	if _, err := a.ApplyContextCommand("s", models.ContextCommand{Action: "switch_branch", BranchID: created.ActiveBranchID}); err != nil {
		t.Fatal(err)
	}
	if got := a.State("s").TaskMemory.Goal; got != "вторая ветка" {
		t.Fatalf("second branch memory=%q", got)
	}
	if _, err := a.ApplyContextCommand("s", models.ContextCommand{Action: "set_strategy", Strategy: models.StrategySlidingWindow}); err != nil {
		t.Fatal(err)
	}
	if got := a.State("s").TaskMemory.Goal; got != "вторая ветка" {
		t.Fatalf("branch-to-window strategy transition lost memory: %q", got)
	}
}

func TestDeprecatedMiniChatStateMigratesOnlyWhenOrdinaryHistoryIsEmpty(t *testing.T) {
	legacyMessage := models.ChatMessage{Role: "user", Content: "Цель: сохранённая старая задача"}
	legacyTask := models.TaskMemory{Goal: "сохранённая старая задача", GoalQuote: legacyMessage.Content, GoalTurn: 1}
	t.Run("empty ordinary history migrates", func(t *testing.T) {
		store := NewJSONStore(filepath.Join(t.TempDir(), "state.json"))
		if err := store.Save(PersistentState{Sessions: map[string]ConversationState{"s": {UserID: "u", MiniChat: MiniChatState{Messages: []models.ChatMessage{legacyMessage}, Task: legacyTask}}}}); err != nil {
			t.Fatal(err)
		}
		a, err := NewPersistent(&fakeCompleter{}, store)
		if err != nil {
			t.Fatal(err)
		}
		if got := a.History("s"); len(got) != 1 || got[0].Content != legacyMessage.Content {
			t.Fatalf("legacy history not migrated: %#v", got)
		}
		if got := a.StateForUser("u", "s").TaskMemory.Goal; got != legacyTask.Goal {
			t.Fatalf("legacy task memory not migrated: %q", got)
		}
	})
	t.Run("existing ordinary history keeps legacy archive", func(t *testing.T) {
		store := NewJSONStore(filepath.Join(t.TempDir(), "state.json"))
		ordinary := models.ChatMessage{Role: "user", Content: "Текущая обычная история"}
		if err := store.Save(PersistentState{Sessions: map[string]ConversationState{"s": {UserID: "u", Messages: []models.ChatMessage{ordinary}, MiniChat: MiniChatState{Messages: []models.ChatMessage{legacyMessage}, Task: legacyTask}}}}); err != nil {
			t.Fatal(err)
		}
		a, err := NewPersistent(&fakeCompleter{answer: "Продолжение"}, store)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.RespondWithUserOptions(context.Background(), "u", "s", "Новый вопрос", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel); err != nil {
			t.Fatal(err)
		}
		persisted, err := store.Load()
		if err != nil {
			t.Fatal(err)
		}
		saved := persisted.Sessions["s"]
		if len(saved.MiniChat.Messages) != 1 || saved.MiniChat.Messages[0].Content != legacyMessage.Content || len(saved.Messages) != 3 || saved.Messages[0].Content != ordinary.Content {
			t.Fatalf("legacy archive or ordinary history was lost: %#v", saved)
		}
	})
}
