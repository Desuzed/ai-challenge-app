package agent

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"ai-challenge-app/internal/models"
)

type fakeCompleter struct {
	requests [][]models.ChatMessage
	models   []string
	answer   string
	err      error
	usage    models.ModelUsage
}

func (f *fakeCompleter) CompleteMessagesModel(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	f.models = append(f.models, model)
	return f.CompleteMessages(ctx, messages, settings)
}

func TestPersistentAgentRestoresHistoryAfterRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "state", "agent-history.json"))
	firstClient := &fakeCompleter{answer: "Запомнил"}
	firstAgent, err := NewPersistent(firstClient, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := firstAgent.Respond(context.Background(), "session-a", "Мой любимый цвет — зелёный"); err != nil {
		t.Fatal(err)
	}

	secondClient := &fakeCompleter{answer: "Продолжаю"}
	secondAgent, err := NewPersistent(secondClient, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := secondAgent.Respond(context.Background(), "session-a", "Какой мой любимый цвет?"); err != nil {
		t.Fatal(err)
	}
	if got, want := secondClient.requests[0][1:], []models.ChatMessage{
		{Role: "user", Content: "Мой любимый цвет — зелёный"},
		{Role: "assistant", Content: "Запомнил"},
		{Role: "user", Content: "Какой мой любимый цвет?"},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("restored request = %#v, want %#v", got, want)
	}
}

func TestClearRemovesPersistentSession(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	first, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Respond(context.Background(), "session-a", "Сообщение"); err != nil {
		t.Fatal(err)
	}
	if err := first.Clear("session-a"); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if got := second.History("session-a"); len(got) != 0 {
		t.Fatalf("history after clear = %#v, want empty", got)
	}
}

func TestClearKeepsLongTermMemoryButRemovesDialogueAndWorkingMemory(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	agent, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Текущая задача", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryWorking, Value: "Текущая задача"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryLongTerm, Category: "profile", Value: "Пишет по-русски"}); err != nil {
		t.Fatal(err)
	}
	if err := agent.Clear("session"); err != nil {
		t.Fatal(err)
	}
	state := agent.State("session")
	if len(state.Messages) != 0 || len(state.Memory.ShortTerm) != 0 || len(state.Memory.Working) != 0 {
		t.Fatalf("cleared task state = %#v", state)
	}
	if got := state.Memory.LongTerm; len(got) != 1 || got[0].Value != "Пишет по-русски" {
		t.Fatalf("long-term memory after clear = %#v", got)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "clear_memory_layer", Layer: models.MemoryLongTerm}); err != nil {
		t.Fatal(err)
	}
	if got := agent.State("session").Memory.LongTerm; len(got) != 0 {
		t.Fatalf("long-term memory after separate clear = %#v", got)
	}
}

func (f *fakeCompleter) CompleteMessages(_ context.Context, messages []models.ChatMessage, _ models.GenerationSettings) (models.ModelCompletion, error) {
	f.requests = append(f.requests, append([]models.ChatMessage(nil), messages...))
	if f.err != nil {
		return models.ModelCompletion{}, f.err
	}
	return models.ModelCompletion{Answer: f.answer, Usage: f.usage}, nil
}

func TestRespondReportsExactProviderUsageAndCumulativeCost(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ", usage: models.ModelUsage{InputTokens: 120, OutputTokens: 30, CacheMissTokens: 120}}
	agent := New(client)
	result, err := agent.Respond(context.Background(), "session", "Короткий вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if result.Tokens.RequestTokens != 120 || result.Tokens.ResponseTokens != 30 || result.Tokens.CumulativeInputTokens != 120 || result.Tokens.CumulativeOutputTokens != 30 {
		t.Fatalf("tokens = %#v", result.Tokens)
	}
	if result.Tokens.EstimatedRequestTokens == 0 || result.Tokens.CurrentMessageTokens == 0 || result.Tokens.CumulativeCostUSD <= 0 {
		t.Fatalf("incomplete token report = %#v", result.Tokens)
	}
	if result.Tokens.CacheMissTokens != 120 || len(result.RequestMessages) != 2 || result.RequestMessages[1].Content != "Короткий вопрос" {
		t.Fatalf("request details = %#v, tokens = %#v", result.RequestMessages, result.Tokens)
	}
}

func TestRespondRejectsDialogueBeyondContextBeforeProviderCall(t *testing.T) {
	client := &fakeCompleter{answer: "Не должен вызываться"}
	agent := New(client)
	conversation := agent.conversation("session")
	conversation.strategy = models.StrategyBranching
	conversation.ensureRootBranchLocked()
	conversation.branches["root"].messages = make([]models.ChatMessage, 100)
	for i := range conversation.branches["root"].messages {
		conversation.branches["root"].messages[i] = models.ChatMessage{Role: "user", Content: strings.Repeat("я", maxMessageCharacters)}
	}
	if _, err := agent.Respond(context.Background(), "session", "Ещё один вопрос"); !errors.Is(err, ErrContextLimit) {
		t.Fatalf("Respond error = %v, want ErrContextLimit", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("provider requests = %d, want 0", len(client.requests))
	}
	if got := len(agent.History("session")); got != 100 {
		t.Fatalf("history length = %d, want 100; rejected message must not persist", got)
	}
}

func TestTokenReportDoesNotCountSystemInstructionAsDialogueHistory(t *testing.T) {
	agent := New(&fakeCompleter{})
	if err := agent.Clear("session"); err != nil {
		t.Fatal(err)
	}
	report := agent.TokenReport("session")
	if report.HistoryTokens != 0 || report.CurrentMessageTokens != 0 {
		t.Fatalf("cleared dialogue tokens = %#v, want zero", report)
	}
	if report.EstimatedRequestTokens == 0 {
		t.Fatal("system instruction must remain in full request estimate")
	}
}

func TestRespondBuildsHistoryPerSession(t *testing.T) {
	client := &fakeCompleter{answer: "Первый ответ"}
	agent := New(client)
	first, err := agent.Respond(context.Background(), "session-a", "Первый вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := first.Messages, []models.ChatMessage{{Role: "user", Content: "Первый вопрос"}, {Role: "assistant", Content: "Первый ответ"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("first history = %#v, want %#v", got, want)
	}

	client.answer = "Второй ответ"
	_, err = agent.Respond(context.Background(), "session-a", "Второй вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := client.requests[1][1:], []models.ChatMessage{
		{Role: "user", Content: "Первый вопрос"}, {Role: "assistant", Content: "Первый ответ"}, {Role: "user", Content: "Второй вопрос"},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second LLM request = %#v, want %#v", got, want)
	}

	client.answer = "Другой диалог"
	other, err := agent.Respond(context.Background(), "session-b", "Другой вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := other.Messages, []models.ChatMessage{{Role: "user", Content: "Другой вопрос"}, {Role: "assistant", Content: "Другой диалог"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second session history = %#v, want %#v", got, want)
	}
}

func TestRespondDoesNotKeepFailedMessage(t *testing.T) {
	client := &fakeCompleter{err: errors.New("provider unavailable")}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Вопрос"); err == nil {
		t.Fatal("Respond succeeded, want provider error")
	}
	if got := agent.History("session"); len(got) != 0 {
		t.Fatalf("history after failure = %#v, want empty", got)
	}
}

func TestSlidingWindowKeepsOnlyRecentNWithoutSummaryCall(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Первый факт", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Второй факт", 2); err != nil {
		t.Fatal(err)
	}
	result, err := agent.Respond(context.Background(), "session", "Третий вопрос", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 2 || result.Messages[0].Content != "Третий вопрос" {
		t.Fatalf("raw history = %#v, want only last two messages", result.Messages)
	}
	if len(client.requests) != 3 {
		t.Fatalf("calls = %d, want one provider request per user message", len(client.requests))
	}
	request := client.requests[2]
	if len(request) != 3 || request[1].Role != "assistant" || request[1].Content != "Ответ" || request[2].Content != "Третий вопрос" {
		t.Fatalf("window request = %#v", request)
	}
}

func TestFactsSurviveWindowAndAreSentAsSeparateSystemBlock(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Цель: сделать приложение привычек", 2, models.StrategyFacts); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Ограничение: только iOS", 2, models.StrategyFacts); err != nil {
		t.Fatal(err)
	}
	result, err := agent.RespondWithStrategy(context.Background(), "session", "Что мы делаем?", 2, models.StrategyFacts)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 2 || len(result.Facts) == 0 {
		t.Fatalf("state = %#v", result)
	}
	if !strings.Contains(client.requests[2][1].Content, "Постоянные факты") || !strings.Contains(client.requests[2][1].Content, "сделать приложение") {
		t.Fatalf("facts prompt = %#v", client.requests[2])
	}
}

func TestMemoryLayersAreSeparateAndOnlyExplicitCommandsPersistWorkingAndLongTerm(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Обсуждаем экран заказа", 2); err != nil {
		t.Fatal(err)
	}
	if state := agent.State("session"); len(state.Memory.Working) != 0 || len(state.Memory.LongTerm) != 0 {
		t.Fatalf("ordinary dialogue must not auto-save explicit layers: %#v", state.Memory)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryWorking, Key: "task", Value: "Экран отслеживания заказа"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "profile", Key: "language", Value: "Русский"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "decisions", Key: "map", Value: "Не показывать координаты курьера"}); err != nil {
		t.Fatal(err)
	}

	state := agent.State("session")
	if got := len(state.Memory.ShortTerm); got != 2 {
		t.Fatalf("short-term entries = %d, want dialogue only", got)
	}
	if got := state.Memory.Working; len(got) != 1 || got[0].Key != "task" {
		t.Fatalf("working memory = %#v", got)
	}
	if got := state.Memory.LongTerm; len(got) != 2 || got[0].Category != "decisions" || got[1].Category != "profile" {
		t.Fatalf("long-term memory = %#v", got)
	}

	if _, err := agent.Respond(context.Background(), "session", "Что учесть в ответе?", 2); err != nil {
		t.Fatal(err)
	}
	request := client.requests[len(client.requests)-1]
	if !strings.Contains(request[1].Content, "Долговременная память") || !strings.Contains(request[1].Content, "profile.language: Русский") {
		t.Fatalf("long-term block = %#v", request)
	}
	if !strings.Contains(request[2].Content, "Рабочая память") || !strings.Contains(request[2].Content, "task: Экран") {
		t.Fatalf("working block = %#v", request)
	}
	if len(request) != 5 || request[3].Role != "assistant" || request[4].Content != "Что учесть в ответе?" {
		t.Fatalf("short-term window in request = %#v", request)
	}
}

func TestMemoryLayersPersistAndShortTermCannotBeWrittenDirectly(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "memory.json"))
	first, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryWorking, Key: "deadline", Value: "пятница"}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "api", Value: "Используем существующее API"}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryShortTerm, Key: "x", Value: "y"}); err == nil {
		t.Fatal("short-term memory write succeeded")
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.State("session")
	if len(state.Memory.ShortTerm) != 0 || len(state.Memory.Working) != 1 || len(state.Memory.LongTerm) != 1 {
		t.Fatalf("restored layers = %#v", state.Memory)
	}
}

func TestSelectedDialogueMessageIsStoredWholeWithoutUserKey(t *testing.T) {
	agent := New(&fakeCompleter{answer: "Ответ агента"})
	if _, err := agent.Respond(context.Background(), "session", "Не показывать карту курьера", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryWorking, Value: "Не показывать карту курьера"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryLongTerm, Category: "decisions", Value: "Ответ агента"}); err != nil {
		t.Fatal(err)
	}
	state := agent.State("session")
	if got := state.Memory.Working; len(got) != 1 || got[0].Key != "message-1" || got[0].Value != "Не показывать карту курьера" {
		t.Fatalf("working memory = %#v", got)
	}
	if got := state.Memory.LongTerm; len(got) != 1 || got[0].Key != "message-2" || got[0].Category != "decisions" || got[0].Value != "Ответ агента" {
		t.Fatalf("long-term memory = %#v", got)
	}
}

func TestModelSelectionPersistsAndUsesSelectedModel(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ Pro"}
	agent := New(client)
	result, err := agent.RespondWithOptions(context.Background(), "session", "Составь план", 2, models.StrategySlidingWindow, models.DeepSeekProModel)
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != models.DeepSeekProModel || len(client.models) != 1 || client.models[0] != models.DeepSeekProModel {
		t.Fatalf("selected model state=%q calls=%#v", result.Model, client.models)
	}
	if state := agent.State("session"); state.Model != models.DeepSeekProModel {
		t.Fatalf("model after response = %q", state.Model)
	}
}

func TestBranchingCreatesIndependentChildrenFromCheckpoint(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Общая задача", 2, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	state, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "checkpoint", Name: "до решения"})
	if err != nil || len(state.Checkpoints) != 1 {
		t.Fatalf("checkpoint state = %#v, err=%v", state, err)
	}
	checkpointID := state.Checkpoints[0].ID
	state, err = agent.ApplyContextCommand("session", models.ContextCommand{Action: "create_branch", CheckpointID: checkpointID, Name: "вариант A"})
	if err != nil {
		t.Fatal(err)
	}
	branchA := state.ActiveBranchID
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Продолжение A", 0, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "create_branch", CheckpointID: checkpointID, Name: "вариант B"}); err != nil {
		t.Fatal(err)
	}
	if got := agent.History("session"); len(got) != 2 || got[0].Content != "Общая задача" {
		t.Fatalf("new branch history = %#v", got)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "switch_branch", BranchID: branchA}); err != nil {
		t.Fatal(err)
	}
	if got := agent.History("session"); len(got) != 4 || got[2].Content != "Продолжение A" {
		t.Fatalf("restored branch history = %#v", got)
	}
}
