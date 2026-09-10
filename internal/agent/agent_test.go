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
	answer   string
	err      error
	usage    models.ModelUsage
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
	conversation.messages = make([]models.ChatMessage, 100)
	for i := range conversation.messages {
		conversation.messages[i] = models.ChatMessage{Role: "user", Content: strings.Repeat("я", maxMessageCharacters)}
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
