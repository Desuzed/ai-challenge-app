package agent

import (
	"context"
	"errors"
	"path/filepath"
	"reflect"
	"testing"

	"ai-challenge-app/internal/models"
)

type fakeCompleter struct {
	requests [][]models.ChatMessage
	answer   string
	err      error
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
	return models.ModelCompletion{Answer: f.answer}, nil
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
