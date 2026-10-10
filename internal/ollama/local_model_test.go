package ollama

import (
	"ai-challenge-app/internal/models"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestQwenStructuredOutputDisablesThinkingWithoutChangingMessages(t *testing.T) {
	var sent chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			_, _ = w.Write([]byte(`{"thinking":{"values":[true],"default":true}}`))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"message":{"content":"{\"claims\":[]}"},"done":true,"done_reason":"stop"}`))
	}))
	defer server.Close()
	client := New(server.URL, time.Second)
	_, err := client.CompleteMessagesModelJSON(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "user", Content: "исходный вопрос"}}, models.GenerationSettings{MaxTokens: 2048})
	if err != nil || sent.Think || sent.Format != "json" || sent.Messages[0].Content != "исходный вопрос" {
		t.Fatalf("err=%v request=%+v", err, sent)
	}
}

func TestRemoteAliasNeverReachesChatEndpoint(t *testing.T) {
	chatCalled := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			_, _ = w.Write([]byte(`{"remote_host":"https://ollama.com","remote_model":"remote"}`))
			return
		}
		chatCalled = true
	}))
	defer server.Close()
	client := New(server.URL, time.Second)
	if _, err := client.CompleteMessagesModelJSON(context.Background(), "ollama/local-looking-alias", nil, models.GenerationSettings{MaxTokens: 20}); err == nil || chatCalled {
		t.Fatalf("err=%v chatCalled=%v", err, chatCalled)
	}
}

func TestGroundedJSONSendsNativeSchemaToOllama(t *testing.T) {
	var sent chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			_, _ = w.Write([]byte(`{"thinking":{"values":[true]}}`))
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&sent)
		_, _ = w.Write([]byte(`{"message":{"content":"{\"answer\":\"нет\",\"claims\":[]}"},"done":true,"done_reason":"stop"}`))
	}))
	defer server.Close()
	_, err := New(server.URL, time.Second).CompleteMessagesModelJSON(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "system", Content: `{"claims":[{"evidence":[{"chunkId":"..."}]}]}`}}, models.GenerationSettings{MaxTokens: 512})
	schema, ok := sent.Format.(map[string]any)
	if err != nil || !ok || schema["type"] != "object" || sent.Think {
		t.Fatalf("err=%v request=%+v", err, sent)
	}
}
