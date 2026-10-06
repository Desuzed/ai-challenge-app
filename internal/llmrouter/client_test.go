package llmrouter

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/ollama"
)

func TestLocalModelIDNeverNeedsDeepSeekKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			_, _ = w.Write([]byte(`{"thinking":{"values":[false],"default":false}}`))
			return
		}
		if r.URL.Path != "/api/chat" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"model":"qwen3:4b","message":{"role":"assistant","content":"offline"},"done":true,"done_reason":"stop"}`))
	}))
	defer server.Close()
	client := &Client{DeepSeek: deepseek.NewClient("", time.Second), Ollama: ollama.New(server.URL, time.Second)}
	result, err := client.CompleteMessagesModel(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "user", Content: "test"}}, models.GenerationSettings{MaxTokens: 8})
	if err != nil || result.Answer != "offline" {
		t.Fatalf("local completion = %#v, %v", result, err)
	}
}
