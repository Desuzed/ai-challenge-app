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

func TestSameModelOnTwoServersRoutesByProvider(t *testing.T) {
	makeServer := func(label string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				_, _ = w.Write([]byte(`{"thinking":{"default":false}}`))
				return
			}
			_, _ = w.Write([]byte(`{"message":{"content":"` + label + `"},"done":true,"done_reason":"stop"}`))
		}))
	}
	local, remote := makeServer("mac"), makeServer("windows")
	defer local.Close()
	defer remote.Close()
	client := &Client{Ollama: ollama.New(local.URL, time.Second), RemoteOllama: ollama.NewRemote(remote.URL, time.Second)}
	for _, item := range []struct{ model, want string }{{"ollama/qwen3:4b", "mac"}, {"ollama-remote/qwen3:4b", "windows"}} {
		result, err := client.CompleteMessagesModel(context.Background(), item.model, nil, models.GenerationSettings{MaxTokens: 8})
		if err != nil || result.Answer != item.want {
			t.Fatalf("%s => %#v %v", item.model, result, err)
		}
	}
	client.RemoteOllama = nil
	if _, err := client.CompleteMessagesModel(context.Background(), "ollama-remote/qwen3:4b", nil, models.GenerationSettings{}); err == nil {
		t.Fatal("missing remote provider fell through")
	}
}
