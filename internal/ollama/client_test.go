package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai-challenge-app/internal/models"
)

func TestListsOnlyLocalCompletionModelsAndUsesGenerationSettings(t *testing.T) {
	var request chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen3:4b","capabilities":["completion","tools"],"details":{"family":"qwen"}},{"name":"embeddinggemma:latest","capabilities":["embedding"],"details":{"family":"bert"}},{"name":"remote:latest","remote_host":"https://ollama.com","capabilities":["completion"],"details":{"family":"qwen"}}]}`))
		case "/api/chat":
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatal(err)
			}
			_, _ = w.Write([]byte(`{"model":"qwen3:4b","message":{"role":"assistant","thinking":"Скрытая цепочка рассуждений","content":"Привет"},"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":22}`))
		case "/api/show":
			_, _ = w.Write([]byte(`{"thinking":{"values":[true],"default":true}}`))
		default:
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
	}))
	defer server.Close()
	client := New(server.URL, time.Second)
	modelsFound, err := client.ListModels(context.Background())
	if err != nil || len(modelsFound) != 1 || modelsFound[0] != "ollama/qwen3:4b" {
		t.Fatalf("ListModels() = %v, %v", modelsFound, err)
	}
	temperature := 0.25
	completion, err := client.CompleteMessagesModel(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "user", Content: "Привет"}}, models.GenerationSettings{Temperature: &temperature, MaxTokens: 123})
	if err != nil {
		t.Fatal(err)
	}
	if completion.Answer != "Привет" || completion.Usage.InputTokens != 4 || completion.Usage.OutputTokens != 22 {
		t.Fatalf("completion = %#v", completion)
	}
	if request.Model != "qwen3:4b" || request.Stream || !request.Think || request.Options["num_predict"] != float64(123) || request.Options["num_ctx"] != float64(8192) || request.Options["temperature"] != 0.25 {
		t.Fatalf("unexpected local request: %#v", request)
	}
	if strings.Contains(request.Messages[0].Content, "/no_think") || strings.Contains(completion.Answer, "Скрытая цепочка") {
		t.Fatalf("private reasoning leaked into prompt or answer: prompt=%q answer=%q", request.Messages[0].Content, completion.Answer)
	}
}

func TestToolCallsAndTopPUseOllamaNativeShape(t *testing.T) {
	var request chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			_, _ = w.Write([]byte(`{"thinking":{"values":[true],"default":true}}`))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","tool_calls":[{"function":{"name":"lookup","arguments":{"id":7}}}]},"done":true,"done_reason":"stop"}`))
	}))
	defer server.Close()
	topP := 0.8
	client := New(server.URL, time.Second)
	completion, err := client.CompleteMessagesModelWithTools(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "assistant", ToolCalls: []models.ToolCall{{ID: "call-1", Type: "function", Function: models.ToolCallFunction{Name: "lookup", Arguments: `{"id":7}`}}}}, {Role: "tool", ToolCallID: "call-1", Content: "found"}}, models.GenerationSettings{TopP: &topP, MaxTokens: 96}, []models.ToolDefinition{{Type: "function", Function: models.ToolFunction{Name: "lookup"}}})
	if err != nil {
		t.Fatal(err)
	}
	if request.Options["top_p"] != 0.8 {
		t.Fatalf("options = %#v", request.Options)
	}
	if _, exists := request.Options["temperature"]; exists {
		t.Fatalf("temperature sent with top_p: %#v", request.Options)
	}
	if len(request.Messages[0].ToolCalls) != 1 || request.Messages[0].ToolCalls[0].Function.Arguments["id"] != float64(7) || request.Messages[1].ToolCallID != "call-1" {
		t.Fatalf("native tool messages = %#v", request.Messages)
	}
	if len(completion.ToolCalls) != 1 || completion.ToolCalls[0].ID == "" || completion.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("returned tool calls = %#v", completion.ToolCalls)
	}
}

func TestThinkingOnlyTruncatedResponseKeepsNoReasoningInAnswer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			_, _ = w.Write([]byte(`{"thinking":{"values":[true],"default":true}}`))
			return
		}
		_, _ = w.Write([]byte(`{"message":{"thinking":"private reasoning","content":""},"done":true,"done_reason":"length","prompt_eval_count":8,"eval_count":120}`))
	}))
	defer server.Close()
	result, err := New(server.URL, time.Second).CompleteMessagesModel(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "user", Content: "Вычисли ответ"}}, models.GenerationSettings{MaxTokens: 32})
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "" || result.FinishReason != "length" || result.Usage.OutputTokens != 120 {
		t.Fatalf("result = %#v", result)
	}
	if strings.Contains(result.Answer, "private reasoning") {
		t.Fatal("reasoning leaked into answer")
	}
}

func TestShowFailureDoesNotSendChatWithUnknownThinkingMode(t *testing.T) {
	chatCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		if r.URL.Path == "/api/chat" {
			chatCalls++
		}
	}))
	defer server.Close()
	_, err := New(server.URL, time.Second).CompleteMessagesModel(context.Background(), "ollama/qwen3:4b", []models.ChatMessage{{Role: "user", Content: "test"}}, models.GenerationSettings{MaxTokens: 32})
	if err == nil || !strings.Contains(err.Error(), "безопасно определить режим рассуждений") {
		t.Fatalf("error = %v", err)
	}
	if chatCalls != 0 {
		t.Fatalf("chat calls with unknown thinking mode = %d", chatCalls)
	}
}

func TestLocalClientRejectsRemoteBaseURL(t *testing.T) {
	if _, err := New("https://ollama.com", time.Second).ListModels(context.Background()); err == nil || !strings.Contains(err.Error(), "локальный адрес") {
		t.Fatalf("error = %v", err)
	}
}
