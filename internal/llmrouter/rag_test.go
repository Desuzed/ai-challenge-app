package llmrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"ai-challenge-app/internal/agent"
	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/ollama"
	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

type localRAGEmbedder struct{}

func (localRAGEmbedder) Embed(_ context.Context, texts []string) ([][]float64, error) {
	vectors := make([][]float64, len(texts))
	for i := range vectors {
		vectors[i] = []float64{1, 0}
	}
	return vectors, nil
}

// Exercise the existing chat agent, saved index, provider router and Ollama
// adapter together. A missing cloud key must not affect local RAG.
func TestExistingChatLocalRAGUsesIndexAndValidatedOllamaEvidence(t *testing.T) {
	root := t.TempDir()
	quote := "История агента хранится в файле .local/agent-history.json."
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte("# История агента\n"+quote+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	docs, err := ragindex.Load(root, []string{"guide.md"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ragindex.ChunkDocuments(docs, "structure")
	if err != nil {
		t.Fatal(err)
	}
	for i := range chunks {
		chunks[i].Vector = []float64{1, 0}
	}
	data, err := json.Marshal(ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "index-structure.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	question := "Где хранится история агента?"
	chatCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			_, _ = w.Write([]byte(`{"thinking":{"values":[true],"default":true}}`))
		case "/api/chat":
			chatCalls++
			var sent struct {
				Model    string               `json:"model"`
				Think    bool                 `json:"think"`
				Format   map[string]any       `json:"format"`
				Messages []models.ChatMessage `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
				t.Error(err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if sent.Model != "qwen3:4b" || sent.Think || sent.Format["type"] != "object" {
				t.Errorf("local RAG request lacks model, schema or think=false: %+v", sent)
			}
			var hasQuestion, hasEvidence bool
			for _, message := range sent.Messages {
				hasQuestion = hasQuestion || (message.Role == "user" && message.Content == question)
				hasEvidence = hasEvidence || strings.Contains(message.Content, quote)
			}
			if !hasQuestion || !hasEvidence {
				t.Error("original question and indexed evidence must reach local model")
			}
			if chatCalls == 1 {
				_ = json.NewEncoder(w).Encode(map[string]any{"message": map[string]string{"content": `{"answer":"","claims":[`}, "done": true, "done_reason": "length", "prompt_eval_count": 100, "eval_count": 512})
				return
			}
			envelope, _ := json.Marshal(map[string]any{"answer": "", "claims": []any{map[string]any{
				"text": quote, "evidence": []any{map[string]string{"chunkId": chunks[0].ChunkID, "quote": quote}},
			}}})
			_ = json.NewEncoder(w).Encode(map[string]any{"model": "qwen3:4b", "message": map[string]string{"role": "assistant", "content": string(envelope)}, "done": true, "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 80})
		default:
			t.Errorf("unexpected local request: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	client := &Client{DeepSeek: deepseek.NewClient("", time.Second), Ollama: ollama.New(server.URL, time.Second)}
	a := agent.New(client)
	a.SetRetriever(&rag.Searcher{Root: root, IndexDir: root, Model: "test-model", Embedder: localRAGEmbedder{}})
	result, err := a.RespondWithUserOptionsRAGConfiguredSettings(context.Background(), "user", "session", question, 2, models.StrategySlidingWindow, "ollama/qwen3:4b", true, models.RAGOptions{}, models.GenerationSettings{MaxTokens: 512})
	if err != nil {
		t.Fatal(err)
	}
	if chatCalls != 2 || result.Model != "ollama/qwen3:4b" || !result.RAGEnabled || result.RAGAbstained || len(result.RAGSources) != 1 || result.RAGSources[0].Quote != quote || !strings.Contains(result.Answer, quote) {
		t.Fatalf("existing local chat did not return validated RAG answer: %+v", result)
	}
	if result.RAGTrace.CompletionAttempts != 2 || result.RAGTrace.OutputTokenBudget != 2200 || result.Tokens.ReservedOutputTokens != 2200 || result.Tokens.RequestTokens != 200 || result.Tokens.ResponseTokens != 592 || result.Settings.MaxTokens != 512 || len(result.Messages) != 2 {
		t.Fatalf("retry usage, budget, settings or history are inconsistent: %+v", result)
	}
}
