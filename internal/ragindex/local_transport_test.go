package ragindex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestChatEmbeddingReleasesRunnerAfterRetrieval(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if payload["keep_alive"] != "0" || payload["model"] != "embeddinggemma" {
			t.Errorf("chat embedding request must release its runner: %v", payload)
		}
		_, _ = w.Write([]byte(`{"embeddings":[[1,0]]}`))
	}))
	defer server.Close()
	vectors, err := (Ollama{URL: server.URL, Model: "embeddinggemma", KeepAlive: "0"}).Embed(context.Background(), []string{"question"})
	if err != nil || len(vectors) != 1 || len(vectors[0]) != 2 {
		t.Fatalf("vectors=%v err=%v", vectors, err)
	}
}

func TestEmbeddingRejectsRemoteEndpointAndRedirect(t *testing.T) {
	if _, err := (Ollama{URL: "https://example.com", Model: "test"}).Embed(context.Background(), []string{"question"}); err == nil {
		t.Fatal("remote embedding URL accepted")
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://example.com/embedding", http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	if _, err := (Ollama{URL: server.URL, Model: "test"}).Embed(context.Background(), []string{"question"}); err == nil {
		t.Fatal("remote embedding redirect accepted")
	}
}
