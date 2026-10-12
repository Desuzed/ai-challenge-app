package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-challenge-app/internal/models"
)

func TestRemoteModelDefaultsAndCloudExclusion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen3:4b"},{"name":"cloud-alias","remote_host":"https://ollama.com"}]}`))
		case "/api/show":
			_, _ = w.Write([]byte(`{"thinking":{"default":false}}`))
		case "/api/chat":
			var request chatRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Error(err)
			}
			if request.Model != "qwen3:4b" {
				t.Errorf("model = %q", request.Model)
			}
			for _, field := range []string{"num_ctx", "num_predict"} {
				if _, found := request.Options[field]; found {
					t.Errorf("unexpected imposed budget: %s", field)
				}
			}
			_, _ = w.Write([]byte(`{"message":{"content":"remote"},"done":true,"done_reason":"stop"}`))
		default:
			t.Errorf("unexpected endpoint %s", r.URL.Path)
		}
	}))
	defer server.Close()
	client := NewRemote(server.URL, time.Second)
	catalog, err := client.ListModels(context.Background())
	if err != nil || len(catalog) != 1 || catalog[0] != "ollama-remote/qwen3:4b" {
		t.Fatalf("catalog=%v err=%v", catalog, err)
	}
	completion, err := client.CompleteMessagesModel(context.Background(), catalog[0], []models.ChatMessage{{Role: "user", Content: "hi"}}, models.GenerationSettings{})
	if err != nil || completion.Answer != "remote" {
		t.Fatalf("completion=%#v err=%v", completion, err)
	}
	if !NewRemote("http://192.0.2.10:11434", 0).remote {
		t.Fatal("remote connection lost its identity")
	}
	if err := New("http://192.0.2.10:11434", time.Second).validateURL(); err == nil {
		t.Fatal("offline client accepted LAN")
	}
}

func TestRemoteDoesNotFollowRedirect(t *testing.T) {
	followed := false
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { followed = true }))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()
	if _, err := NewRemote(source.URL, time.Second).ListModels(context.Background()); err == nil {
		t.Fatal("redirect was accepted")
	}
	if followed {
		t.Fatal("remote request followed a redirect to another host")
	}
}
