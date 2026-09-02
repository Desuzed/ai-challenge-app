package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
)

type fakeClient struct {
	answer   string
	err      error
	prompt   string
	settings models.GenerationSettings
}

func (f *fakeClient) Complete(_ context.Context, prompt string, settings models.GenerationSettings) (string, error) {
	f.prompt = prompt
	f.settings = settings
	return f.answer, f.err
}

func TestChatSuccess(t *testing.T) {
	client := &fakeClient{answer: "HTTP — это способ общения браузера и сервера."}
	recorder := postJSON(t, New(client), `{"prompt":"  Что такое HTTP?  ","settings":{"topP":0.9,"maxTokens":256}}`)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if client.prompt != "Что такое HTTP?" {
		t.Fatalf("prompt = %q", client.prompt)
	}
	if client.settings.TopP == nil || *client.settings.TopP != 0.9 || client.settings.MaxTokens != 256 {
		t.Fatalf("settings = %#v", client.settings)
	}
	var response struct {
		Answer string `json:"answer"`
		Debug  struct {
			PromptCharacters int `json:"promptCharacters"`
		}
	}
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	if response.Answer != client.answer {
		t.Fatalf("answer = %q", response.Answer)
	}
	if response.Debug.PromptCharacters != len([]rune("Что такое HTTP?")) {
		t.Fatalf("prompt characters = %d", response.Debug.PromptCharacters)
	}
}

func TestChatValidation(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"empty question", `{"prompt":"   "}`},
		{"unknown field", `{"prompt":"hi","apiKey":"secret"}`},
		{"two sampling modes", `{"prompt":"hi","settings":{"temperature":0.7,"topP":0.9}}`},
		{"invalid JSON", `{`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, New(&fakeClient{}), test.body)
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
			}
		})
	}
}

func TestChatErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
	}{
		{"missing key", deepseek.ErrNoAPIKey, http.StatusServiceUnavailable},
		{"rejected key", deepseek.ErrUnauthorized, http.StatusBadGateway},
		{"rate limited", deepseek.ErrRateLimited, http.StatusTooManyRequests},
		{"upstream", errors.New("network"), http.StatusBadGateway},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, New(&fakeClient{err: test.err}), `{"prompt":"Hello"}`)
			if recorder.Code != test.want {
				t.Fatalf("status = %d, want %d", recorder.Code, test.want)
			}
		})
	}
}

func TestChatOnlyAllowsPOST(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
	recorder := httptest.NewRecorder()
	New(&fakeClient{}).Chat(recorder, req)
	if recorder.Code != http.StatusMethodNotAllowed || recorder.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("unexpected response: %d, Allow=%q", recorder.Code, recorder.Header().Get("Allow"))
	}
}

func postJSON(t *testing.T, handler *Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.Chat(recorder, req)
	return recorder
}
