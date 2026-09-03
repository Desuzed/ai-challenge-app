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
	mode     models.ResponseMode
}

func (f *fakeClient) Complete(_ context.Context, prompt string, mode models.ResponseMode, settings models.GenerationSettings) (string, string, error) {
	f.prompt = prompt
	f.mode = mode
	f.settings = settings
	return f.answer, "stop", f.err
}

func TestChatSuccess(t *testing.T) {
	client := &fakeClient{answer: "HTTP — это способ общения браузера и сервера."}
	recorder := postJSON(t, New(client), `{"prompt":"  Что такое HTTP?  ","mode":"unrestricted","settings":{"topP":0.9,"maxTokens":256}}`)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if client.prompt != "Что такое HTTP?" {
		t.Fatalf("prompt = %q", client.prompt)
	}
	if client.mode != models.ModeUnrestricted {
		t.Fatalf("mode = %q", client.mode)
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
		{"empty question", `{"prompt":"   ","mode":"unrestricted"}`},
		{"unknown field", `{"prompt":"hi","apiKey":"secret"}`},
		{"two sampling modes", `{"prompt":"hi","mode":"unrestricted","settings":{"temperature":0.7,"topP":0.9}}`},
		{"unknown response mode", `{"prompt":"hi","mode":"other"}`},
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
			recorder := postJSON(t, New(&fakeClient{err: test.err}), `{"prompt":"Hello","mode":"unrestricted"}`)
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

func TestModeTokenLimits(t *testing.T) {
	for _, test := range []struct {
		mode models.ResponseMode
		want int
	}{
		{models.ModeUnrestricted, 512},
		{models.ModeLength, 120},
		{models.ModeAll, 180},
	} {
		settings := models.GenerationSettings{MaxTokens: 512}
		applyModeTokenLimit(&settings, test.mode)
		if settings.MaxTokens != test.want {
			t.Fatalf("mode %q: maxTokens = %d, want %d", test.mode, settings.MaxTokens, test.want)
		}
	}
}

func TestChatDefaultsMissingModeToUnrestricted(t *testing.T) {
	client := &fakeClient{answer: "ok"}
	recorder := postJSON(t, New(client), `{"prompt":"Кто снимался в фильме интерстеллар"}`)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if client.mode != models.ModeUnrestricted {
		t.Fatalf("mode = %q, want %q", client.mode, models.ModeUnrestricted)
	}
}

func TestStopSequenceForMode(t *testing.T) {
	if got := stopSequenceForMode(models.ModeFinish); got != models.StopSequence {
		t.Fatalf("finish stop sequence = %q", got)
	}
	if got := stopSequenceForMode(models.ModeAll); got != models.StopSequence {
		t.Fatalf("all stop sequence = %q", got)
	}
	if got := stopSequenceForMode(models.ModeFormat); got != "" {
		t.Fatalf("format stop sequence = %q", got)
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
