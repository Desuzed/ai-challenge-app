package deepseek

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-challenge-app/internal/models"
)

func TestCompleteBuildsSafeDeepSeekRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s", r.Method)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("authorization = %q", got)
		}
		var request completionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.Model != "deepseek-v4-flash" || request.Thinking.Type != "disabled" || request.Temperature == nil || *request.Temperature != 0.7 || request.MaxTokens != 256 || request.TopP != nil {
			t.Fatalf("unexpected request: %#v", request)
		}
		if len(request.Messages) != 2 || request.Messages[1].Content != "Привет" {
			t.Fatalf("unexpected messages: %#v", request.Messages)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"Здравствуйте!"}}]}`))
	}))
	defer server.Close()

	client := newClient("test-key", &http.Client{Timeout: time.Second}, server.URL)
	temperature := 0.7
	answer, _, err := client.Complete(context.Background(), "Привет", models.ModeUnrestricted, models.GenerationSettings{Temperature: &temperature, MaxTokens: 256})
	if err != nil || answer != "Здравствуйте!" {
		t.Fatalf("Complete() = %q, %v", answer, err)
	}
}

func TestCompleteErrors(t *testing.T) {
	if _, _, err := NewClient("", time.Second).Complete(context.Background(), "hi", models.ModeUnrestricted, models.GenerationSettings{MaxTokens: 512}); !errors.Is(err, ErrNoAPIKey) {
		t.Fatalf("error = %v", err)
	}

	for _, test := range []struct {
		status int
		want   error
	}{
		{http.StatusUnauthorized, ErrUnauthorized},
		{http.StatusTooManyRequests, ErrRateLimited},
		{http.StatusInternalServerError, ErrUpstream},
	} {
		t.Run(http.StatusText(test.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(test.status)
			}))
			defer server.Close()
			_, _, err := newClient("key", server.Client(), server.URL).Complete(context.Background(), "hi", models.ModeUnrestricted, models.GenerationSettings{MaxTokens: 512})
			if !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestCompleteWithSystemUsesProvidedInstruction(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request completionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if len(request.Messages) != 2 || request.Messages[0].Content != "Точная учебная инструкция" || request.Messages[1].Content != "Реши задачу" {
			t.Fatalf("unexpected messages: %#v", request.Messages)
		}
		if request.ResponseFormat != nil || len(request.Stop) != 0 || request.MaxTokens != 320 || request.TopP == nil || *request.TopP != 0.9 {
			t.Fatalf("unexpected request controls: %#v", request)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"Ответ"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	client := newClient("test-key", server.Client(), server.URL)
	topP := 0.9
	answer, finishReason, err := client.CompleteWithSystem(context.Background(), "Точная учебная инструкция", "Реши задачу", models.GenerationSettings{TopP: &topP, MaxTokens: 320})
	if err != nil || answer != "Ответ" || finishReason != "stop" {
		t.Fatalf("CompleteWithSystem() = %q, %q, %v", answer, finishReason, err)
	}
}

func TestRequestControls(t *testing.T) {
	_, format, stop, tokens := requestControls(models.ModeFormat, 512)
	if format == nil || format.Type != "json_object" || len(stop) != 0 || tokens != 512 {
		t.Fatalf("format controls are invalid: %#v %#v %d", format, stop, tokens)
	}

	_, format, stop, tokens = requestControls(models.ModeLength, 512)
	if format != nil || len(stop) != 0 || tokens != 120 {
		t.Fatalf("length controls are invalid: %#v %#v %d", format, stop, tokens)
	}

	_, format, stop, tokens = requestControls(models.ModeFinish, 512)
	if format != nil || len(stop) != 1 || stop[0] != models.StopSequence || tokens != 512 {
		t.Fatalf("finish controls are invalid: %#v %#v %d", format, stop, tokens)
	}

	_, format, stop, tokens = requestControls(models.ModeAll, 512)
	if format == nil || format.Type != "json_object" || len(stop) != 1 || stop[0] != models.StopSequence || tokens != 180 {
		t.Fatalf("all controls are invalid: %#v %#v %d", format, stop, tokens)
	}
}
