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
		if request.Model != "deepseek-flash" || request.Thinking.Type != "disabled" || request.Temperature == nil || *request.Temperature != 0.7 || request.MaxTokens != 256 || request.TopP != nil {
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

func TestCompleteMessagesModelJSONRequestsJSONModeAndPreservesBudget(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request completionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.Model != "deepseek-flash" || request.ResponseFormat == nil || request.ResponseFormat.Type != "json_object" || request.MaxTokens != 2200 || len(request.Stop) != 0 {
			t.Fatalf("structured completion controls = %#v", request)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"answer\":\"ok\",\"claims\":[]}"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	client := newClient("test-key", server.Client(), server.URL)
	completion, err := client.CompleteMessagesModelJSON(context.Background(), "deepseek-flash", []models.ChatMessage{{Role: "user", Content: "Return JSON"}}, models.GenerationSettings{MaxTokens: 2200})
	if err != nil || completion.FinishReason != "stop" || completion.Answer != `{"answer":"ok","claims":[]}` {
		t.Fatalf("structured completion = %#v, %v", completion, err)
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

func TestCloudRequestPreservesCommonSettingsAndOmitsLocalOptions(t *testing.T) {
	temperature, topP, topK, seed, penalty, think := 0.35, 0.9, 20, int64(42), 1.2, true
	for _, settings := range []models.GenerationSettings{
		{Temperature: &temperature, MaxTokens: 3000},
		{TopP: &topP, MaxTokens: 3000},
	} {
		settings.ContextWindow, settings.TopK, settings.Seed = 2048, &topK, &seed
		settings.RepeatPenalty, settings.LocalJSON, settings.LocalThink = &penalty, true, &think
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]json.RawMessage
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if string(body["max_tokens"]) != "3000" || string(body["model"]) != `"deepseek-flash"` {
				t.Fatalf("cloud budget/model changed: %s", body)
			}
			if settings.TopP != nil {
				if string(body["top_p"]) != "0.9" || body["temperature"] != nil {
					t.Fatalf("top_p mode changed: %s", body)
				}
			} else if string(body["temperature"]) != "0.35" || body["top_p"] != nil {
				t.Fatalf("temperature mode changed: %s", body)
			}
			for _, key := range []string{"contextWindow", "num_ctx", "topK", "top_k", "seed", "repeatPenalty", "repeat_penalty", "localJson", "localThink", "format", "options", "response_format"} {
				if body[key] != nil {
					t.Fatalf("local parameter %q leaked into cloud API", key)
				}
			}
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"Ответ"},"finish_reason":"stop"}]}`))
		}))
		client := newClient("test-key", server.Client(), server.URL)
		result, err := client.CompleteMessagesModel(context.Background(), models.DeepSeekFlashModel, []models.ChatMessage{{Role: "user", Content: "Привет"}}, settings)
		server.Close()
		if err != nil || result.Answer != "Ответ" {
			t.Fatalf("cloud completion = %#v, %v", result, err)
		}
	}
}
