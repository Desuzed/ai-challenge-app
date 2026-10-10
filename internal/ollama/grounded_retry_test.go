package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"ai-challenge-app/internal/models"
)

func TestCompactGroundedOutputRetriesOnlyTruncatedJSONOnce(t *testing.T) {
	valid := `{"answer":"","claims":[]}`
	for _, tt := range []struct {
		name, first, reason, second string
		compact                     bool
		documents                   bool
		wantCalls                   int
	}{
		{"recover", `{"answer":"","claims":[`, "length", valid, true, false, 2},
		{"still-truncated", `{"answer":"","claims":[`, "length", `{"answer":""`, true, false, 2},
		{"complete-at-limit", valid, "length", "", true, false, 1},
		{"bad-format-without-limit", "bad", "stop", "", true, false, 1},
		{"ordinary-json", `{"partial":`, "length", "", false, false, 1},
		{"empty-with-documents", valid, "stop", `{"answer":"","claims":[{"text":"Факт","evidence":[{"chunkId":"c","quote":"Факт"}]}]}`, true, true, 2},
		{"unknown-stays-empty", valid, "stop", valid, true, true, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var requests []chatRequest
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/show" {
					_, _ = w.Write([]byte(`{"thinking":{"values":[true]}}`))
					return
				}
				var sent chatRequest
				if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
					t.Error(err)
				}
				requests = append(requests, sent)
				answer, reason := tt.first, tt.reason
				if len(requests) == 2 {
					answer = tt.second
					if json.Valid([]byte(answer)) {
						reason = "stop"
					}
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"message": map[string]string{"content": answer}, "done": true, "done_reason": reason, "prompt_eval_count": 100, "eval_count": 50})
			}))
			defer server.Close()
			messages := []models.ChatMessage{{Role: "user", Content: "Вопрос"}}
			if tt.compact {
				messages = append([]models.ChatMessage{{Role: "system", Content: `Формат "claims", "chunkId"; используй "answer":""`}}, messages...)
			}
			if tt.documents {
				messages = append([]models.ChatMessage{{Role: "system", Content: "Фрагменты локальных документов для ответа на последний вопрос.\nФрагмент 1 — [guide.md], chunk_id=c\nФакт"}}, messages...)
			}
			result, err := New(server.URL, time.Second).CompleteMessagesModelJSON(context.Background(), "ollama/qwen3:4b", messages, models.GenerationSettings{MaxTokens: 512})
			if err != nil || len(requests) != tt.wantCalls {
				t.Fatalf("requests=%d err=%v", len(requests), err)
			}
			if tt.wantCalls == 2 {
				if requests[0].Options["num_predict"] != float64(512) || requests[1].Options["num_predict"] != float64(2200) || !reflect.DeepEqual(requests[0].Messages, requests[1].Messages) || result.Answer != tt.second || result.Usage.InputTokens != 200 || result.Usage.OutputTokens != 100 || result.OutputTokenBudget != 2200 || result.CompletionAttempts != 2 {
					t.Fatalf("inconsistent bounded retry: requests=%+v result=%+v", requests, result)
				}
			}
		})
	}
}

func TestCompactSchemaBoundsEnvelopeAndPlannerRetainsPlan(t *testing.T) {
	schema := groundedSchema(true, false)
	properties := schema["properties"].(map[string]any)
	if _, ok := properties["plan"]; ok {
		t.Fatal("ordinary RAG must not generate a plan")
	}
	if !reflect.DeepEqual(properties["answer"].(map[string]any)["enum"], []string{""}) {
		t.Fatal("facts must not be duplicated in answer")
	}
	claims := properties["claims"].(map[string]any)
	fields := claims["items"].(map[string]any)["properties"].(map[string]any)
	evidence := fields["evidence"].(map[string]any)
	quotes := evidence["items"].(map[string]any)["properties"].(map[string]any)
	if claims["maxItems"] != 2 || evidence["maxItems"] != 1 || fields["text"].(map[string]any)["maxLength"] != nil || quotes["quote"].(map[string]any)["maxLength"] != nil {
		t.Fatalf("unbounded compact schema: %+v", schema)
	}
	planner := groundedSchema(false, true)["properties"].(map[string]any)
	if planner["plan"] == nil || planner["answer"].(map[string]any)["enum"] != nil {
		t.Fatal("planner/memory output must keep its fields")
	}
}
