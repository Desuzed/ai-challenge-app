package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
)

type fakeClient struct {
	answer            string
	err               error
	prompt            string
	settings          models.GenerationSettings
	mode              models.ResponseMode
	reasoningAnswers  []string
	reasoningPrompts  []string
	reasoningSystems  []string
	reasoningSettings []models.GenerationSettings
	catalog           []string
	modelCalls        []string
	modelPrompt       string
	messageRequests   [][]models.ChatMessage
	messageUsage      models.ModelUsage
}

func (f *fakeClient) CompleteMessages(_ context.Context, messages []models.ChatMessage, _ models.GenerationSettings) (models.ModelCompletion, error) {
	f.messageRequests = append(f.messageRequests, append([]models.ChatMessage(nil), messages...))
	return models.ModelCompletion{Answer: f.answer, Usage: f.messageUsage}, f.err
}

func TestTokenDemoUsesSameFinalTaskAndCanForceCacheMiss(t *testing.T) {
	client := &fakeClient{answer: "Итог: 1 375 ₽", messageUsage: models.ModelUsage{InputTokens: 900, OutputTokens: 12, CacheMissTokens: 900}}
	handler := New(client)
	short := httptest.NewRecorder()
	handler.TokenDemo(short, httptest.NewRequest(http.MethodPost, "/api/agent/token-demo", strings.NewReader(`{"scenario":"short","forceCacheMiss":true}`)))
	long := httptest.NewRecorder()
	handler.TokenDemo(long, httptest.NewRequest(http.MethodPost, "/api/agent/token-demo", strings.NewReader(`{"scenario":"long","forceCacheMiss":true}`)))
	if short.Code != http.StatusOK || long.Code != http.StatusOK || len(client.messageRequests) != 2 {
		t.Fatalf("statuses %d/%d, calls %d", short.Code, long.Code, len(client.messageRequests))
	}
	for _, request := range client.messageRequests {
		if got := request[len(request)-1].Content; got != tokenDemoTask {
			t.Fatalf("final task = %q", got)
		}
		if !strings.Contains(request[0].Content, "Уникальный маркер") {
			t.Fatalf("unique cache-bypass marker missing: %q", request[0].Content)
		}
	}
	if len(client.messageRequests[1]) <= len(client.messageRequests[0]) {
		t.Fatal("long scenario did not add history")
	}
}

func TestTokenDemoOverflowDoesNotCallModel(t *testing.T) {
	client := &fakeClient{}
	recorder := httptest.NewRecorder()
	New(client).TokenDemo(recorder, httptest.NewRequest(http.MethodPost, "/api/agent/token-demo", strings.NewReader(`{"scenario":"overflow"}`)))
	if recorder.Code != http.StatusOK || len(client.messageRequests) != 0 {
		t.Fatalf("status %d, calls %d", recorder.Code, len(client.messageRequests))
	}
}

func TestContextDemoRunsFullAndCompressedVersions(t *testing.T) {
	client := &fakeClient{answer: "Проверяемый ответ", messageUsage: models.ModelUsage{InputTokens: 100, OutputTokens: 10}}
	recorder := httptest.NewRecorder()
	New(client).ContextDemo(recorder, httptest.NewRequest(http.MethodPost, "/api/agent/context-demo", nil))
	if recorder.Code != http.StatusOK || len(client.messageRequests) != 2 {
		t.Fatalf("status %d, calls %d", recorder.Code, len(client.messageRequests))
	}
	if len(client.messageRequests[0]) <= len(client.messageRequests[1]) {
		t.Fatal("compressed request must contain fewer messages")
	}
	var result models.ContextDemoResult
	if err := json.NewDecoder(recorder.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.FullAnswer == "" || !strings.Contains(result.CompressedPromptPreview, "Сжатое резюме") {
		t.Fatalf("result = %#v", result)
	}
}

func TestRecentDemoUsesSelectedNWithoutChangingChat(t *testing.T) {
	client := &fakeClient{answer: "Android: сохранять ключ; сервер: дедупликация; iOS: не срочно", messageUsage: models.ModelUsage{InputTokens: 77, OutputTokens: 18}}
	recorder := httptest.NewRecorder()
	New(client).RecentDemo(recorder, httptest.NewRequest(http.MethodPost, "/api/agent/recent-demo", strings.NewReader(`{"recentMessages":4}`)))
	if recorder.Code != http.StatusOK || len(client.messageRequests) != 1 {
		t.Fatalf("status %d, calls %d", recorder.Code, len(client.messageRequests))
	}
	var result models.RecentDemoResult
	if err := json.NewDecoder(recorder.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.RecentMessages != 4 || result.InputTokens != 77 || !strings.Contains(result.PromptPreview, "Сжатое резюме") {
		t.Fatalf("result = %#v", result)
	}
}

func (f *fakeClient) Complete(_ context.Context, prompt string, mode models.ResponseMode, settings models.GenerationSettings) (string, string, error) {
	f.prompt = prompt
	f.mode = mode
	f.settings = settings
	return f.answer, "stop", f.err
}

func (f *fakeClient) CompleteWithSystem(_ context.Context, system, prompt string, settings models.GenerationSettings) (string, string, error) {
	f.reasoningSystems = append(f.reasoningSystems, system)
	f.reasoningPrompts = append(f.reasoningPrompts, prompt)
	f.reasoningSettings = append(f.reasoningSettings, settings)
	if len(f.reasoningAnswers) > 0 {
		answer := f.reasoningAnswers[0]
		f.reasoningAnswers = f.reasoningAnswers[1:]
		return answer, "stop", f.err
	}
	return f.answer, "stop", f.err
}

func (f *fakeClient) ListModels(_ context.Context) ([]string, error) { return f.catalog, f.err }

func (f *fakeClient) CompleteModel(_ context.Context, name, _ string, prompt string, _ models.GenerationSettings) (deepseek.Completion, error) {
	f.modelCalls = append(f.modelCalls, name)
	f.modelPrompt = prompt
	return deepseek.Completion{Answer: "Разбор " + name, FinishReason: "stop", Usage: models.ModelUsage{InputTokens: 100, OutputTokens: 50, TotalTokens: 150}}, f.err
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

func TestModelVersionsUsesFullFixedPrompt(t *testing.T) {
	client := &fakeClient{catalog: []string{"deepseek-v4-pro", "deepseek-v4-flash"}}
	req := httptest.NewRequest(http.MethodPost, "/api/model-versions", strings.NewReader(`{}`))
	recorder := httptest.NewRecorder()
	New(client).ModelVersions(recorder, req)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d", recorder.Code)
	}
	if got := strings.Join(client.modelCalls, ","); got != "deepseek-v4-flash,deepseek-v4-pro" {
		t.Fatalf("model calls = %q", got)
	}
	if client.modelPrompt != modelVersionsPrompt {
		t.Fatal("lesson prompt was changed before request")
	}
	if !strings.Contains(client.modelPrompt, "Предложи минимум пять автоматических проверок.") || !strings.Contains(client.modelPrompt, "Не придумывай отсутствующие факты.") {
		t.Fatal("required prompt parts missing")
	}
	var response models.ModelVersionsResponse
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	if len(response.Runs) != 2 || response.Runs[0].Provider != "DeepSeek" || response.Runs[1].Provider != "DeepSeek" {
		t.Fatalf("runs = %#v", response.Runs)
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
	recorder := postJSON(t, New(client), `{"prompt":"Что такое HTTP?"}`)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if client.mode != models.ModeUnrestricted {
		t.Fatalf("mode = %q, want %q", client.mode, models.ModeUnrestricted)
	}
}

func TestChatPreservesTemperatureExperimentValues(t *testing.T) {
	for _, temperature := range []float64{0, 0.7, 1.2, 1.8} {
		t.Run(fmt.Sprintf("temperature %.1f", temperature), func(t *testing.T) {
			client := &fakeClient{answer: "ok"}
			body := fmt.Sprintf(`{"prompt":"Один и тот же запрос","mode":"unrestricted","settings":{"temperature":%g,"maxTokens":512}}`, temperature)
			recorder := postJSON(t, New(client), body)

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
			}
			if client.settings.TopP != nil || client.settings.Temperature == nil || *client.settings.Temperature != temperature || client.settings.MaxTokens != 512 {
				t.Fatalf("settings = %#v", client.settings)
			}
		})
	}
}

func TestReasoningPromptDesignerUsesGeneratedPrompt(t *testing.T) {
	client := &fakeClient{reasoningAnswers: []string{"Выдели данные, выполни расчёт и проверь итог.", "Ответ: 1/3\nПроверка: условная вероятность учтена."}}
	recorder := postReasoningJSON(t, New(client), `{"task":"  Найди вероятность.  ","approach":"prompt_designer","settings":{"topP":0.9,"maxTokens":256}}`)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if len(client.reasoningPrompts) != 2 || len(client.reasoningSystems) != 2 {
		t.Fatalf("reasoning calls = %d, want 2", len(client.reasoningPrompts))
	}
	if client.reasoningPrompts[0] != "<task>\nНайди вероятность.\n</task>" {
		t.Fatalf("designer prompt = %q", client.reasoningPrompts[0])
	}
	if !strings.Contains(client.reasoningPrompts[1], "<generated_prompt>\nВыдели данные") || !strings.Contains(client.reasoningPrompts[1], "<original_task>\nНайди вероятность.") {
		t.Fatalf("solver prompt = %q", client.reasoningPrompts[1])
	}
	if client.reasoningSettings[0].Temperature == nil || *client.reasoningSettings[0].Temperature != 0.2 || client.reasoningSettings[0].TopP != nil || client.reasoningSettings[0].MaxTokens != 720 {
		t.Fatalf("designer settings = %#v", client.reasoningSettings[0])
	}
	if client.reasoningSettings[1].TopP == nil || *client.reasoningSettings[1].TopP != 0.9 || client.reasoningSettings[1].MaxTokens != 8192 {
		t.Fatalf("solver settings = %#v", client.reasoningSettings[1])
	}

	var response models.ReasoningResponse
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	if response.PreparedPrompt != "Выдели данные, выполни расчёт и проверь итог." || response.Answer != "Ответ: 1/3\nПроверка: условная вероятность учтена." || response.Debug.Requests != 2 {
		t.Fatalf("response = %#v", response)
	}
}

func TestReasoningApproachesUseDifferentInstructions(t *testing.T) {
	tests := []struct {
		approach models.ReasoningApproach
		wantPart string
	}{
		{models.ReasoningDirect, reasoningBaseInstruction},
		{models.ReasoningStepByStep, "Решай пошагово"},
		{models.ReasoningExpertPanel, "Аналитик"},
	}
	for _, test := range tests {
		t.Run(string(test.approach), func(t *testing.T) {
			client := &fakeClient{answer: "Готовое решение"}
			recorder := postReasoningJSON(t, New(client), `{"task":"Задача","approach":"`+string(test.approach)+`","settings":{"maxTokens":256}}`)
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
			}
			if len(client.reasoningSystems) != 1 || !strings.Contains(client.reasoningSystems[0], test.wantPart) {
				t.Fatalf("system prompts = %#v, want part %q", client.reasoningSystems, test.wantPart)
			}
		})
	}
}

func TestReasoningValidation(t *testing.T) {
	client := &fakeClient{answer: "ok"}
	for _, body := range []string{
		`{"task":"","approach":"direct"}`,
		`{"task":"Задача","approach":"unknown"}`,
		`{"task":"Задача","approach":"direct","settings":{"temperature":0.2,"topP":0.9}}`,
	} {
		recorder := postReasoningJSON(t, New(client), body)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("body %s: status = %d, want %d", body, recorder.Code, http.StatusBadRequest)
		}
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

func postReasoningJSON(t *testing.T, handler *Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/reasoning", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.Reasoning(recorder, req)
	return recorder
}
