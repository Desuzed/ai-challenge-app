package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

type ragTestEmbedder struct{}

func (ragTestEmbedder) Embed(context.Context, []string) ([][]float64, error) {
	return [][]float64{{1, 0}}, nil
}

func TestRAGToggleAddsSourcesOnlyToEnabledRequest(t *testing.T) {
	root := t.TempDir()
	content := "# Storage\nThe private project uses a local JSON file for agent history.\n"
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte(content), 0600); err != nil {
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
	chunks[0].Vector = []float64{1, 0}
	data, err := json.Marshal(ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks})
	if err != nil {
		t.Fatal(err)
	}
	indexDir := filepath.Join(root, "index")
	if err := os.Mkdir(indexDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(indexDir, "index-structure.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	client := &fakeCompleter{answer: `{"claims":[{"text":"История хранится в локальном JSON-файле.","evidence":[{"chunkId":"` + chunks[0].ChunkID + `","quote":"The private project uses a local JSON file for agent history."}]}]}`}
	a := New(client)
	a.SetRetriever(&rag.Searcher{Root: root, IndexDir: indexDir, Model: "test-model", Embedder: ragTestEmbedder{}})
	for _, session := range []string{"plain", "with-rag"} {
		if _, err := a.ApplyContextCommand(session, models.ContextCommand{Action: "set_planner_mode", PlannerMode: "disabled"}); err != nil {
			t.Fatal(err)
		}
	}
	question := "Где хранится история проекта?"
	plain, err := a.RespondWithUserOptionsRAG(context.Background(), "plain", "plain", question, 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, false)
	if err != nil {
		t.Fatal(err)
	}
	withRAG, err := a.RespondWithUserOptionsRAG(context.Background(), "with-rag", "with-rag", question, 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, true)
	if err != nil {
		t.Fatal(err)
	}
	if plain.RAGEnabled || len(plain.RAGSources) != 0 || len(withRAG.RAGSources) != 1 || !withRAG.RAGEnabled {
		t.Fatalf("plain=%#v rag=%#v", plain.RAGSources, withRAG.RAGSources)
	}
	if withRAG.RAGSources[0].Quote != "The private project uses a local JSON file for agent history." || !strings.Contains(withRAG.Answer, withRAG.RAGSources[0].Quote) {
		t.Fatalf("answer/source did not carry the verified quote: %#v", withRAG)
	}
	if strings.Contains(fmt.Sprint(plain.RequestMessages), "private project") || !strings.Contains(fmt.Sprint(withRAG.RequestMessages), "private project") {
		t.Fatal("RAG context was not isolated to enabled request")
	}
	if !strings.Contains(fmt.Sprint(withRAG.Messages), "The private project uses a local JSON file for agent history.") {
		t.Fatal("the persistent RAG answer did not retain its verified quote")
	}
}

func TestRAGStructuredEnvelopeAcceptsWrappedJSONAndRejectsTruncation(t *testing.T) {
	question := "Чем сохранённая история отличается от сообщений, отправляемых модели в текущем запросе?"
	forms := []struct {
		name         string
		wrap         func(string) string
		finishReason string
		wantValid    bool
		wantReason   string
	}{
		{"plain", func(value string) string { return value }, "stop", true, ""},
		{"markdown-fence", func(value string) string { return "```json\n" + value + "\n```" }, "stop", true, ""},
		{"surrounding-prose", func(value string) string { return "Ответ в формате JSON:\n" + value + "\nГотово." }, "stop", true, ""},
		{"truncated-at-token-limit", func(value string) string { return strings.TrimSuffix(value, "}") }, "length", false, "structured_output_truncated"},
		{"nested-envelope-inside-truncated-root", func(value string) string { return `{"nested":` + value }, "stop", false, "invalid_structured_output"},
		{"two-envelopes", func(value string) string { return value + "\n" + value }, "stop", false, "invalid_structured_output"},
	}
	for _, test := range forms {
		t.Run(test.name, func(t *testing.T) {
			a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
			searchOptions := rag.DefaultSearchOptions()
			searchOptions.Rewrite = false
			searchOptions.Rerank = false
			found, err := a.retriever.SearchWithOptions(context.Background(), question, searchOptions)
			if err != nil || len(found.Matches) == 0 {
				t.Fatalf("fixture retrieval failed: matches=%d err=%v", len(found.Matches), err)
			}
			quote := strings.TrimSpace(found.Matches[0].Chunk.Text)
			validJSON := fmt.Sprintf(`{"answer":"Подтверждённый факт о хранении.","claims":[{"text":"История агента хранится в локальном JSON-файле.","evidence":[{"chunkId":%q,"quote":%q}]}]}`, found.Matches[0].Chunk.ChunkID, quote)
			client.answer = test.wrap(validJSON)
			client.finishReason = test.finishReason
			structuredClient := &structuredFakeCompleter{fakeCompleter: client}
			a.client = structuredClient
			response, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", question, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{Rewrite: boolPointer(false), Rerank: boolPointer(false)})
			if err != nil {
				t.Fatal(err)
			}
			if test.wantValid {
				if response.RAGAbstained || len(response.RAGSources) != 1 || response.RAGSources[0].Quote != quote || !strings.Contains(response.Answer, "История агента хранится в локальном JSON-файле") {
					t.Fatalf("valid structured output was rejected or ungrounded: %#v", response)
				}
				if response.RAGTrace == nil || !response.RAGTrace.StructuredOutput || response.RAGTrace.OutputTokenBudget != 2200 || response.RAGTrace.CompletionFinishReason != "stop" || response.RAGTrace.CompletionCharacters == 0 || structuredClient.calls != 1 || len(client.settings) != 1 || client.settings[0].MaxTokens != 2200 {
					t.Fatalf("structured completion diagnostics missing: trace=%#v calls=%d settings=%#v", response.RAGTrace, structuredClient.calls, client.settings)
				}
			} else if !response.RAGAbstained || response.RAGReason != test.wantReason || len(response.RAGSources) != 0 || strings.Contains(response.Answer, "История агента хранится") {
				t.Fatalf("malformed output was not safely rejected with diagnostics: %#v", response)
			}
		})
	}
}

func TestDetailedGroundedRequestOverridesEarlierBriefPreferenceAndReservesMoreTokens(t *testing.T) {
	question := "Подготовь подробное итоговое объяснение для моего видео."
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	if _, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", "Ограничения: краткий итоговый ответ", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, false, models.RAGOptions{}); err != nil {
		t.Fatal(err)
	}
	match, err := a.retriever.SearchWithOptions(context.Background(), question, rag.DefaultSearchOptions())
	if err != nil || len(match.Matches) == 0 {
		t.Fatalf("fixture retrieval failed: result=%#v err=%v", match, err)
	}
	quote := strings.TrimSpace(match.Matches[0].Chunk.Text)
	client.answer = fmt.Sprintf(`{"answer":"Краткий и непроверяемый пересказ.","claims":[{"text":"Развёрнутое подтверждённое объяснение первого этапа.","evidence":[{"chunkId":%q,"quote":%q}]}]}`, match.Matches[0].Chunk.ChunkID, quote)
	structured := &structuredFakeCompleter{fakeCompleter: client}
	a.client = structured
	response, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", question, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.settings) != 2 || client.settings[1].MaxTokens != detailedRAGMaxTokens || response.Tokens.ReservedOutputTokens != detailedRAGMaxTokens || response.RAGTrace == nil || response.RAGTrace.OutputTokenBudget != detailedRAGMaxTokens {
		t.Fatalf("detailed response budget was not applied consistently: settings=%#v tokens=%#v trace=%#v", client.settings, response.Tokens, response.RAGTrace)
	}
	if len(response.RAGSources) != 1 || response.RAGSources[0].Quote != quote || !strings.Contains(response.Answer, "Итоговое объяснение\n1. Развёрнутое подтверждённое объяснение первого этапа.") || !strings.Contains(response.Answer, quote) || strings.Contains(response.Answer, "Краткий и непроверяемый пересказ") {
		t.Fatalf("detailed answer was not rendered from verified claims and sources: %#v", response)
	}
	request := fmt.Sprint(client.requests[len(client.requests)-1])
	if !strings.Contains(request, "приоритет над более ранней просьбой о кратком стиле") || !strings.Contains(request, "Ограничения (ход 1): краткий итоговый ответ") {
		t.Fatalf("latest detailed style did not override older preference explicitly: %s", request)
	}
}

func boolPointer(value bool) *bool { return &value }

func TestRAGPlannerAppliesOnlyCanonicalValidatedEnvelope(t *testing.T) {
	question := "Уточни текущий план по документации."
	a, client := makeRAGFixture(t, []float64{1, 0}, "", models.ModelUsage{})
	searchOptions := rag.DefaultSearchOptions()
	searchOptions.Rewrite, searchOptions.Rerank = false, false
	found, err := a.retriever.SearchWithOptions(context.Background(), question, searchOptions)
	if err != nil || len(found.Matches) == 0 {
		t.Fatalf("fixture retrieval failed: matches=%d err=%v", len(found.Matches), err)
	}
	quote := strings.TrimSpace(found.Matches[0].Chunk.Text)
	client.answer = fmt.Sprintf("Ответ ниже:\n```json\n{\"answer\":\"План обновлён по пользовательской цели.\",\"claims\":[{\"text\":\"История агента хранится в локальном JSON-файле.\",\"evidence\":[{\"chunkId\":%q,\"quote\":%q}]}],\"plan\":{\"phase\":\"planning\",\"specification\":\"Проверенная спецификация\",\"currentStep\":\"Сверить шаги\",\"expectedAction\":\"Подтвердить план\",\"openQuestions\":[],\"decisions\":[],\"nextSteps\":[\"Проверить цитату\"]}}\n```\nГотово.", found.Matches[0].Chunk.ChunkID, quote)
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "enabled"}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommandForUser("u", "s", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Объяснить хранение", Phases: []string{"planning", "execution"}}}); err != nil {
		t.Fatal(err)
	}
	response, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", question, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{Rewrite: boolPointer(false), Rerank: boolPointer(false)})
	if err != nil {
		t.Fatal(err)
	}
	if response.RAGAbstained || len(response.RAGSources) != 1 || response.Task.Specification != "Проверенная спецификация" || response.Task.Phase != "planning" || response.Task.PlanApproved {
		t.Fatalf("canonical grounded planner update was rejected or advanced lifecycle: %#v", response)
	}
	if !strings.Contains(response.Answer, "Предложение по плану задачи") || strings.Contains(response.Answer, "Ответ ниже:") || strings.Contains(response.Answer, "Готово.") {
		t.Fatalf("planner response escaped canonical JSON root: %q", response.Answer)
	}
}

func TestValidateGroundedClaimsRequiresExactEvidenceForEveryClaim(t *testing.T) {
	matches := []rag.Match{{Chunk: ragindex.Chunk{ChunkID: "chunk-1", Text: "History is stored in a local JSON file."}}}
	valid := `{"claims":[{"text":"History is stored locally.","evidence":[{"chunkId":"chunk-1","quote":"stored in a local JSON file"}]}]}`
	claims := validateGroundedClaims(valid, matches)
	if len(claims) != 1 || claims[0].Evidence[0].Quote != "stored in a local JSON file" {
		t.Fatalf("valid evidence rejected: %#v", claims)
	}
	invalid := `{"claims":[{"text":"Made-up statement.","evidence":[{"chunkId":"chunk-1","quote":"not in the chunk"}]},{"text":"Unknown chunk.","evidence":[{"chunkId":"chunk-2","quote":"text"}]}]}`
	if got := validateGroundedClaims(invalid, matches); len(got) != 0 {
		t.Fatalf("unsupported evidence passed validation: %#v", got)
	}
	partial := `{"claims":[{"text":"Composite claim.","evidence":[{"chunkId":"chunk-1","quote":"local JSON file"},{"chunkId":"chunk-1","quote":"fabricated"}]}]}`
	if got := validateGroundedClaims(partial, matches); len(got) != 0 {
		t.Fatalf("partially supported compound claim passed validation: %#v", got)
	}
}

func makeRAGFixture(t *testing.T, vector []float64, answer string, usage models.ModelUsage) (*Agent, *fakeCompleter) {
	t.Helper()
	root := t.TempDir()
	content := "# Storage\nThe private project uses a local JSON file for agent history.\n"
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte(content), 0600); err != nil {
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
	chunks[0].Vector = vector
	data, err := json.Marshal(ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks})
	if err != nil {
		t.Fatal(err)
	}
	indexDir := filepath.Join(root, "index")
	if err = os.Mkdir(indexDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(indexDir, "index-structure.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	client := &fakeCompleter{answer: answer, usage: usage}
	a := New(client)
	a.SetRetriever(&rag.Searcher{Root: root, IndexDir: indexDir, Model: "test-model", Embedder: ragTestEmbedder{}})
	return a, client
}

func TestRAGBelowThresholdAbstainsWithoutModelCallAndReturnsEmptyJSONSources(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{0, 1}, `{"claims":[]}`, models.ModelUsage{})
	response, err := a.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", "Вопрос о несуществующей теме", 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{MinSimilarity: 0.4})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 0 || !response.RAGModelCallSkipped || !response.RAGAbstained || response.RAGReason != "retrieval_empty" || response.RAGTrace == nil || response.RAGTrace.ClaimedCount != 0 || response.RAGTrace.VerifiedClaimCount != 0 {
		t.Fatalf("expected pre-model refusal; client calls=%d response=%#v", len(client.requests), response)
	}
	if !strings.HasPrefix(response.Answer, "Не знаю") || !strings.Contains(response.Answer, "Уточните") || response.RAGSources == nil || len(response.RAGSources) != 0 {
		t.Fatalf("invalid refusal contract: %#v", response)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err = json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if sources, ok := decoded["ragSources"].([]any); !ok || len(sources) != 0 {
		t.Fatalf("ragSources must be JSON []: %s", encoded)
	}
	if response.Messages[len(response.Messages)-1].Content != response.Answer || response.Tokens.RequestTokens != 0 || response.Tokens.ResponseTokens != 0 || len(response.RequestMessages) != 0 {
		t.Fatalf("refusal history/token/request mismatch: %#v", response)
	}
}

func TestRAGTraceDistinguishesModelNoClaimsFromEmptyRetrieval(t *testing.T) {
	a, client := makeRAGFixture(t, []float64{1, 0}, `{"claims":[]}`, models.ModelUsage{})
	response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "Где хранится история агента?", 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, true)
	if err != nil {
		t.Fatal(err)
	}
	if response.RAGReason != "model_no_claims" || response.RAGTrace == nil || response.RAGTrace.ClaimedCount != 0 || response.RAGTrace.VerifiedClaimCount != 0 || len(client.requests) != 1 {
		t.Fatalf("empty model claims were confused with retrieval failure: %#v", response)
	}
}

func TestRAGInvalidQuoteAbstainsAndPersistsTheSameAnswerWithUsage(t *testing.T) {
	answer := `{"claims":[{"text":"Unsupported answer","evidence":[{"chunkId":"invalid-id","quote":"fabricated quote"}]}]}`
	usage := models.ModelUsage{InputTokens: 321, OutputTokens: 54}
	a, client := makeRAGFixture(t, []float64{1, 0}, answer, usage)
	response, err := a.RespondWithUserOptionsRAG(context.Background(), "u", "s", "Где история?", 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 || !response.RAGAbstained || response.RAGModelCallSkipped || len(response.RAGSources) != 0 || response.RAGSources == nil || !strings.HasPrefix(response.Answer, "Не знаю") || response.RAGReason != "invalid_evidence" || response.RAGTrace == nil || response.RAGTrace.ClaimedCount != 1 || response.RAGTrace.VerifiedClaimCount != 0 {
		t.Fatalf("invalid evidence was not refused: %#v", response)
	}
	if got := response.Messages[len(response.Messages)-1].Content; got != response.Answer {
		t.Fatalf("persisted answer %q differs from response %q", got, response.Answer)
	}
	if response.Tokens.RequestTokens != usage.InputTokens || response.Tokens.ResponseTokens != usage.OutputTokens || len(response.RequestMessages) == 0 {
		t.Fatalf("completion accounting/request missing: %#v", response.Tokens)
	}
}

type fakeCompleter struct {
	requests     [][]models.ChatMessage
	models       []string
	settings     []models.GenerationSettings
	answer       string
	finishReason string
	err          error
	usage        models.ModelUsage
}

type structuredFakeCompleter struct {
	*fakeCompleter
	calls int
}

func (f *structuredFakeCompleter) CompleteMessagesModelJSON(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	f.calls++
	return f.fakeCompleter.CompleteMessagesModel(ctx, model, messages, settings)
}

func (f *fakeCompleter) CompleteMessagesModel(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	f.models = append(f.models, model)
	return f.CompleteMessages(ctx, messages, settings)
}

func TestPersistentAgentRestoresHistoryAfterRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "state", "agent-history.json"))
	firstClient := &fakeCompleter{answer: "Запомнил"}
	firstAgent, err := NewPersistent(firstClient, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := firstAgent.Respond(context.Background(), "session-a", "Мой любимый цвет — зелёный"); err != nil {
		t.Fatal(err)
	}

	secondClient := &fakeCompleter{answer: "Продолжаю"}
	secondAgent, err := NewPersistent(secondClient, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := secondAgent.Respond(context.Background(), "session-a", "Какой мой любимый цвет?"); err != nil {
		t.Fatal(err)
	}
	if got, want := secondClient.requests[0][len(secondClient.requests[0])-3:], []models.ChatMessage{
		{Role: "user", Content: "Мой любимый цвет — зелёный"},
		{Role: "assistant", Content: "Запомнил"},
		{Role: "user", Content: "Какой мой любимый цвет?"},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("restored request = %#v, want %#v", got, want)
	}
}

func TestModelSwitchResetsOnlyOutputLimitAndManualLimitPersists(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "state", "agent-settings.json"))
	agent, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	assertSettings := func(model string, maxTokens int, temperature float64) {
		t.Helper()
		state, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: model})
		if err != nil {
			t.Fatal(err)
		}
		if state.Settings.MaxTokens != maxTokens {
			t.Fatalf("model %q maxTokens=%d, want %d", model, state.Settings.MaxTokens, maxTokens)
		}
		if state.Settings.Temperature == nil || *state.Settings.Temperature != temperature {
			t.Fatalf("model %q temperature=%v, want %v", model, state.Settings.Temperature, temperature)
		}
	}
	initial, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: models.DeepSeekFlashModel})
	if err != nil {
		t.Fatal(err)
	}
	if initial.Settings.MaxTokens != 512 {
		t.Fatalf("initial legacy maxTokens=%d, want 512", initial.Settings.MaxTokens)
	}
	temperature := 0.35
	manual := models.GenerationSettings{Temperature: &temperature, MaxTokens: 3000}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_generation_settings", Settings: manual}); err != nil {
		t.Fatal(err)
	}
	assertSettings(models.DeepSeekFlashModel, 3000, temperature) // same-model selection keeps manual value
	localModel := "ollama/qwen3:4b"
	assertSettings(localModel, 1024, temperature)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_generation_settings", Settings: manual}); err != nil {
		t.Fatal(err)
	}
	current, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state, err := current.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: localModel})
	if err != nil {
		t.Fatal(err)
	}
	if state.Settings.MaxTokens != 3000 {
		t.Fatalf("same-model reload lost manual maxTokens: %d", state.Settings.MaxTokens)
	}
	agent = current
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: models.DeepSeekProModel}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: localModel}); err != nil {
		t.Fatal(err)
	}
	state, err = agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: localModel})
	if err != nil {
		t.Fatal(err)
	}
	if state.Settings.MaxTokens != 1024 {
		t.Fatalf("switching back to local maxTokens=%d, want 1024", state.Settings.MaxTokens)
	}
}

type scriptedToolClient struct {
	completions []models.ModelCompletion
	requests    [][]models.ChatMessage
	tools       [][]models.ToolDefinition
	settings    []models.GenerationSettings
}

func (f *scriptedToolClient) CompleteMessages(_ context.Context, _ []models.ChatMessage, _ models.GenerationSettings) (models.ModelCompletion, error) {
	return models.ModelCompletion{}, errors.New("unexpected completion without tools")
}

func (f *scriptedToolClient) CompleteMessagesModelWithTools(_ context.Context, _ string, messages []models.ChatMessage, settings models.GenerationSettings, tools []models.ToolDefinition) (models.ModelCompletion, error) {
	f.requests = append(f.requests, copyMessages(messages))
	f.tools = append(f.tools, append([]models.ToolDefinition(nil), tools...))
	f.settings = append(f.settings, settings)
	if len(f.completions) == 0 {
		return models.ModelCompletion{}, errors.New("no scripted completion")
	}
	result := f.completions[0]
	f.completions = f.completions[1:]
	return result, nil
}

type fakeToolRuntime struct {
	calls []struct {
		name string
		args map[string]any
	}
}

func (f *fakeToolRuntime) ToolsForModel(context.Context) ([]models.ToolDefinition, error) {
	return []models.ToolDefinition{
		{Type: "function", Function: models.ToolFunction{Name: "list_desktop_videos", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "upload_video_to_yandex", Parameters: map[string]any{"type": "object"}}},
	}, nil
}

func (f *fakeToolRuntime) CallForModel(_ context.Context, name string, args map[string]any) (string, bool, error) {
	copied := make(map[string]any, len(args))
	for key, value := range args {
		copied[key] = value
	}
	f.calls = append(f.calls, struct {
		name string
		args map[string]any
	}{name: name, args: copied})
	if name == "list_desktop_videos" {
		return `{"videos":[{"name":"demo.mov","sizeBytes":100}]}`, false, nil
	}
	return `{"diskPath":"disk:/AI Challenge/lession 16/demo.mov","sizeBytes":100}`, false, nil
}

func TestAgentUsesMCPAndUploadsWithoutConfirmation(t *testing.T) {
	client := &scriptedToolClient{completions: []models.ModelCompletion{
		{ToolCalls: []models.ToolCall{{ID: "upload-1", Type: "function", Function: models.ToolCallFunction{Name: "upload_video_to_yandex", Arguments: `{"videoName":"demo.mov","lessonFolder":"lession 16"}`}}}},
	}}
	runtime := &fakeToolRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)

	first, err := a.Respond(context.Background(), "tool-session", "Загрузи последнее видео в папку lession 16")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(first.Answer, "действительно загружено") {
		t.Fatalf("answer = %q", first.Answer)
	}
	if len(runtime.calls) != 2 || runtime.calls[0].name != "list_desktop_videos" || runtime.calls[1].name != "upload_video_to_yandex" {
		t.Fatalf("calls = %#v", runtime.calls)
	}
	if len(client.tools) == 0 || len(client.tools[0]) != 3 || client.tools[0][2].Function.Name != "save_current_report_to_github" {
		t.Fatalf("tools passed to model = %#v", client.tools)
	}
}

func TestRAGToolTurnParsesGroundedEnvelopeAndKeepsMCPProvenance(t *testing.T) {
	base, _, chunkID, quote := newScenarioAgent(t, nil, false)
	client := &scriptedToolClient{completions: []models.ModelCompletion{{
		Answer: fmt.Sprintf(`{"answer":"Найден файл demo.mov.","claims":[{"text":%q,"evidence":[{"chunkId":%q,"quote":%q}]}]}`, quote, chunkID, quote),
	}}}
	base.client = client
	runtime := &fakeToolRuntime{}
	base.SetToolRuntime(runtime)

	response, err := base.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", "Покажи список видео на рабочем столе", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(response.Answer, "Найден файл demo.mov.") || strings.Contains(response.Answer, `{"answer"`) {
		t.Fatalf("visible response did not extract envelope answer: %q", response.Answer)
	}
	if !strings.Contains(response.Answer, "Факты из документов") || !strings.Contains(response.Answer, "Источники MCP") {
		t.Fatalf("document and MCP provenance were not both rendered: %q", response.Answer)
	}
	if len(response.RAGSources) != 2 || !strings.HasPrefix(response.RAGSources[0].Source, "guide.md") || response.RAGSources[1].Source != "MCP:list_desktop_videos" || response.RAGSources[1].Quote == "" {
		t.Fatalf("sources should include validated document and MCP evidence: %#v", response.RAGSources)
	}
	if len(runtime.calls) != 1 || runtime.calls[0].name != "list_desktop_videos" {
		t.Fatalf("MCP calls = %#v", runtime.calls)
	}
}

type trustedWeatherToolRuntime struct{}

func (trustedWeatherToolRuntime) ToolsForModel(context.Context) ([]models.ToolDefinition, error) {
	return []models.ToolDefinition{
		{Type: "function", Function: models.ToolFunction{Name: "weather_history", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "stop_weather_scheduler", Parameters: map[string]any{"type": "object"}}},
	}, nil
}

func (trustedWeatherToolRuntime) CallForModel(_ context.Context, name string, _ map[string]any) (string, bool, error) {
	if name == "weather_history" {
		return `{"measurementCount":1,"observations":[{"collectedAt":"2026-10-05T12:00:00Z","condition":"ясно","temperatureC":12,"apparentTemperatureC":11,"humidityPercent":50,"precipitationMM":0,"windSpeedKMH":3}]}`, false, nil
	}
	return `{"stopped":true}`, false, nil
}

func TestRAGTrustedServerToolSuccessKeepsWeatherAcknowledgments(t *testing.T) {
	cases := []struct {
		name, input, tool, expected string
	}{
		{"weather-history", "Покажи историю погоды", "weather_history", "Все сохранённые измерения погоды"},
		{"stop-weather-scheduler", "Останови сбор погоды", "stop_weather_scheduler", "Фоновый сбор погоды остановлен"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			base, _, _, _ := newScenarioAgent(t, nil, false)
			client := &scriptedToolClient{completions: []models.ModelCompletion{{ToolCalls: []models.ToolCall{{
				ID: "weather-1", Type: "function", Function: models.ToolCallFunction{Name: test.tool, Arguments: `{}`},
			}}}}}
			base.client = client
			base.SetToolRuntime(trustedWeatherToolRuntime{})
			response, err := base.RespondWithUserOptionsRAGConfigured(context.Background(), "u", "s", test.input, 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(response.Answer, test.expected) || !strings.Contains(response.Answer, "Источники MCP") || strings.Contains(response.Answer, "проверку формата") {
				t.Fatalf("trusted server acknowledgment was rejected: %q", response.Answer)
			}
			if len(response.RAGSources) != 1 || response.RAGSources[0].Source != "MCP:"+test.tool || response.RAGTrace == nil {
				t.Fatalf("MCP source/trace missing: %#v", response)
			}
			if len(response.ToolExecutions) != 1 || response.ToolExecutions[0].IsError {
				t.Fatalf("expected successful execution: %#v", response.ToolExecutions)
			}
		})
	}
}

type newsPipelineRuntime struct {
	calls []string
	args  []map[string]any
}

func (r *newsPipelineRuntime) ToolsForModel(context.Context) ([]models.ToolDefinition, error) {
	return []models.ToolDefinition{
		{Type: "function", Function: models.ToolFunction{Name: "search_news", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "github_put_file", Parameters: map[string]any{"type": "object"}}},
	}, nil
}

func (r *newsPipelineRuntime) CallForModel(_ context.Context, name string, args map[string]any) (string, bool, error) {
	r.calls = append(r.calls, name)
	copyArgs := make(map[string]any, len(args))
	for key, value := range args {
		copyArgs[key] = value
	}
	r.args = append(r.args, copyArgs)
	if name == "search_news" {
		return `{"articles":[{"title":"Новость","url":"https://example.test/news"}]}`, false, nil
	}
	return `{"path":"reports/news.md","branch":"main","commitSha":"0123456789abcdef0123456789abcdef01234567","url":"https://github.com/acme/demo/blob/main/reports/news.md"}`, false, nil
}

func TestFinishNewsDigestSummarizesThenSavesWithServerArguments(t *testing.T) {
	client := &fakeCompleter{answer: "# Новости Москвы\n\n## Ключевые события\n\n- Новость из источника."}
	runtime := &newsPipelineRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)
	deadline := time.Date(2026, time.September, 26, 23, 20, 0, 0, moscowLocation())
	a.finishNewsDigest("digest-session", "digest-user", newsDigestSchedule{City: "Москва", Deadline: deadline}, []string{`{"articles":[{"title":"Новость","url":"https://example.test/news"}]}`})

	if !reflect.DeepEqual(runtime.calls, []string{"github_put_file"}) {
		t.Fatalf("calls=%#v", runtime.calls)
	}
	if got, want := runtime.args[0]["path"], "reports/москва-2026-09-26.md"; got != want {
		t.Fatalf("path=%q, want %q", got, want)
	}
	if got := runtime.args[0]["content"]; !strings.Contains(got.(string), "Ключевые события") {
		t.Fatalf("content=%q", got)
	}
	conversation := a.sessions["digest-session"]
	if conversation == nil || !strings.Contains(conversation.messages[len(conversation.messages)-1].Content, "сводка сохранена") {
		t.Fatalf("messages=%#v", conversation.messages)
	}
}

func TestAgentRunsNewsSearchThenWritesDigestWithoutConfirmation(t *testing.T) {
	client := &scriptedToolClient{completions: []models.ModelCompletion{
		{ToolCalls: []models.ToolCall{{ID: "search", Type: "function", Function: models.ToolCallFunction{Name: "search_news", Arguments: `{"city":"Москва","from":"2026-09-01","to":"2026-09-02"}`}}}},
		{ToolCalls: []models.ToolCall{{ID: "save", Type: "function", Function: models.ToolCallFunction{Name: "github_put_file", Arguments: `{"path":"reports/news.md","content":"# Сводка","message":"docs: add news digest"}`}}}},
	}}
	runtime := &newsPipelineRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)
	response, err := a.Respond(context.Background(), "news-session", "Собери новостную сводку Москвы за 1 сентября")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(runtime.calls, []string{"search_news", "github_put_file"}) {
		t.Fatalf("calls=%#v", runtime.calls)
	}
	if !strings.Contains(response.Answer, "0123456789abcdef0123456789abcdef01234567") {
		t.Fatalf("answer=%q", response.Answer)
	}
	if len(response.Messages) < 4 || !strings.Contains(response.Messages[len(response.Messages)-2].Content, "github_put_file") {
		t.Fatalf("messages=%#v", response.Messages)
	}
}

type orchestrationRuntime struct {
	calls []string
	args  []map[string]any
}

func (r *orchestrationRuntime) ToolsForModel(context.Context) ([]models.ToolDefinition, error) {
	return []models.ToolDefinition{
		{Type: "function", Function: models.ToolFunction{Name: "collect_weather", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "weather_latest", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "get_city_weather", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "search_news", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: models.ToolFunction{Name: "github_put_file", Parameters: map[string]any{"type": "object"}}},
	}, nil
}

func (r *orchestrationRuntime) CallForModel(_ context.Context, name string, args map[string]any) (string, bool, error) {
	r.calls = append(r.calls, name)
	copyArgs := make(map[string]any, len(args))
	for key, value := range args {
		copyArgs[key] = value
	}
	r.args = append(r.args, copyArgs)
	switch name {
	case "collect_weather":
		return `{"collectedAt":"2026-09-27T10:00:00+03:00","temperatureC":12}`, false, nil
	case "weather_latest":
		return `{"collectedAt":"2026-09-27T10:00:00+03:00","condition":"облачно","temperatureC":12}`, false, nil
	case "get_city_weather":
		return `{"location":"Вологда","observation":{"condition":"ясно","temperatureC":9}}`, false, nil
	case "search_news":
		return `{"articles":[{"title":"Городская новость","url":"https://example.test/news"}]}`, false, nil
	default:
		return `{"path":"reports/moscow-briefing-2026-09-27.md","branch":"main","commitSha":"89abcdef0123456789abcdef0123456789abcdef","url":"https://github.com/acme/demo/blob/main/reports/moscow-briefing-2026-09-27.md"}`, false, nil
	}
}

func TestAgentOrchestratesWeatherNewsAndGitHubReport(t *testing.T) {
	client := &scriptedToolClient{completions: []models.ModelCompletion{
		{ToolCalls: []models.ToolCall{{ID: "collect", Type: "function", Function: models.ToolCallFunction{Name: "collect_weather", Arguments: `{}`}}}},
		{ToolCalls: []models.ToolCall{{ID: "latest", Type: "function", Function: models.ToolCallFunction{Name: "weather_latest", Arguments: `{}`}}}},
		{ToolCalls: []models.ToolCall{{ID: "news", Type: "function", Function: models.ToolCallFunction{Name: "search_news", Arguments: `{"city":"Москва","query":"","from":"2026-09-27","to":"2026-09-27"}`}}}},
		{ToolCalls: []models.ToolCall{{ID: "save", Type: "function", Function: models.ToolCallFunction{Name: "github_put_file", Arguments: `{"path":"reports/moscow-briefing-2026-09-27.md","content":"# Брифинг Москвы\n\n## Погода\n\nОблачно, 12 °C.\n\n## Ключевые события\n\n- Городская новость\n\n## Источники\n\n- https://example.test/news","message":"docs: add Moscow briefing 2026-09-27"}`}}}},
	}}
	runtime := &orchestrationRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)

	response, err := a.Respond(context.Background(), "orchestration-session", "Подготовь ежедневный брифинг по погоде и событиям Москвы и положи отчёт в репозиторий")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := runtime.calls, []string{"collect_weather", "weather_latest", "search_news", "github_put_file"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("calls=%#v, want=%#v", got, want)
	}
	if content := runtime.args[3]["content"].(string); !strings.Contains(content, "Погода") || !strings.Contains(content, "Городская новость") {
		t.Fatalf("report content=%q", content)
	}
	if !strings.Contains(response.Answer, "89abcdef0123456789abcdef0123456789abcdef") {
		t.Fatalf("answer=%q", response.Answer)
	}
}

func TestToolIntentRecognizesReportWordingVariants(t *testing.T) {
	for _, message := range []string{
		"Сделай отчёт о погоде и новостях Москвы, сохрани его в GitHub",
		"Нужен городской брифинг: температура, события за сегодня и файл в репозитории",
		"Подготовь ежедневную сводку по Москве и закоммить в гитхаб",
		"Собери погодные данные и важные события столицы в Markdown для репозитория",
	} {
		if !toolIntent(message) {
			t.Fatalf("toolIntent(%q) = false", message)
		}
	}
}

func TestEducationalRAGQuestionsDoNotTriggerMCPButExplicitMediaActionsDo(t *testing.T) {
	for _, message := range []string{
		"Цель: объяснить архитектуру RAG и памяти задачи для учебного видео.\nТермин: память задачи = отдельно сохранённые цель и ограничения.\nКакие этапы проходит новый вопрос в обычном чате с включённым RAG?",
		"Какие метаданные позволяют показать источник найденного фрагмента?",
		"Подготовь итоговое объяснение для моего видео: путь вопроса, поиск, память, проверка цитат и сохранение истории.",
	} {
		if toolIntent(message) {
			t.Errorf("educational RAG question triggered MCP: %q", message)
		}
	}
	for _, message := range []string{"Загрузи видео demo.mov на Яндекс Диск", "Сколько длится видео на рабочем столе?", "Какая сейчас погода в Москве?"} {
		if !toolIntent(message) {
			t.Errorf("explicit tool request was missed: %q", message)
		}
	}
}

func TestEducationalRAGQuestionsWithMCPRuntimeUseRegularCompletion(t *testing.T) {
	client := &fakeCompleter{answer: "Объясняю по документации проекта."}
	runtime := &newsPipelineRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)
	questions := []string{
		"Цель: объяснить архитектуру RAG для учебного видео.\nТермин: память задачи = сведения пользователя.\nКакие этапы проходит вопрос?",
		"Какие метаданные показывают источник фрагмента?",
		"Подготовь итоговое объяснение для моего видео: поиск, память, цитаты и сохранение истории.",
	}
	for i, question := range questions {
		if _, err := a.Respond(context.Background(), "s", question, 4); err != nil {
			t.Fatalf("question %d: %v", i+1, err)
		}
	}
	if len(client.requests) != len(questions) || len(runtime.calls) != 0 {
		t.Fatalf("educational requests used MCP unexpectedly: model=%d MCP=%#v", len(client.requests), runtime.calls)
	}
}

func TestBriefingIntentIncludesPlainWeatherAndNewsQuestion(t *testing.T) {
	if !briefingIntent("Напиши погоду в Вологде и новости в Новой Зеландии") {
		t.Fatal("weather-and-news request must reserve the briefing response budget")
	}
}

func TestGitHubSaveIsOnlyStagedForExplicitRequestAndAcceptsShortAgreement(t *testing.T) {
	if shouldStageGitHubReport("Напиши погоду и новости в Лондоне", nil, "# Лондон") {
		t.Fatal("plain weather-and-news request must not create a pending GitHub report")
	}
	if !shouldStageGitHubReport("Покажи брифинг с погодой и новостями, я подтвержу и ты оформишь его в GitHub", nil, "# Лондон") {
		t.Fatal("deferred explicit GitHub request must create a pending report")
	}
	for _, message := range []string{"да", "ок", "окей", "подтверждаю", "хорошо", "сохраняй"} {
		if !reportSaveConfirmed(message) {
			t.Fatalf("%q must confirm a pending report", message)
		}
	}
}

func TestSaveExistingReportIntentFindsLatestMarkdownReport(t *testing.T) {
	if !saveExistingReportIntent("Сохрани эту сводку как отчёт в GitHub") {
		t.Fatal("explicit request to save the shown report must be recognized")
	}
	if saveExistingReportIntent("Напиши погоду в Лондоне") {
		t.Fatal("ordinary weather request must not save anything")
	}
	messages := []models.ChatMessage{
		{Role: "assistant", Content: "Этап MCP — get_city_weather (готово)"},
		{Role: "assistant", Content: "# Погода в Лондоне\n\n22 °C"},
	}
	if got := latestReportContent(messages); got != "# Погода в Лондоне\n\n22 °C" {
		t.Fatalf("latest report=%q", got)
	}
}

func TestAgentSavesPreviousReportViaVirtualToolForEnglishRequest(t *testing.T) {
	client := &scriptedToolClient{completions: []models.ModelCompletion{
		{ToolCalls: []models.ToolCall{{ID: "save", Type: "function", Function: models.ToolCallFunction{Name: "save_current_report_to_github", Arguments: `{}`}}}},
	}}
	runtime := &orchestrationRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)
	conversation := a.conversationForUser("save-english", "save-english")
	conversation.setActiveMessagesLocked([]models.ChatMessage{{Role: "assistant", Content: "# Погода в Лондоне\n\n22 °C"}})

	response, err := a.Respond(context.Background(), "save-english", "save to GitHub")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := runtime.calls, []string{"github_put_file"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("calls=%#v, want=%#v", got, want)
	}
	if got := runtime.args[0]["content"]; got != "# Погода в Лондоне\n\n22 °C" {
		t.Fatalf("saved content=%q", got)
	}
	if !strings.Contains(response.Answer, "89abcdef0123456789abcdef0123456789abcdef") {
		t.Fatalf("answer=%q", response.Answer)
	}
}

func TestFormatToolExecutionCompactsNewsURLs(t *testing.T) {
	result := `{"articles":[{"title":"Первая новость","source":"Источник 1","url":"https://example.test/` + strings.Repeat("x", 1000) + `"},{"title":"Вторая новость","source":"Источник 2"}]}`
	text := formatToolExecutionForChat(models.ToolExecution{Name: "search_news", Arguments: map[string]any{"city": "Испания"}, Result: result})
	if strings.Contains(text, "https://") || !strings.Contains(text, "Получено новостей: 2") || !strings.Contains(text, "Первая новость") {
		t.Fatalf("formatted tool result=%q", text)
	}
}

func TestAgentShowsDeferredCityBriefingWithoutGitHubError(t *testing.T) {
	client := &scriptedToolClient{completions: []models.ModelCompletion{
		{ToolCalls: []models.ToolCall{{ID: "weather", Type: "function", Function: models.ToolCallFunction{Name: "get_city_weather", Arguments: `{"city":"Вологда"}`}}}},
		{ToolCalls: []models.ToolCall{{ID: "news", Type: "function", Function: models.ToolCallFunction{Name: "search_news", Arguments: `{"city":"Россия","query":"","from":"2026-09-27","to":"2026-09-27"}`}}}},
		{Answer: "# Брифинг\n\n## Погода\n\nВологда: ясно, 9 °C.\n\nПодтвердите сохранение в GitHub."},
	}}
	runtime := &orchestrationRuntime{}
	a := New(client)
	a.SetToolRuntime(runtime)

	response, err := a.Respond(context.Background(), "deferred-briefing", "Мне нужен брифинг: погода в Вологде и события за сегодня в России. Выведи сюда, я ознакомлюсь и подтвержу сохранение в GitHub")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := runtime.calls, []string{"get_city_weather", "search_news"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("calls=%#v, want=%#v", got, want)
	}
	if !strings.Contains(response.Answer, "Подтвердите сохранение") || strings.Contains(response.Answer, "не выполнена") {
		t.Fatalf("answer=%q", response.Answer)
	}
	if len(client.settings) != 3 || client.settings[0].MaxTokens != briefingMaxTokens {
		t.Fatalf("briefing settings=%#v", client.settings)
	}
	if len(client.requests) == 0 || !strings.Contains(client.requests[0][1].Content, "Сегодня в часовом поясе Москвы:") {
		t.Fatalf("tool context must include the current date: %#v", client.requests)
	}
	client.completions = append(client.completions, models.ModelCompletion{ToolCalls: []models.ToolCall{{ID: "save", Type: "function", Function: models.ToolCallFunction{Name: "github_put_file", Arguments: `{"path":"reports/vologda-briefing-2026-09-27.md","content":"# Брифинг\n\n## Погода\n\nВологда: ясно, 9 °C.","message":"docs: add city briefing 2026-09-27"}`}}}})
	confirmed, err := a.Respond(context.Background(), "deferred-briefing", "Подтверждаю сохранение в GitHub")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := runtime.calls, []string{"get_city_weather", "search_news", "github_put_file"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("calls after confirmation=%#v, want=%#v", got, want)
	}
	if len(client.requests) != 3 {
		t.Fatalf("confirmation must save server-side without a fourth model call; requests=%d", len(client.requests))
	}
	if content, _ := runtime.args[2]["content"].(string); !strings.Contains(content, "# Брифинг") {
		t.Fatalf("saved report content=%q", content)
	}
	if !strings.Contains(confirmed.Answer, "89abcdef0123456789abcdef0123456789abcdef") {
		t.Fatalf("confirmed answer=%q", confirmed.Answer)
	}
}

func TestToolIntentIncludesReadOnlyDesktopQuestions(t *testing.T) {
	for _, message := range []string{
		"Какие видео есть на рабочем столе?",
		"Посмотри записи на Desktop",
		"Покажи, что лежит на рабочем столе",
		"Проанализируй последнее видео",
		"Какое разрешение и FPS у записи?",
		"Поместится ли видео на Яндекс Диск?",
	} {
		if !toolIntent(message) {
			t.Fatalf("toolIntent(%q) = false", message)
		}
	}
	if toolIntent("Расскажи, что такое рабочий стол операционной системы") {
		t.Fatal("generic desktop question must not activate MCP")
	}
}

func TestParseNewsDigestScheduleUnderstandsClockTimeAndWords(t *testing.T) {
	schedule, ok := parseNewsDigestSchedule("Собери сводку новостей к 22:30 с периодичностью раз в пять минут в Москве", time.Date(2026, 9, 26, 21, 0, 0, 0, moscowLocation()))
	if !ok || schedule.City != "Москва" || schedule.Interval != 5*time.Minute || schedule.Deadline.Hour() != 22 || schedule.Deadline.Minute() != 30 {
		t.Fatalf("schedule=%#v ok=%v", schedule, ok)
	}
}

func TestToolFollowupIntentUsesPreviousVideoContext(t *testing.T) {
	previous := []models.ChatMessage{{Role: "assistant", Content: "Нашёл видео test_name.mov."}}
	if !toolFollowupIntent("Сколько места оно займёт?", previous) {
		t.Fatal("video follow-up must activate MCP")
	}
	if toolFollowupIntent("Расскажи подробнее", previous) {
		t.Fatal("generic follow-up must not activate MCP")
	}
	retryContext := []models.ChatMessage{{Role: "assistant", Content: "Загрузить видео на Яндекс Диск не удалось: папка отсутствует."}}
	if !toolFollowupIntent("Папку создал, попробуй ещё раз", retryContext) {
		t.Fatal("retry after creating folder must activate MCP")
	}
	if !explicitUploadConfirmation("Папку создал, попробуй ещё раз") {
		t.Fatal("explicit retry must authorize the repeated upload")
	}
}

func TestToolFollowupIntentUsesWeatherContextForEveryMeasurement(t *testing.T) {
	previous := []models.ChatMessage{{Role: "assistant", Content: "Сводка погоды Москвы: измерений 22, температура 9 °C."}}
	if !toolFollowupIntent("Нет, напиши каждое измерение, которое ты измерил", previous) {
		t.Fatal("weather follow-up must activate MCP tools")
	}
	if !toolFollowupIntent("да", previous) {
		t.Fatal("weather acknowledgement must activate MCP tools")
	}
}

func TestToolFollowupNeedsRealWeatherOrVideoAssetContext(t *testing.T) {
	if toolFollowupIntent("Какие метаданные у найденного фрагмента?", []models.ChatMessage{{Role: "assistant", Content: "Выбранный планировщик учитывается при сборе RAG-контекста"}}) {
		t.Fatal("planner/RAG wording must not keep MCP intent alive")
	}
	if toolFollowupIntent("Какие метаданные у найденного фрагмента?", []models.ChatMessage{{Role: "assistant", Content: "Для учебного видео RAG ищет по метаданным документа."}}) {
		t.Fatal("educational video wording must not look like a video asset")
	}
	if !toolFollowupIntent("Нет, напиши каждое измерение", []models.ChatMessage{{Role: "assistant", Content: "Вызов weather_summary: Сводка по Москве."}}) {
		t.Fatal("real weather tool response must keep weather follow-ups active")
	}
}

func TestWeatherHistoryClearUsesConversationState(t *testing.T) {
	if !weatherClearRequest("удали историю погодных измерений") {
		t.Fatal("weather clear request must start a confirmation flow")
	}
	if weatherClearRequest("подтверждаю") {
		t.Fatal("acknowledgement must not be tied to a hardcoded phrase")
	}
	if !weatherHistoryWasCleared([]models.ToolExecution{{Name: "clear_weather_history"}}) {
		t.Fatal("successful weather clear must close the confirmation flow")
	}
	if explicitGitHubConfirmation("да") {
		t.Fatal("plain confirmation must not be treated as a GitHub confirmation")
	}
	if explicitUploadConfirmation("да") {
		t.Fatal("plain confirmation must not be treated as an upload confirmation")
	}
}

func TestFormatWeatherHistoryForChatUsesPlainTextRowsAndMoscowTime(t *testing.T) {
	formatted := formatWeatherHistoryForChat(`{"measurementCount":2,"observations":[{"collectedAt":"2026-09-26T06:20:12Z","condition":"Sunny","temperatureC":9,"apparentTemperatureC":7,"humidityPercent":79,"precipitationMM":0,"windSpeedKMH":9},{"collectedAt":"2026-09-26T06:21:12Z","condition":"Sunny","temperatureC":9,"apparentTemperatureC":7,"humidityPercent":79,"precipitationMM":0,"windSpeedKMH":9}]}`)
	if !strings.Contains(formatted, "Все сохранённые измерения погоды Москвы: 2.") || !strings.Contains(formatted, "1. 26.09.26, 09:20") || !strings.Contains(formatted, "2. 26.09.26, 09:21") {
		t.Fatalf("formatted history = %q", formatted)
	}
}

func TestGroundedUploadAnswerNeverClaimsSuccessWithoutToolResult(t *testing.T) {
	answer := groundedUploadAnswer("Файл загружен, ожидайте.", nil, true)
	if strings.Contains(answer, "ожидайте") || !strings.Contains(answer, "не выполнена") {
		t.Fatalf("answer = %q", answer)
	}
	success := groundedUploadAnswer("модельный текст", []models.ToolExecution{{
		Name: "upload_video_to_yandex", Result: `{"diskPath":"disk:/AI Challenge/lession 16/demo.mov","sizeBytes":100}`,
	}}, true)
	if !strings.Contains(success, "действительно загружено") || !strings.Contains(success, "lession 16") {
		t.Fatalf("success = %q", success)
	}
}

func TestClearRemovesPersistentSession(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	first, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.Respond(context.Background(), "session-a", "Сообщение"); err != nil {
		t.Fatal(err)
	}
	if err := first.Clear("session-a"); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if got := second.History("session-a"); len(got) != 0 {
		t.Fatalf("history after clear = %#v, want empty", got)
	}
}

func TestTaskStateMachinePausesAndResumesFromSavedStep(t *testing.T) {
	client := &fakeCompleter{answer: "Продолжаю работу"}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "enabled"}); err != nil {
		t.Fatal(err)
	}
	configured, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{
		Goal:           "Подготовить запуск",
		Phases:         []string{"planning", "execution", "validation", "done"},
		CurrentStep:    "Собрать требования",
		ExpectedAction: "Согласовать приоритеты",
	}})
	if err != nil {
		t.Fatal(err)
	}
	if configured.Task.Phase != "planning" || configured.Task.Status != models.TaskActive || configured.Task.PhaseIndex != 0 {
		t.Fatalf("configured task = %#v", configured.Task)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "pause_task"}); err != nil {
		t.Fatal(err)
	}
	paused, err := agent.Respond(context.Background(), "session", "Что делать дальше?")
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 0 || !strings.Contains(paused.Answer, "не было отправлено модели") {
		t.Fatalf("paused response = %#v, calls=%d", paused, len(client.requests))
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "resume_task"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Продолжаем"); err != nil {
		t.Fatal(err)
	}
	resumedContext := client.requests[0][1].Content
	if !strings.Contains(resumedContext, "Статус: active") || !strings.Contains(resumedContext, "Не повторяй прежнее объяснение") {
		t.Fatalf("resumed task context = %q", resumedContext)
	}
}

func TestPausedTaskDoesNotCallModelUntilResumed(t *testing.T) {
	client := &fakeCompleter{answer: "Модельный ответ"}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Собрать MVP", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "pause_task"}); err != nil {
		t.Fatal(err)
	}
	paused, err := agent.Respond(context.Background(), "session", "Измени состав экранов")
	if err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 0 || !strings.Contains(paused.Answer, "не было отправлено модели") || len(paused.Messages) != 2 {
		t.Fatalf("paused response = %#v, calls=%d", paused, len(client.requests))
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "resume_task"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Измени состав экранов"); err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 1 {
		t.Fatalf("calls after resume = %d, want 1", len(client.requests))
	}
}

func TestProjectPlannerBuildsDashboardFromChatMessage(t *testing.T) {
	client := &fakeCompleter{answer: `{"answer":"Черновик ТЗ готов.","plan":{"specification":"MVP: клиентское приложение с каталогом, корзиной и оформлением заказа.","currentStep":"Согласовать состав экранов","expectedAction":"Подтвердить границы MVP","openQuestions":["Нужен ли поиск?"],"decisions":["Делаем только текстовое ТЗ"],"nextSteps":["Описать пользовательские сценарии"]}}`}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "enabled"}); err != nil {
		t.Fatal(err)
	}
	result, err := agent.Respond(context.Background(), "session", "Разработка MVP приложения\n\n- Цель: «Сделать MVP приложения доставки еды».\n- Этапы: `planning → execution → validation → done`.\n- Шаг: «Собрать список ключевых экранов».")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "Черновик ТЗ готов." || result.Task.Goal != "Сделать MVP приложения доставки еды" || result.Task.Phase != "planning" || result.Task.CurrentStep != "Согласовать состав экранов" || result.Task.Specification == "" || len(result.Task.OpenQuestions) != 1 || len(result.Task.Decisions) != 1 {
		t.Fatalf("planner result = %#v", result)
	}
	if len(client.requests) != 1 || !strings.Contains(client.requests[0][2].Content, "агент-проектировщик") {
		t.Fatalf("planner request = %#v", client.requests)
	}
}

func TestProjectPlannerSalvagesAnswerAndSpecificationFromTruncatedJSON(t *testing.T) {
	answer, plan := applyPlannerCompletion(models.TaskState{Goal: "MVP", CurrentStep: "Исходный шаг"}, `{"answer":"Черновик готов.","plan":{"specification":"Подробное ТЗ для MVP.","currentStep":"Согласовать экраны","expectedAction":"Подтвердить список","openQuestions":["Нужен поиск?"`)
	if answer != "Черновик готов." || plan.Specification != "Подробное ТЗ для MVP." || plan.CurrentStep != "Согласовать экраны" || plan.ExpectedAction != "Подтвердить список" {
		t.Fatalf("salvaged plan = %#v, answer=%q", plan, answer)
	}
}

func TestProjectPlannerStoresFinalArtifactSeparatelyWithoutCompletingTask(t *testing.T) {
	answer, plan := applyPlannerCompletion(models.TaskState{
		Goal:          "MVP доставки",
		Phases:        []string{"planning", "done"},
		PhaseIndex:    1,
		Phase:         "done",
		Status:        models.TaskActive,
		Specification: "Рабочий черновик: согласовать список экранов.",
	}, `{"answer":"Итоговое ТЗ сформировано.","plan":{"artifactTitle":"ТЗ MVP доставки еды","artifactContent":"# ТЗ MVP\n\n## Цель\nСобрать приложение доставки еды.\n\n## Критерии приёмки\nПользователь оформляет заказ."}}`)
	if answer != "Итоговое ТЗ сформировано." || plan.ArtifactTitle != "ТЗ MVP доставки еды" || !strings.Contains(plan.ArtifactContent, "Критерии приёмки") {
		t.Fatalf("artifact = %#v, answer=%q", plan, answer)
	}
	if plan.Specification != "Рабочий черновик: согласовать список экранов." {
		t.Fatalf("working specification was overwritten: %q", plan.Specification)
	}
	if plan.Status == models.TaskDone {
		t.Fatalf("model artifact must not complete task: %#v", plan)
	}
}

func TestProjectPlannerRecognizesSingleLineStarterMessage(t *testing.T) {
	task, ok := taskFromMessage("Разработка MVP приложения - Цель: «Сделать MVP приложения доставки еды». - Этапы: planning → execution → validation → done. - Шаг: «Собрать список ключевых экранов».")
	if !ok || task.Goal != "Сделать MVP приложения доставки еды" || !reflect.DeepEqual(task.Phases, []string{"planning", "execution", "validation", "done"}) || task.CurrentStep != "Собрать список ключевых экранов" {
		t.Fatalf("parsed task = %#v, ok=%v", task, ok)
	}
}

func TestPlanApprovalRefreshesDashboardQuestionsAndNextSteps(t *testing.T) {
	updated := updatePlannerProgress(models.TaskState{
		Goal:           "MVP",
		Phases:         []string{"planning", "execution", "validation", "done"},
		Phase:          "planning",
		PhaseIndex:     0,
		ExpectedAction: "Согласовать состав MVP",
		OpenQuestions:  []string{"Нужен поиск?", "Нужны акции?"},
	}, "Подтверждаю план, всё согласовано.")
	if updated.Phase != "execution" || updated.PhaseIndex != 1 || !updated.PlanApproved || len(updated.OpenQuestions) != 0 || len(updated.Decisions) != 1 || !strings.Contains(updated.CurrentStep, "execution") || !strings.Contains(updated.ExpectedAction, "реализацию") || len(updated.NextSteps) != 1 {
		t.Fatalf("updated dashboard = %#v", updated)
	}
}

func TestProjectPlannerCannotChangePhaseFromModelResponse(t *testing.T) {
	_, plan := applyPlannerCompletion(models.TaskState{Goal: "MVP", Phases: []string{"planning", "execution", "validation", "done"}, Phase: "planning"}, `{"answer":"Готово.","plan":{"phase":"validation"}}`)
	if plan.Phase != "planning" || plan.PhaseIndex != 0 {
		t.Fatalf("planner skipped a phase: %#v", plan)
	}
	_, plan = applyPlannerCompletion(plan, `{"answer":"Готово.","plan":{"phase":"execution"}}`)
	if plan.Phase != "planning" || plan.PhaseIndex != 0 {
		t.Fatalf("planner changed a phase: %#v", plan)
	}
	_, plan = applyPlannerCompletion(plan, `{"answer":"Возвращаю.","plan":{"phase":"planning"}}`, "Вернись в planning, нужно доработать требования.")
	if plan.Phase != "planning" || plan.PhaseIndex != 0 {
		t.Fatalf("planner changed a phase on return request: %#v", plan)
	}
}

func TestProjectDashboardRejectsDirectPhaseSwitch(t *testing.T) {
	agent := New(&fakeCompleter{})
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Текстовое ТЗ", Phases: []string{"planning", "execution", "validation", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "switch_task_phase", Phase: "execution"}); err == nil || !strings.Contains(err.Error(), "запрещено") {
		t.Fatalf("direct switch must be rejected, err=%v", err)
	}
}

func TestInvariantsAreSeparateFromDialogueAndIncludedAsMandatoryRules(t *testing.T) {
	client := &fakeCompleter{answer: "Отказываюсь от MySQL: это нарушает инвариант. Могу предложить PostgreSQL."}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Сделать API", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_invariant", Invariant: models.Invariant{Scope: "task", Rule: "Использовать только PostgreSQL."}}); err != nil {
		t.Fatal(err)
	}
	result, err := agent.Respond(context.Background(), "session", "Давай вместо этого используем MySQL")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Answer, "нарушает инвариант") || len(result.Task.TaskInvariants) != 1 {
		t.Fatalf("conflict result = %#v", result)
	}
	if len(client.requests) != 1 || !strings.Contains(client.requests[0][1].Content, "ОБЯЗАТЕЛЬНЫЕ ИНВАРИАНТЫ") || !strings.Contains(client.requests[0][1].Content, "PostgreSQL") || !strings.Contains(client.requests[0][1].Content, "откажись") {
		t.Fatalf("invariants were not supplied as mandatory context: %#v", client.requests)
	}
	for _, message := range result.Messages {
		if strings.Contains(message.Content, "ОБЯЗАТЕЛЬНЫЕ ИНВАРИАНТЫ") {
			t.Fatalf("invariant leaked into dialogue: %#v", result.Messages)
		}
	}
}

func TestStateInvariantBlocksAutomaticTransitionUntilApproval(t *testing.T) {
	_, task := applyPlannerCompletion(models.TaskState{
		Goal: "MVP", Phases: []string{"planning", "execution"}, Phase: "planning", ExpectedAction: "Подтвердить план",
		RequireApprovalForTransition: true,
	}, `{"answer":"Перехожу.","plan":{"phase":"execution"}}`, "Продолжай без подтверждения")
	if task.Phase != "planning" {
		t.Fatalf("transition without approval = %#v", task)
	}
	_, task = applyPlannerCompletion(task, `{"answer":"Перехожу.","plan":{"phase":"execution"}}`, "Подтверждаю план")
	if task.Phase != "planning" {
		t.Fatalf("model transition with approval = %#v", task)
	}
}

func TestPlannerCanBeDisabledWithoutDiscardingTaskState(t *testing.T) {
	client := &fakeCompleter{answer: "Обычный ответ, не JSON"}
	agent := New(client)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "MVP", Phases: []string{"planning", "done"}, CurrentStep: "Собрать требования"}}); err != nil {
		t.Fatal(err)
	}
	disabled, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "disabled"})
	if err != nil || disabled.PlannerMode != "disabled" || disabled.Task.CurrentStep != "Собрать требования" {
		t.Fatalf("disabled task = %#v, err=%v", disabled.Task, err)
	}
	result, err := agent.Respond(context.Background(), "session", "Расскажи про Android-разработку")
	if err != nil {
		t.Fatal(err)
	}
	if result.Answer != "Обычный ответ, не JSON" || result.PlannerMode != "disabled" {
		t.Fatalf("response = %#v", result)
	}
	for _, message := range client.requests[0] {
		if strings.Contains(message.Content, "агент-проектировщик") {
			t.Fatalf("planner prompt must be absent: %#v", client.requests[0])
		}
	}
}

func TestPlannerModeCanBeChangedBeforeAnyTaskExists(t *testing.T) {
	agent := New(&fakeCompleter{})
	state, err := agent.ApplyContextCommand("empty-session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "disabled"})
	if err != nil || state.PlannerMode != "disabled" || state.Task.Goal != "" {
		t.Fatalf("empty-chat planner mode = %#v, err=%v", state, err)
	}
}

func TestPlannerIsDisabledByDefaultForNewConversation(t *testing.T) {
	agent := New(&fakeCompleter{})
	state := agent.State("new-session")
	if state.PlannerMode != "disabled" {
		t.Fatalf("default planner mode = %q, want disabled", state.PlannerMode)
	}
}

func TestGlobalInvariantSurvivesNewSession(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "invariants.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommandForUser("user", "one", models.ContextCommand{Action: "save_invariant", Invariant: models.Invariant{Scope: "global", Rule: "Не транслитерировать английские термины."}}); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.StateForUser("user", "two")
	if len(state.GlobalInvariants) != 1 || !strings.Contains(state.GlobalInvariants[0].Rule, "транслитерировать") {
		t.Fatalf("global invariants = %#v", state.GlobalInvariants)
	}
}

func TestPauseInFlightInterruptsPlannerBeforeProviderWork(t *testing.T) {
	agent, client, _, _ := newScenarioAgent(t, nil, false)
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "set_planner_mode", PlannerMode: "enabled"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "ТЗ", Phases: []string{"planning", "done"}}}); err != nil {
		t.Fatal(err)
	}
	result := make(chan models.AgentResponse, 1)
	failure := make(chan error, 1)
	go func() {
		response, err := agent.RespondWithUserOptionsRAGConfigured(context.Background(), "session", "session", "Подготовь черновик ТЗ", 4, models.StrategySlidingWindow, models.DeepSeekFlashModel, true, models.RAGOptions{})
		if err != nil {
			failure <- err
			return
		}
		result <- response
	}()

	var interrupted bool
	deadline := time.Now().Add(300 * time.Millisecond)
	for time.Now().Before(deadline) {
		if response, ok := agent.PauseInFlight("session", "session"); ok {
			interrupted = response.Task.Status == models.TaskPaused
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !interrupted {
		t.Fatal("pause did not interrupt active planner work")
	}
	select {
	case err := <-failure:
		t.Fatal(err)
	case response := <-result:
		if response.Task.Status != models.TaskPaused || response.PendingMessage != "Подготовь черновик ТЗ" || !strings.Contains(response.Answer, "приостановлена") || !strings.Contains(response.Answer, "Источники:") || response.RAGTrace == nil || !response.RAGEnabled || !response.RAGModelCallSkipped || len(response.RAGSources) != 0 || len(client.requests) != 0 {
			t.Fatalf("pause response = %#v, calls=%d", response, len(client.requests))
		}
	case <-time.After(time.Second):
		t.Fatal("planner did not finish after pause")
	}
}

func TestTaskStateMachineOnlyMovesToAdjacentPhasesAndPersists(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "Проверить задачу", Phases: []string{"planning", "execution", "validation", "done"}}}); err != nil {
		t.Fatal(err)
	}
	advanced, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "approve_plan"})
	if err != nil || advanced.Task.Phase != "execution" || advanced.Task.PhaseIndex != 1 || advanced.Task.Status != models.TaskActive {
		t.Fatalf("advanced state = %#v, err=%v", advanced.Task, err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "complete_implementation"}); err != nil {
		t.Fatal(err)
	}
	completed, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "pass_validation"})
	if err != nil || completed.Task.Phase != "done" || completed.Task.Status != models.TaskDone {
		t.Fatalf("completed state = %#v, err=%v", completed.Task, err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil {
		t.Fatal("advance after final phase unexpectedly succeeded")
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if restored := second.State("session").Task; restored.Phase != "done" || restored.Status != models.TaskDone || !restored.PlanApproved || !restored.ImplementationCompleted || !restored.ValidationPassed || !reflect.DeepEqual(restored.Phases, []string{"planning", "execution", "validation", "done"}) {
		t.Fatalf("restored task = %#v", restored)
	}
}

func TestTaskLifecycleRejectsInvalidTransitionsAndKeepsState(t *testing.T) {
	agent := New(&fakeCompleter{})
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{
		Goal: "Проверить жизненный цикл", Phases: []string{"planning", "execution", "validation", "done"},
		// A caller must not be able to forge evidence in the configuration.
		PlanApproved: true, ValidationPassed: true,
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil || !strings.Contains(err.Error(), "утвердите план") {
		t.Fatalf("planning -> execution without approval must fail, err=%v", err)
	}
	if state := agent.State("session").Task; state.Phase != "planning" || state.PlanApproved || state.ValidationPassed {
		t.Fatalf("failed transition changed or accepted forged state: %#v", state)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "complete_implementation"}); err == nil || !strings.Contains(err.Error(), "Нельзя") {
		t.Fatalf("completion outside execution must fail, err=%v", err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "approve_plan"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil || !strings.Contains(err.Error(), "реализацию") {
		t.Fatalf("execution -> validation without completion must fail, err=%v", err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "complete_implementation"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "advance_task"}); err == nil || !strings.Contains(err.Error(), "валидацию") {
		t.Fatalf("validation -> done without validation must fail, err=%v", err)
	}
}

func TestReturningForReworkInvalidatesLaterLifecycleEvidence(t *testing.T) {
	agent := New(&fakeCompleter{})
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "configure_task", Task: models.TaskState{Goal: "ТЗ", Phases: []string{"planning", "execution", "validation", "done"}}}); err != nil {
		t.Fatal(err)
	}
	for _, action := range []string{"approve_plan", "complete_implementation", "pass_validation"} {
		if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: action}); err != nil {
			t.Fatalf("%s: %v", action, err)
		}
	}
	returned, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "previous_task"})
	if err != nil || returned.Task.Phase != "validation" || returned.Task.Status != models.TaskActive || returned.Task.ValidationPassed {
		t.Fatalf("return from done must reopen validation: %#v, err=%v", returned.Task, err)
	}
	returned, err = agent.ApplyContextCommand("session", models.ContextCommand{Action: "previous_task"})
	if err != nil || returned.Task.Phase != "execution" || returned.Task.ImplementationCompleted || returned.Task.ValidationPassed {
		t.Fatalf("return to execution must invalidate later proof: %#v, err=%v", returned.Task, err)
	}
}

func TestClearKeepsLongTermMemoryButRemovesDialogueAndWorkingMemory(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "agent-history.json"))
	agent, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Текущая задача", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryWorking, Value: "Текущая задача"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryLongTerm, Category: "knowledge", Value: "Пишет по-русски"}); err != nil {
		t.Fatal(err)
	}
	if err := agent.Clear("session"); err != nil {
		t.Fatal(err)
	}
	state := agent.State("session")
	if len(state.Messages) != 0 || len(state.Memory.ShortTerm) != 0 || len(state.Memory.Working) != 0 {
		t.Fatalf("cleared task state = %#v", state)
	}
	if got := state.Memory.LongTerm; len(got) != 1 || got[0].Value != "Пишет по-русски" {
		t.Fatalf("long-term memory after clear = %#v", got)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "clear_memory_layer", Layer: models.MemoryLongTerm}); err != nil {
		t.Fatal(err)
	}
	if got := agent.State("session").Memory.LongTerm; len(got) != 0 {
		t.Fatalf("long-term memory after separate clear = %#v", got)
	}
}

func (f *fakeCompleter) CompleteMessages(_ context.Context, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	f.requests = append(f.requests, append([]models.ChatMessage(nil), messages...))
	f.settings = append(f.settings, settings)
	if f.err != nil {
		return models.ModelCompletion{}, f.err
	}
	return models.ModelCompletion{Answer: f.answer, FinishReason: f.finishReason, Usage: f.usage}, nil
}

func TestRespondReportsExactProviderUsageAndCumulativeCost(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ", usage: models.ModelUsage{InputTokens: 120, OutputTokens: 30, CacheMissTokens: 120}}
	agent := New(client)
	result, err := agent.Respond(context.Background(), "session", "Короткий вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if result.Tokens.RequestTokens != 120 || result.Tokens.ResponseTokens != 30 || result.Tokens.CumulativeInputTokens != 120 || result.Tokens.CumulativeOutputTokens != 30 {
		t.Fatalf("tokens = %#v", result.Tokens)
	}
	if result.Tokens.EstimatedRequestTokens == 0 || result.Tokens.CurrentMessageTokens == 0 || result.Tokens.CumulativeCostUSD <= 0 {
		t.Fatalf("incomplete token report = %#v", result.Tokens)
	}
	if result.Tokens.CacheMissTokens != 120 || len(result.RequestMessages) != 3 || result.RequestMessages[len(result.RequestMessages)-1].Content != "Короткий вопрос" || !strings.Contains(result.RequestMessages[1].Content, "Актуальная память задачи") {
		t.Fatalf("request details = %#v, tokens = %#v", result.RequestMessages, result.Tokens)
	}
}

func TestRespondRejectsDialogueBeyondContextBeforeProviderCall(t *testing.T) {
	client := &fakeCompleter{answer: "Не должен вызываться"}
	agent := New(client)
	conversation := agent.conversation("session")
	conversation.strategy = models.StrategyBranching
	conversation.ensureRootBranchLocked()
	conversation.branches["root"].messages = make([]models.ChatMessage, 100)
	for i := range conversation.branches["root"].messages {
		conversation.branches["root"].messages[i] = models.ChatMessage{Role: "user", Content: strings.Repeat("я", maxMessageCharacters)}
	}
	if _, err := agent.Respond(context.Background(), "session", "Ещё один вопрос"); !errors.Is(err, ErrContextLimit) {
		t.Fatalf("Respond error = %v, want ErrContextLimit", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("provider requests = %d, want 0", len(client.requests))
	}
	if got := len(agent.History("session")); got != 100 {
		t.Fatalf("history length = %d, want 100; rejected message must not persist", got)
	}
}

func TestTokenReportDoesNotCountSystemInstructionAsDialogueHistory(t *testing.T) {
	agent := New(&fakeCompleter{})
	if err := agent.Clear("session"); err != nil {
		t.Fatal(err)
	}
	report := agent.TokenReport("session")
	if report.HistoryTokens != 0 || report.CurrentMessageTokens != 0 {
		t.Fatalf("cleared dialogue tokens = %#v, want zero", report)
	}
	if report.EstimatedRequestTokens == 0 {
		t.Fatal("system instruction must remain in full request estimate")
	}
}

func TestRespondBuildsHistoryPerSession(t *testing.T) {
	client := &fakeCompleter{answer: "Первый ответ"}
	agent := New(client)
	first, err := agent.Respond(context.Background(), "session-a", "Первый вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := first.Messages, []models.ChatMessage{{Role: "user", Content: "Первый вопрос"}, {Role: "assistant", Content: "Первый ответ"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("first history = %#v, want %#v", got, want)
	}

	client.answer = "Второй ответ"
	_, err = agent.Respond(context.Background(), "session-a", "Второй вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := client.requests[1][len(client.requests[1])-3:], []models.ChatMessage{
		{Role: "user", Content: "Первый вопрос"}, {Role: "assistant", Content: "Первый ответ"}, {Role: "user", Content: "Второй вопрос"},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second LLM request = %#v, want %#v", got, want)
	}

	client.answer = "Другой диалог"
	other, err := agent.Respond(context.Background(), "session-b", "Другой вопрос")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := other.Messages[len(other.Messages)-2:], []models.ChatMessage{{Role: "user", Content: "Другой вопрос"}, {Role: "assistant", Content: "Другой диалог"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("second session history = %#v, want %#v", got, want)
	}
}

func TestRespondDoesNotKeepFailedMessage(t *testing.T) {
	client := &fakeCompleter{err: errors.New("provider unavailable")}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Вопрос"); err == nil {
		t.Fatal("Respond succeeded, want provider error")
	}
	if got := agent.History("session"); len(got) != 0 {
		t.Fatalf("history after failure = %#v, want empty", got)
	}
}

func TestSlidingWindowKeepsOnlyRecentNWithoutSummaryCall(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Первый факт", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "session", "Второй факт", 2); err != nil {
		t.Fatal(err)
	}
	result, err := agent.Respond(context.Background(), "session", "Третий вопрос", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 6 || result.Messages[4].Content != "Третий вопрос" {
		t.Fatalf("archived history = %#v, want all six messages", result.Messages)
	}
	if len(client.requests) != 3 {
		t.Fatalf("calls = %d, want one provider request per user message", len(client.requests))
	}
	request := client.requests[2]
	if len(request) < 3 || request[len(request)-2].Role != "assistant" || request[len(request)-2].Content != "Ответ" || request[len(request)-1].Content != "Третий вопрос" {
		t.Fatalf("window request = %#v", request)
	}
}

func TestFactsSurviveWindowAndAreSentAsSeparateSystemBlock(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Цель: сделать приложение привычек", 2, models.StrategyFacts); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Ограничение: только iOS", 2, models.StrategyFacts); err != nil {
		t.Fatal(err)
	}
	result, err := agent.RespondWithStrategy(context.Background(), "session", "Что мы делаем?", 2, models.StrategyFacts)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Messages) != 6 || len(result.Facts) == 0 {
		t.Fatalf("state = %#v", result)
	}
	if !strings.Contains(client.requests[2][1].Content, "Исторические sticky facts") || !strings.Contains(client.requests[2][1].Content, "сделать приложение") {
		t.Fatalf("facts prompt = %#v", client.requests[2])
	}
}

func TestMemoryLayersAreSeparateAndOnlyExplicitCommandsPersistWorkingAndLongTerm(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.Respond(context.Background(), "session", "Обсуждаем экран заказа", 2); err != nil {
		t.Fatal(err)
	}
	if state := agent.State("session"); len(state.Memory.Working) != 0 || len(state.Memory.LongTerm) != 0 {
		t.Fatalf("ordinary dialogue must not auto-save explicit layers: %#v", state.Memory)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryWorking, Key: "task", Value: "Экран отслеживания заказа"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "language", Value: "Русский"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "decisions", Key: "map", Value: "Не показывать координаты курьера"}); err != nil {
		t.Fatal(err)
	}

	state := agent.State("session")
	if got := len(state.Memory.ShortTerm); got != 2 {
		t.Fatalf("short-term entries = %d, want dialogue only", got)
	}
	if got := state.Memory.Working; len(got) != 1 || got[0].Key != "task" {
		t.Fatalf("working memory = %#v", got)
	}
	if got := state.Memory.LongTerm; len(got) != 2 || got[0].Category != "decisions" || got[1].Category != "knowledge" {
		t.Fatalf("long-term memory = %#v", got)
	}

	if _, err := agent.Respond(context.Background(), "session", "Что учесть в ответе?", 2); err != nil {
		t.Fatal(err)
	}
	request := client.requests[len(client.requests)-1]
	if !strings.Contains(request[1].Content, "Долговременная память") || !strings.Contains(request[1].Content, "knowledge.language: Русский") {
		t.Fatalf("long-term block = %#v", request)
	}
	if !strings.Contains(request[2].Content, "Рабочая память") || !strings.Contains(request[2].Content, "task: Экран") {
		t.Fatalf("working block = %#v", request)
	}
	if len(request) < 6 || request[len(request)-2].Role != "assistant" || request[len(request)-1].Content != "Что учесть в ответе?" {
		t.Fatalf("short-term window in request = %#v", request)
	}
}

func TestMemoryLayersPersistAndShortTermCannotBeWrittenDirectly(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "memory.json"))
	first, err := NewPersistent(&fakeCompleter{answer: "Ответ"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryWorking, Key: "deadline", Value: "пятница"}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "api", Value: "Используем существующее API"}); err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryShortTerm, Key: "x", Value: "y"}); err == nil {
		t.Fatal("short-term memory write succeeded")
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.State("session")
	if len(state.Memory.ShortTerm) != 0 || len(state.Memory.Working) != 1 || len(state.Memory.LongTerm) != 1 {
		t.Fatalf("restored layers = %#v", state.Memory)
	}
}

func TestProfileIsSentForEveryRequestAndKeepsProfilesIndependent(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	anna := models.UserProfile{Name: "Анна", Style: "деловой", Format: "ровно 3 пункта", Constraints: "без англицизмов"}
	maxim := models.UserProfile{Name: "Максим", Style: "неформальный", Format: "подробный разбор", Constraints: "добавь пример к каждому совету"}
	if _, err := agent.ApplyContextCommand("anna", models.ContextCommand{Action: "set_profile", Profile: anna}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("maxim", models.ContextCommand{Action: "set_profile", Profile: maxim}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "anna", "Как начать изучать Go?"); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Respond(context.Background(), "maxim", "Как начать изучать Go?"); err != nil {
		t.Fatal(err)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(client.requests))
	}
	annaPrompt, maximPrompt := client.requests[0][1].Content, client.requests[1][1].Content
	if !strings.Contains(annaPrompt, "Анна") || !strings.Contains(annaPrompt, "ровно 3 пункта") || !strings.Contains(annaPrompt, "без англицизмов") {
		t.Fatalf("Anna profile was not sent: %q", annaPrompt)
	}
	if !strings.Contains(maximPrompt, "Максим") || !strings.Contains(maximPrompt, "подробный разбор") || !strings.Contains(maximPrompt, "добавь пример") {
		t.Fatalf("Maxim profile was not sent: %q", maximPrompt)
	}
	if annaPrompt == maximPrompt {
		t.Fatal("different profiles produced identical model context")
	}
	if got := agent.State("anna").Profile; got.Name != anna.Name || got.Style != anna.Style || got.Format != anna.Format || got.Constraints != anna.Constraints || got.ID == "" {
		t.Fatalf("Anna profile = %#v, want saved values %#v", got, anna)
	}
	if err := agent.Clear("anna"); err != nil {
		t.Fatal(err)
	}
	if got := agent.State("anna").Profile; got.Name != anna.Name || got.ID == "" {
		t.Fatalf("profile after clearing task = %#v, want saved values %#v", got, anna)
	}
}

func TestProfilePersistsAfterRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "profile.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	want := models.UserProfile{Name: "Ирина", Style: "лаконичный", Format: "список", Constraints: "не более 4 пунктов"}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "set_profile", Profile: want}); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if got := second.State("session").Profile; got.Name != want.Name || got.Style != want.Style || got.ID == "" {
		t.Fatalf("restored profile = %#v, want saved values %#v", got, want)
	}
}

func TestNamedProfilesAreSelectedPerSessionAndLongTermMemoryIsGlobalPerUser(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	const userID = "user-a"
	chemistry := models.UserProfile{Name: "Химик", Perspective: "химик-популяризатор: объясняй причины и механизмы", Style: "научно-популярный", Format: "короткий пост"}
	psychology := models.UserProfile{Name: "Психолог", Perspective: "психолог-практик: объясняй поведение без диагнозов", Style: "бережный", Format: "пост с примером"}

	chemistryState, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "create_profile", Profile: chemistry})
	if err != nil {
		t.Fatal(err)
	}
	chemistryID := chemistryState.ActiveProfileID
	psychologyState, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "create_profile", Profile: psychology})
	if err != nil {
		t.Fatal(err)
	}
	psychologyID := psychologyState.ActiveProfileID
	if chemistryID == psychologyID || len(psychologyState.Profiles) != 2 {
		t.Fatalf("profiles = %#v", psychologyState.Profiles)
	}
	if _, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "set_active_profile", ProfileID: chemistryID}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommandForUser(userID, "session-review", models.ContextCommand{Action: "set_active_profile", ProfileID: psychologyID}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithUserOptions(context.Background(), userID, "session-post", "Напиши пост о кофеине", 2, models.StrategySlidingWindow, ""); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.RespondWithUserOptions(context.Background(), userID, "session-review", "Напиши пост о кофеине", 2, models.StrategySlidingWindow, ""); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(client.requests[0][1].Content, "химик-популяризатор") || !strings.Contains(client.requests[1][1].Content, "психолог-практик") {
		t.Fatalf("profile prompts = %#v", client.requests)
	}
	if agent.StateForUser(userID, "session-post").ActiveProfileID != chemistryID || agent.StateForUser(userID, "session-review").ActiveProfileID != psychologyID {
		t.Fatal("active profile must belong to a session")
	}
	if _, err := agent.ApplyContextCommandForUser(userID, "session-post", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "audience", Value: "Пишем для начинающих"}); err != nil {
		t.Fatal(err)
	}
	otherSession := agent.StateForUser(userID, "session-review")
	if len(otherSession.Memory.LongTerm) != 1 || otherSession.Memory.LongTerm[0].Value != "Пишем для начинающих" {
		t.Fatalf("global long-term memory = %#v", otherSession.Memory.LongTerm)
	}
	if got := agent.StateForUser("user-b", "other-session").Memory.LongTerm; len(got) != 0 {
		t.Fatalf("another user's memory leaked: %#v", got)
	}
}

func TestUserProfilesAndGlobalMemoryPersistAcrossRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "user-state.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	const userID = "user-persisted"
	created, err := first.ApplyContextCommandForUser(userID, "first-session", models.ContextCommand{Action: "create_profile", Profile: models.UserProfile{Name: "Экономист", Perspective: "экономист, объясняй стимулы и последствия"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommandForUser(userID, "first-session", models.ContextCommand{Action: "save_memory", Layer: models.MemoryLongTerm, Category: "knowledge", Key: "audience", Value: "Предприниматели"}); err != nil {
		t.Fatal(err)
	}
	second, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	state := second.StateForUser(userID, "second-session")
	if len(state.Profiles) != 1 || state.Profiles[0].Name != "Экономист" || len(state.Memory.LongTerm) != 1 || state.Memory.LongTerm[0].Value != "Предприниматели" {
		t.Fatalf("restored user state = %#v", state)
	}
	if _, err := second.ApplyContextCommandForUser(userID, "second-session", models.ContextCommand{Action: "set_active_profile", ProfileID: created.ActiveProfileID}); err != nil {
		t.Fatal(err)
	}
	if got := second.StateForUser(userID, "second-session").Profile.Name; got != "Экономист" {
		t.Fatalf("active profile after restart = %q", got)
	}
}

func TestSelectedDialogueMessageIsStoredWholeWithoutUserKey(t *testing.T) {
	agent := New(&fakeCompleter{answer: "Ответ агента"})
	if _, err := agent.Respond(context.Background(), "session", "Не показывать карту курьера", 2); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryWorking, Value: "Не показывать карту курьера"}); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "save_message", Layer: models.MemoryLongTerm, Category: "decisions", Value: "Ответ агента"}); err != nil {
		t.Fatal(err)
	}
	state := agent.State("session")
	if got := state.Memory.Working; len(got) != 1 || got[0].Key != "message-1" || got[0].Value != "Не показывать карту курьера" {
		t.Fatalf("working memory = %#v", got)
	}
	if got := state.Memory.LongTerm; len(got) != 1 || got[0].Key != "message-2" || got[0].Category != "decisions" || got[0].Value != "Ответ агента" {
		t.Fatalf("long-term memory = %#v", got)
	}
}

func TestModelSelectionPersistsAndUsesSelectedModel(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ Pro"}
	agent := New(client)
	result, err := agent.RespondWithOptions(context.Background(), "session", "Составь план", 2, models.StrategySlidingWindow, models.DeepSeekProModel)
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != models.DeepSeekProModel || len(client.models) != 1 || client.models[0] != models.DeepSeekProModel {
		t.Fatalf("selected model state=%q calls=%#v", result.Model, client.models)
	}
	if state := agent.State("session"); state.Model != models.DeepSeekProModel {
		t.Fatalf("model after response = %q", state.Model)
	}
}

func TestBranchingCreatesIndependentChildrenFromCheckpoint(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	agent := New(client)
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Общая задача", 2, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	state, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "checkpoint", Name: "до решения"})
	if err != nil || len(state.Checkpoints) != 1 {
		t.Fatalf("checkpoint state = %#v, err=%v", state, err)
	}
	checkpointID := state.Checkpoints[0].ID
	state, err = agent.ApplyContextCommand("session", models.ContextCommand{Action: "create_branch", CheckpointID: checkpointID, Name: "вариант A"})
	if err != nil {
		t.Fatal(err)
	}
	branchA := state.ActiveBranchID
	if _, err := agent.RespondWithStrategy(context.Background(), "session", "Продолжение A", 0, models.StrategyBranching); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "create_branch", CheckpointID: checkpointID, Name: "вариант B"}); err != nil {
		t.Fatal(err)
	}
	if got := agent.History("session"); len(got) != 2 || got[0].Content != "Общая задача" {
		t.Fatalf("new branch history = %#v", got)
	}
	if _, err := agent.ApplyContextCommand("session", models.ContextCommand{Action: "switch_branch", BranchID: branchA}); err != nil {
		t.Fatal(err)
	}
	if got := agent.History("session"); len(got) != 4 || got[2].Content != "Продолжение A" {
		t.Fatalf("restored branch history = %#v", got)
	}
}
