package agent

import (
	"ai-challenge-app/internal/models"
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLocalGenerationSettingsSurviveRestart(t *testing.T) {
	store := NewJSONStore(filepath.Join(t.TempDir(), "settings.json"))
	first, err := NewPersistent(&fakeCompleter{}, store)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: "ollama/qwen3:4b"}); err != nil {
		t.Fatal(err)
	}
	temperature, think, penalty, topK, seed := 0.0, false, 1.15, 20, int64(0)
	settings := models.GenerationSettings{Temperature: &temperature, MaxTokens: 256, ContextWindow: 2048, LocalJSON: true, LocalThink: &think, TopK: &topK, Seed: &seed, RepeatPenalty: &penalty}
	if _, err := first.ApplyContextCommand("session", models.ContextCommand{Action: "set_generation_settings", Settings: settings}); err != nil {
		t.Fatal(err)
	}
	client := &fakeCompleter{answer: `{"ok":true}`}
	restored, err := NewPersistent(client, store)
	if err != nil {
		t.Fatal(err)
	}
	response, err := restored.Respond(context.Background(), "session", "Привет")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(client.settings[0], settings) || !reflect.DeepEqual(response.Settings, settings) || response.Tokens.ContextLimitTokens != 2048 {
		t.Fatalf("response=%#v settings=%#v", response, client.settings)
	}
}

func TestLocalChatKeepsExistingProfileAndHistory(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	a := New(client)
	if _, err := a.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: "ollama/qwen3:4b"}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommand("session", models.ContextCommand{Action: "set_profile", Profile: models.UserProfile{Name: "Поддержка", Perspective: "PROFILE_SUPPORT_RULES"}}); err != nil {
		t.Fatal(err)
	}
	for _, question := range []string{"PREVIOUS_QUESTION", "CURRENT_QUESTION"} {
		if _, err := a.Respond(context.Background(), "session", question); err != nil {
			t.Fatal(err)
		}
	}
	var contents []string
	for _, message := range client.requests[1] {
		contents = append(contents, message.Content)
	}
	prompt := strings.Join(contents, "\n")
	for _, expected := range []string{"PROFILE_SUPPORT_RULES", "PREVIOUS_QUESTION", "CURRENT_QUESTION"} {
		if !strings.Contains(prompt, expected) {
			t.Fatalf("%q missing in %q", expected, prompt)
		}
	}
}

func TestCloudGenerationIsNotBlockedByLocalSettings(t *testing.T) {
	temperature, topP, topK, seed, penalty, think := 0.35, 0.9, 0, int64(2147483648), -1.0, true
	for _, sampling := range []models.GenerationSettings{
		{Temperature: &temperature, MaxTokens: 3000},
		{TopP: &topP, MaxTokens: 3000},
	} {
		client := &fakeCompleter{answer: "Ответ"}
		a := New(client)
		sampling.ContextWindow = 1
		sampling.TopK, sampling.Seed, sampling.RepeatPenalty = &topK, &seed, &penalty
		sampling.LocalJSON, sampling.LocalThink = true, &think
		if _, err := a.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: models.DeepSeekFlashModel}); err != nil {
			t.Fatal(err)
		}
		if _, err := a.ApplyContextCommand("session", models.ContextCommand{Action: "set_generation_settings", Settings: sampling}); err != nil {
			t.Fatalf("local-only settings blocked cloud save: %v", err)
		}
		response, err := a.RespondWithUserOptionsRAGConfiguredSettings(context.Background(), "session", "session", "Привет", 10, models.StrategySlidingWindow, models.DeepSeekFlashModel, false, models.RAGOptions{}, sampling)
		if err != nil {
			t.Fatalf("local-only settings blocked cloud chat: %v", err)
		}
		if !reflect.DeepEqual(client.settings[0], sampling) || response.Tokens.ContextLimitTokens == 1 {
			t.Fatalf("cloud settings or context changed: %#v", response)
		}
		if _, err := a.ApplyContextCommand("session", models.ContextCommand{Action: "set_model", Model: "ollama/qwen3:4b"}); err != nil {
			t.Fatal(err)
		}
		if _, err := a.ApplyContextCommand("session", models.ContextCommand{Action: "set_generation_settings", Settings: sampling}); err == nil {
			t.Fatal("invalid local settings accepted for local model")
		}
	}
}
