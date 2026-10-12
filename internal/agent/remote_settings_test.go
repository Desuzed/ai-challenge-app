package agent

import (
	"ai-challenge-app/internal/models"
	"context"
	"testing"
)

func TestRemoteChatDoesNotInheritMacBudgets(t *testing.T) {
	client := &fakeCompleter{answer: "Ответ"}
	a := New(client)
	if _, err := a.ApplyContextCommand("remote", models.ContextCommand{Action: "set_model", Model: "ollama/qwen3:4b"}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.ApplyContextCommand("remote", models.ContextCommand{Action: "set_generation_settings", Settings: models.GenerationSettings{MaxTokens: 256, ContextWindow: 2048}}); err != nil {
		t.Fatal(err)
	}
	state, err := a.ApplyContextCommand("remote", models.ContextCommand{Action: "set_model", Model: "ollama-remote/qwen3:4b"})
	if err != nil {
		t.Fatal(err)
	}
	if state.Settings.MaxTokens != 0 || state.Settings.ContextWindow != 0 || state.Tokens.ContextLimitTokens != 0 {
		t.Fatalf("remote inherited budgets: %#v", state.Settings)
	}
	response, err := a.Respond(context.Background(), "remote", "Привет")
	if err != nil {
		t.Fatal(err)
	}
	if client.settings[0].MaxTokens != 0 || client.settings[0].ContextWindow != 0 || response.Tokens.EstimatedCostUSD != 0 {
		t.Fatalf("remote settings=%#v tokens=%#v", client.settings[0], response.Tokens)
	}
	if err := validateGenerationSettings(models.GenerationSettings{MaxTokens: 16000, ContextWindow: 65536}, "ollama-remote/qwen3:4b"); err != nil {
		t.Fatalf("remote budgets were capped by Mac policy: %v", err)
	}
}
