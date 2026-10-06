// Package llmrouter routes model IDs to their named provider.
package llmrouter

import (
	"context"
	"strings"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/ollama"
)

type Client struct {
	DeepSeek *deepseek.Client
	Ollama   *ollama.Client
}

func (c *Client) local(model string) bool { return strings.HasPrefix(model, ollama.ModelPrefix) }
func (c *Client) Complete(ctx context.Context, prompt string, mode models.ResponseMode, settings models.GenerationSettings) (string, string, error) {
	return c.DeepSeek.Complete(ctx, prompt, mode, settings)
}
func (c *Client) CompleteWithSystem(ctx context.Context, system, prompt string, settings models.GenerationSettings) (string, string, error) {
	return c.DeepSeek.CompleteWithSystem(ctx, system, prompt, settings)
}
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	if c.DeepSeek == nil {
		return nil, nil
	}
	return c.DeepSeek.ListModels(ctx)
}

func (c *Client) ListLocalModels(ctx context.Context) ([]string, error) {
	if c.Ollama == nil {
		return nil, nil
	}
	return c.Ollama.ListModels(ctx)
}
func (c *Client) CompleteModel(ctx context.Context, model, system, prompt string, settings models.GenerationSettings) (models.ModelCompletion, error) {
	return c.DeepSeek.CompleteModel(ctx, model, system, prompt, settings)
}
func (c *Client) CompleteMessages(ctx context.Context, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	return c.DeepSeek.CompleteMessages(ctx, messages, settings)
}
func (c *Client) CompleteMessagesModel(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	if c.local(model) {
		return c.Ollama.CompleteMessagesModel(ctx, model, messages, settings)
	}
	return c.DeepSeek.CompleteMessagesModel(ctx, model, messages, settings)
}
func (c *Client) CompleteMessagesModelJSON(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	if c.local(model) {
		return c.Ollama.CompleteMessagesModelJSON(ctx, model, messages, settings)
	}
	return c.DeepSeek.CompleteMessagesModelJSON(ctx, model, messages, settings)
}
func (c *Client) CompleteMessagesModelWithTools(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings, tools []models.ToolDefinition) (models.ModelCompletion, error) {
	if c.local(model) {
		return c.Ollama.CompleteMessagesModelWithTools(ctx, model, messages, settings, tools)
	}
	return c.DeepSeek.CompleteMessagesModelWithTools(ctx, model, messages, settings, tools)
}
