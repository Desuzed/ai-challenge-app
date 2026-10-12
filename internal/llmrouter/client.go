// Package llmrouter routes model IDs to their named provider.
package llmrouter

import (
	"context"
	"errors"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/ollama"
)

type Client struct {
	DeepSeek     *deepseek.Client
	Ollama       *ollama.Client
	RemoteOllama *ollama.Client
}

func (c *Client) ollamaClient(model string) (*ollama.Client, error) {
	if models.IsRemoteOllamaModel(model) {
		if c.RemoteOllama == nil {
			return nil, errors.New("Удалённая Ollama не настроена")
		}
		return c.RemoteOllama, nil
	}
	if c.Ollama == nil {
		return nil, errors.New("Локальная Ollama не настроена")
	}
	return c.Ollama, nil
}
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
	if models.IsOllamaModel(model) {
		provider, err := c.ollamaClient(model)
		if err != nil {
			return models.ModelCompletion{}, err
		}
		return provider.CompleteMessagesModel(ctx, model, messages, settings)
	}
	return c.DeepSeek.CompleteMessagesModel(ctx, model, messages, settings)
}
func (c *Client) CompleteMessagesModelJSON(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	if models.IsOllamaModel(model) {
		provider, err := c.ollamaClient(model)
		if err != nil {
			return models.ModelCompletion{}, err
		}
		return provider.CompleteMessagesModelJSON(ctx, model, messages, settings)
	}
	return c.DeepSeek.CompleteMessagesModelJSON(ctx, model, messages, settings)
}
func (c *Client) CompleteMessagesModelWithTools(ctx context.Context, model string, messages []models.ChatMessage, settings models.GenerationSettings, tools []models.ToolDefinition) (models.ModelCompletion, error) {
	if models.IsOllamaModel(model) {
		provider, err := c.ollamaClient(model)
		if err != nil {
			return models.ModelCompletion{}, err
		}
		return provider.CompleteMessagesModelWithTools(ctx, model, messages, settings, tools)
	}
	return c.DeepSeek.CompleteMessagesModelWithTools(ctx, model, messages, settings, tools)
}
