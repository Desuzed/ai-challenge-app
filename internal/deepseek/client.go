package deepseek

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"ai-challenge-app/internal/models"
)

const (
	endpoint = "https://api.deepseek.com/chat/completions"
	model    = "deepseek-v4-flash"
)

var (
	ErrNoAPIKey     = errors.New("DeepSeek API key is not configured")
	ErrUnauthorized = errors.New("DeepSeek API key was rejected")
	ErrRateLimited  = errors.New("DeepSeek API rate limit reached")
	ErrUpstream     = errors.New("DeepSeek API request failed")
)

type Client struct {
	apiKey     string
	httpClient *http.Client
	endpoint   string
}

func NewClient(apiKey string, timeout time.Duration) *Client {
	return newClient(apiKey, &http.Client{Timeout: timeout}, endpoint)
}

func newClient(apiKey string, httpClient *http.Client, baseURL string) *Client {
	return &Client{apiKey: strings.TrimSpace(apiKey), httpClient: httpClient, endpoint: baseURL}
}

func ModelName() string { return model }

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	Thinking    thinking  `json:"thinking"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	MaxTokens   int       `json:"max_tokens"`
}

type thinking struct {
	Type string `json:"type"`
}

type completionResponse struct {
	Choices []struct {
		Message message `json:"message"`
	} `json:"choices"`
}

// Complete sends one request. The key is kept only in this server-side client.
func (c *Client) Complete(ctx context.Context, prompt string, settings models.GenerationSettings) (string, error) {
	if c.apiKey == "" {
		return "", ErrNoAPIKey
	}

	body, err := json.Marshal(completionRequest{
		Model: model,
		Messages: []message{
			{Role: "system", Content: "You are a helpful assistant. Give clear, accurate answers in the user's language. Do not reveal private reasoning; give a brief explanation when useful."},
			{Role: "user", Content: prompt},
		},
		Thinking:    thinking{Type: "disabled"},
		Temperature: settings.Temperature,
		TopP:        settings.TopP,
		MaxTokens:   settings.MaxTokens,
	})
	if err != nil {
		return "", fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", ErrUpstream
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return "", ErrUnauthorized
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return "", ErrRateLimited
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", ErrUpstream
	}

	var decoded completionResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil {
		return "", ErrUpstream
	}
	if len(decoded.Choices) == 0 || strings.TrimSpace(decoded.Choices[0].Message.Content) == "" {
		return "", ErrUpstream
	}
	return decoded.Choices[0].Message.Content, nil
}
