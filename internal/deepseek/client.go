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
	endpoint       = "https://api.deepseek.com/chat/completions"
	modelsEndpoint = "https://api.deepseek.com/models"
	model          = "deepseek-v4-flash"
)

var (
	ErrNoAPIKey     = errors.New("DeepSeek API key is not configured")
	ErrUnauthorized = errors.New("DeepSeek API key was rejected")
	ErrRateLimited  = errors.New("DeepSeek API rate limit reached")
	ErrTimeout      = errors.New("DeepSeek API request timed out")
	ErrUpstream     = errors.New("DeepSeek API request failed")
)

type Client struct {
	apiKey     string
	httpClient *http.Client
	endpoint   string
}

type Completion = models.ModelCompletion

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
	Model          string          `json:"model"`
	Messages       []message       `json:"messages"`
	Thinking       thinking        `json:"thinking"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	MaxTokens      int             `json:"max_tokens"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
}

type responseFormat struct {
	Type string `json:"type"`
}

type thinking struct {
	Type string `json:"type"`
}

type completionResponse struct {
	Choices []struct {
		Message      message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens          int `json:"prompt_tokens"`
		CompletionTokens      int `json:"completion_tokens"`
		TotalTokens           int `json:"total_tokens"`
		PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
	} `json:"usage"`
}

// Complete sends one request. The key is kept only in this server-side client.
func (c *Client) Complete(ctx context.Context, prompt string, mode models.ResponseMode, settings models.GenerationSettings) (string, string, error) {
	if c.apiKey == "" {
		return "", "", ErrNoAPIKey
	}
	system, responseFormat, stop, maxTokens := requestControls(mode, settings.MaxTokens)
	return c.complete(ctx, system, prompt, settings, responseFormat, stop, maxTokens)
}

// CompleteWithSystem is used by the lesson that compares reasoning approaches.
// The caller supplies only instructional text; the API key remains server-side.
func (c *Client) CompleteWithSystem(ctx context.Context, system, prompt string, settings models.GenerationSettings) (string, string, error) {
	if c.apiKey == "" {
		return "", "", ErrNoAPIKey
	}
	return c.complete(ctx, system, prompt, settings, nil, nil, settings.MaxTokens)
}

// CompleteModel runs a named catalog model with the same system instruction and
// request controls. It is used only by lesson 5.
func (c *Client) CompleteModel(ctx context.Context, modelName, system, prompt string, settings models.GenerationSettings) (Completion, error) {
	if c.apiKey == "" {
		return Completion{}, ErrNoAPIKey
	}
	// Pro can legitimately take longer than the regular chat lesson. Keep the
	// longer allowance local to this explicit comparison rather than slowing all
	// other endpoints.
	comparisonClient := *c
	comparisonHTTPClient := *c.httpClient
	comparisonHTTPClient.Timeout = 120 * time.Second
	comparisonClient.httpClient = &comparisonHTTPClient
	return comparisonClient.completeModel(ctx, modelName, system, prompt, settings, nil, nil, settings.MaxTokens)
}

// ListModels is a small, read-only capability check. It never exposes the key.
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	if c.apiKey == "" {
		return nil, ErrNoAPIKey
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsEndpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("create model request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return nil, ErrTimeout
		}
		return nil, ErrUpstream
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return nil, ErrUnauthorized
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, ErrRateLimited
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, ErrUpstream
	}
	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&decoded); err != nil {
		return nil, ErrUpstream
	}
	ids := make([]string, 0, len(decoded.Data))
	for _, item := range decoded.Data {
		if id := strings.TrimSpace(item.ID); id != "" {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (c *Client) complete(ctx context.Context, system, prompt string, settings models.GenerationSettings, responseFormat *responseFormat, stop []string, maxTokens int) (string, string, error) {
	result, err := c.completeModel(ctx, model, system, prompt, settings, responseFormat, stop, maxTokens)
	return result.Answer, result.FinishReason, err
}

func (c *Client) completeModel(ctx context.Context, modelName, system, prompt string, settings models.GenerationSettings, responseFormat *responseFormat, stop []string, maxTokens int) (Completion, error) {

	body, err := json.Marshal(completionRequest{
		Model: modelName,
		Messages: []message{
			{Role: "system", Content: system},
			{Role: "user", Content: prompt},
		},
		Thinking:       thinking{Type: "disabled"},
		Temperature:    settings.Temperature,
		TopP:           settings.TopP,
		MaxTokens:      maxTokens,
		ResponseFormat: responseFormat,
		Stop:           stop,
	})
	if err != nil {
		return Completion{}, fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return Completion{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return Completion{}, ErrTimeout
		}
		return Completion{}, ErrUpstream
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return Completion{}, ErrUnauthorized
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return Completion{}, ErrRateLimited
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return Completion{}, ErrUpstream
	}

	var decoded completionResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil {
		return Completion{}, ErrUpstream
	}
	if len(decoded.Choices) == 0 || strings.TrimSpace(decoded.Choices[0].Message.Content) == "" {
		return Completion{}, ErrUpstream
	}
	answer := strings.TrimSpace(strings.ReplaceAll(decoded.Choices[0].Message.Content, models.StopSequence, ""))
	return Completion{Answer: answer, FinishReason: decoded.Choices[0].FinishReason, Usage: models.ModelUsage{
		InputTokens: decoded.Usage.PromptTokens, OutputTokens: decoded.Usage.CompletionTokens, TotalTokens: decoded.Usage.TotalTokens,
		CacheHitTokens: decoded.Usage.PromptCacheHitTokens, CacheMissTokens: decoded.Usage.PromptCacheMissTokens,
	}}, nil
}

func requestControls(mode models.ResponseMode, selectedMaxTokens int) (string, *responseFormat, []string, int) {
	base := "You are a helpful assistant. Answer accurately in Russian. Do not reveal private reasoning."
	jsonInstruction := ` Return only valid JSON, without Markdown or extra text, with exactly this shape: {"film":"название фильма","actors":["полное имя"],"answer":"краткий ответ"}. Include 3 to 6 actors.`
	lengthInstruction := " Keep the answer concise: no more than 300 characters."
	finishInstruction := " Finish immediately after the complete answer. Do not add recommendations, questions, or extra commentary."

	switch mode {
	case models.ModeFormat:
		return base + jsonInstruction, &responseFormat{Type: "json_object"}, nil, selectedMaxTokens
	case models.ModeLength:
		return base + lengthInstruction, nil, nil, 120
	case models.ModeFinish:
		return base + finishInstruction + " End the response with the marker " + models.StopSequence + ".", nil, []string{models.StopSequence}, selectedMaxTokens
	case models.ModeAll:
		return base + jsonInstruction + " The value of answer must be no more than 220 characters." + finishInstruction + " After the closing JSON brace, write the marker " + models.StopSequence + ".", &responseFormat{Type: "json_object"}, []string{models.StopSequence}, 180
	default:
		return base, nil, nil, selectedMaxTokens
	}
}
