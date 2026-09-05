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

func (c *Client) complete(ctx context.Context, system, prompt string, settings models.GenerationSettings, responseFormat *responseFormat, stop []string, maxTokens int) (string, string, error) {

	body, err := json.Marshal(completionRequest{
		Model: model,
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
		return "", "", fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return "", "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", "", ErrUpstream
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return "", "", ErrUnauthorized
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return "", "", ErrRateLimited
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", "", ErrUpstream
	}

	var decoded completionResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded); err != nil {
		return "", "", ErrUpstream
	}
	if len(decoded.Choices) == 0 || strings.TrimSpace(decoded.Choices[0].Message.Content) == "" {
		return "", "", ErrUpstream
	}
	answer := strings.TrimSpace(strings.ReplaceAll(decoded.Choices[0].Message.Content, models.StopSequence, ""))
	return answer, decoded.Choices[0].FinishReason, nil
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
