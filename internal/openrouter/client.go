package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"ai-challenge-app/internal/models"
)

const (
	modelsEndpoint = "https://openrouter.ai/api/v1/models"
	chatEndpoint   = "https://openrouter.ai/api/v1/chat/completions"
)

var (
	ErrNoAPIKey = errors.New("OpenRouter API key is not configured")
	ErrUpstream = errors.New("OpenRouter request failed")
)

type APIError struct {
	Status  int
	Message string
}

func (e *APIError) Error() string { return fmt.Sprintf("OpenRouter HTTP %d: %s", e.Status, e.Message) }

type Client struct {
	apiKey     string
	httpClient *http.Client
}
type Candidate struct {
	ID   string
	Name string
}

func NewClient(apiKey string, timeout time.Duration) *Client {
	return &Client{apiKey: strings.TrimSpace(apiKey), httpClient: &http.Client{Timeout: timeout}}
}

func (c *Client) DiscoverWeakFreeModel(ctx context.Context) (Candidate, error) {
	if c.apiKey == "" {
		return Candidate{}, ErrNoAPIKey
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsEndpoint, nil)
	if err != nil {
		return Candidate{}, ErrUpstream
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return Candidate{}, ErrUpstream
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return Candidate{}, ErrUpstream
	}
	var decoded struct {
		Data []struct {
			ID      string `json:"id"`
			Name    string `json:"name"`
			Pricing struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
			} `json:"pricing"`
			Architecture struct {
				Modality string `json:"modality"`
			} `json:"architecture"`
		} `json:"data"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&decoded) != nil {
		return Candidate{}, ErrUpstream
	}
	var candidates []Candidate
	for _, item := range decoded.Data {
		if item.Architecture.Modality != "" && !strings.Contains(item.Architecture.Modality, "text") {
			continue
		}
		if !isFree(item.Pricing.Prompt) || !isFree(item.Pricing.Completion) {
			continue
		}
		id := strings.TrimSpace(item.ID)
		name := strings.TrimSpace(item.Name)
		if id == "" || id == "openrouter/free" || !weakName.MatchString(id+" "+name) {
			continue
		}
		candidates = append(candidates, Candidate{ID: id, Name: name})
	}
	if len(candidates) == 0 {
		return Candidate{}, ErrUpstream
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].ID < candidates[j].ID })
	return candidates[0], nil
}

var weakName = regexp.MustCompile(`(?i)(^|[^0-9])[1-4]b|\b(mini|small|nano)\b`)

func isFree(value string) bool {
	price, err := strconv.ParseFloat(value, 64)
	return err == nil && price == 0
}

func (c *Client) Complete(ctx context.Context, candidate Candidate, system, prompt string) (models.ModelCompletion, error) {
	if c.apiKey == "" {
		return models.ModelCompletion{}, ErrNoAPIKey
	}
	body, err := json.Marshal(struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}{Model: candidate.ID, Messages: []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{{"system", system}, {"user", prompt}}})
	if err != nil {
		return models.ModelCompletion{}, ErrUpstream
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, chatEndpoint, bytes.NewReader(body))
	if err != nil {
		return models.ModelCompletion{}, ErrUpstream
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Title", "AI Challenge App")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return models.ModelCompletion{}, ErrUpstream
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return models.ModelCompletion{}, decodeAPIError(resp)
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 2<<20)).Decode(&decoded) != nil || len(decoded.Choices) == 0 || strings.TrimSpace(decoded.Choices[0].Message.Content) == "" {
		return models.ModelCompletion{}, ErrUpstream
	}
	return models.ModelCompletion{Answer: strings.TrimSpace(decoded.Choices[0].Message.Content), FinishReason: decoded.Choices[0].FinishReason, Usage: models.ModelUsage{InputTokens: decoded.Usage.PromptTokens, OutputTokens: decoded.Usage.CompletionTokens, TotalTokens: decoded.Usage.TotalTokens}}, nil
}

func decodeAPIError(resp *http.Response) error {
	var payload struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	_ = json.NewDecoder(io.LimitReader(resp.Body, 32<<10)).Decode(&payload)
	message := strings.TrimSpace(payload.Error.Message)
	if message == "" {
		message = strings.TrimSpace(payload.Message)
	}
	if message == "" {
		message = "провайдер не указал причину"
	}
	return &APIError{Status: resp.StatusCode, Message: message}
}
