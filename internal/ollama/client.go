// Package ollama provides an offline chat client for locally installed Ollama models.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"

	"ai-challenge-app/internal/models"
)

const ModelPrefix = "ollama/"

type Client struct {
	baseURL    string
	http       *http.Client
	thinkingMu sync.RWMutex
	thinking   map[string]bool
}

func New(baseURL string, timeout time.Duration) *Client {
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), http: &http.Client{Timeout: timeout, Transport: &http.Transport{Proxy: nil}, CheckRedirect: func(request *http.Request, _ []*http.Request) error {
		if !isLoopbackURL(request.URL) {
			return errors.New("Ollama redirect запрещён: локальный клиент не подключается к удалённому адресу")
		}
		return nil
	}}, thinking: make(map[string]bool)}
}

func isLoopbackURL(target *url.URL) bool {
	if target == nil || target.Scheme != "http" && target.Scheme != "https" {
		return false
	}
	host := target.Hostname()
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// IsLocalURL is shared by the local embedding transport.
func IsLocalURL(raw string) bool {
	parsed, err := url.Parse(raw)
	return err == nil && isLoopbackURL(parsed)
}

type chatRequest struct {
	Model    string                  `json:"model"`
	Messages []message               `json:"messages"`
	Stream   bool                    `json:"stream"`
	Think    bool                    `json:"think"`
	Format   any                     `json:"format,omitempty"`
	Options  map[string]any          `json:"options"`
	Tools    []models.ToolDefinition `json:"tools,omitempty"`
}

type showRequest struct {
	Model string `json:"model"`
}

type showResponse struct {
	RemoteHost  string `json:"remote_host"`
	RemoteModel string `json:"remote_model"`
	Thinking    struct {
		Values  []bool `json:"values"`
		Default *bool  `json:"default"`
	} `json:"thinking"`
}

type message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}
type toolCall struct {
	Function struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"function"`
}

type chatResponse struct {
	Model   string `json:"model"`
	Message struct {
		Content   string `json:"content"`
		Thinking  string `json:"thinking"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Type     string `json:"type"`
			Function struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	Done       bool   `json:"done"`
	DoneReason string `json:"done_reason"`
	PromptEval int    `json:"prompt_eval_count"`
	Eval       int    `json:"eval_count"`
}

func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	parsed, _ := url.Parse(c.baseURL)
	if !isLoopbackURL(parsed) {
		return nil, errors.New("OLLAMA_URL должен указывать на локальный адрес")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	response, err := c.http.Do(request)
	if err != nil {
		return nil, fmt.Errorf("Ollama недоступна: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Ollama /api/tags: HTTP %d", response.StatusCode)
	}
	var payload struct {
		Models []struct {
			Name         string   `json:"name"`
			RemoteHost   string   `json:"remote_host"`
			RemoteModel  string   `json:"remote_model"`
			Capabilities []string `json:"capabilities"`
			Details      struct {
				Family   string   `json:"family"`
				Families []string `json:"families"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	result := make([]string, 0, len(payload.Models))
	for _, model := range payload.Models {
		name := strings.TrimSpace(model.Name)
		if name == "" || model.RemoteHost != "" || model.RemoteModel != "" || strings.Contains(strings.ToLower(name), "cloud") || strings.Contains(strings.ToLower(name), "embed") {
			continue
		}
		if len(model.Capabilities) > 0 {
			completion := false
			for _, capability := range model.Capabilities {
				if capability == "completion" {
					completion = true
					break
				}
			}
			if !completion {
				continue
			}
		}
		family := strings.ToLower(model.Details.Family)
		if family == "bert" || strings.Contains(family, "embed") {
			continue
		}
		result = append(result, ModelPrefix+name)
	}
	sort.Strings(result)
	return result, nil
}

func (c *Client) CompleteMessages(ctx context.Context, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	return c.CompleteMessagesModel(ctx, "qwen3:4b", messages, settings)
}
func (c *Client) CompleteMessagesModel(ctx context.Context, modelID string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	return c.complete(ctx, modelID, messages, settings, "", nil)
}
func (c *Client) CompleteMessagesModelJSON(ctx context.Context, modelID string, messages []models.ChatMessage, settings models.GenerationSettings) (models.ModelCompletion, error) {
	var format any = "json"
	compact := false
	for _, item := range messages {
		if item.Role == "system" && strings.Contains(item.Content, `"claims"`) && strings.Contains(item.Content, `"chunkId"`) {
			compact = strings.Contains(item.Content, `используй "answer":""`)
			format = groundedSchema(compact, strings.Contains(item.Content, "Сохрани также поле plan"))
			break
		}
	}
	completion, err := c.complete(ctx, modelID, messages, settings, format, nil)
	if err != nil {
		return completion, err
	}
	if _, grounded := format.(map[string]any); grounded {
		completion.OutputTokenBudget = settings.MaxTokens
		completion.CompletionAttempts = 1
	}
	// A truncated JSON document is never shown or salvaged. Regenerate once
	// with room for the envelope, within the ordinary RAG reservation (2200).
	// Also recheck an empty claim list when documents were supplied; the second
	// response may still abstain, and evidence validation remains mandatory.
	// Only compact factual RAG is eligible; free chat, tools and planner aren't.
	retryReason := ""
	if completion.FinishReason == "length" && !json.Valid([]byte(completion.Answer)) {
		retryReason = "truncated_json"
	} else if compact {
		var envelope struct {
			Claims []json.RawMessage `json:"claims"`
		}
		if json.Unmarshal([]byte(completion.Answer), &envelope) == nil && len(envelope.Claims) == 0 {
			for _, item := range messages {
				if item.Role == "system" && strings.HasPrefix(item.Content, "Фрагменты локальных документов для ответа на последний вопрос.") && strings.Contains(item.Content, "Фрагмент 1") {
					retryReason = "empty_claims_with_retrieved_context"
					break
				}
			}
		}
	}
	if !compact || retryReason == "" || settings.MaxTokens >= 2200 {
		return completion, nil
	}
	retrySettings := settings
	retrySettings.MaxTokens = 2200
	log.Printf("LLM structured retry provider=ollama model=%s initial_budget=%d retry_budget=%d reason=%s", modelID, settings.MaxTokens, retrySettings.MaxTokens, retryReason)
	retry, err := c.complete(ctx, modelID, messages, retrySettings, format, nil)
	if err != nil {
		return models.ModelCompletion{}, err
	}
	retry.Usage.InputTokens += completion.Usage.InputTokens
	retry.Usage.OutputTokens += completion.Usage.OutputTokens
	retry.Usage.TotalTokens += completion.Usage.TotalTokens
	retry.Usage.CacheHitTokens += completion.Usage.CacheHitTokens
	retry.Usage.CacheMissTokens += completion.Usage.CacheMissTokens
	retry.OutputTokenBudget = retrySettings.MaxTokens
	retry.CompletionAttempts = 2
	retry.CompletionRetryReason = retryReason
	return retry, nil
}
func (c *Client) CompleteMessagesModelWithTools(ctx context.Context, modelID string, messages []models.ChatMessage, settings models.GenerationSettings, tools []models.ToolDefinition) (models.ModelCompletion, error) {
	return c.complete(ctx, modelID, messages, settings, "", tools)
}

func (c *Client) complete(ctx context.Context, modelID string, messages []models.ChatMessage, settings models.GenerationSettings, format any, tools []models.ToolDefinition) (models.ModelCompletion, error) {
	modelID = strings.TrimPrefix(modelID, ModelPrefix)
	if strings.Contains(strings.ToLower(modelID), "cloud") {
		return models.ModelCompletion{}, errors.New("Облачная модель Ollama запрещена для локальной генерации")
	}
	if modelID == "" {
		return models.ModelCompletion{}, errors.New("Не выбрана локальная модель Ollama.")
	}
	options := map[string]any{"num_predict": settings.MaxTokens, "num_ctx": 8192}
	if settings.Temperature != nil {
		options["temperature"] = *settings.Temperature
	}
	if settings.TopP != nil {
		options["top_p"] = *settings.TopP
	}
	think, err := c.thinkingPreference(ctx, modelID)
	if err != nil {
		log.Printf("LLM metadata provider=ollama model=%s thinking=unknown error=%v", modelID, err)
		return models.ModelCompletion{}, fmt.Errorf("не удалось безопасно определить режим рассуждений локальной модели: %w", err)
	}
	// Qwen3 accepts think=false for native JSON despite this runner advertising
	// only true in /api/show. With think=true a RAG request can exhaust its token
	// budget in the thinking channel and never produce the JSON envelope.
	// Keep ordinary chat/tool reasoning policy and original messages intact.
	if format != nil && format != "" && strings.HasPrefix(strings.ToLower(modelID), "qwen3") {
		think = false
	}
	if format == "" {
		format = nil
	}
	ollamaMessages := make([]message, 0, len(messages))
	for i, item := range messages {
		converted := message{Role: item.Role, Content: item.Content, ToolCallID: item.ToolCallID}
		for _, call := range item.ToolCalls {
			arguments := map[string]any{}
			_ = json.Unmarshal([]byte(call.Function.Arguments), &arguments)
			entry := toolCall{}
			entry.Function.Name, entry.Function.Arguments = call.Function.Name, arguments
			converted.ToolCalls = append(converted.ToolCalls, entry)
		}
		if converted.Role == "tool" && converted.ToolCallID == "" {
			converted.ToolCallID = fmt.Sprintf("ollama-call-%d", i)
		}
		ollamaMessages = append(ollamaMessages, converted)
	}
	payload := chatRequest{Model: modelID, Messages: ollamaMessages, Stream: false, Think: think, Format: format, Options: options, Tools: tools}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return models.ModelCompletion{}, err
	}
	if parsed, _ := url.Parse(c.baseURL); !isLoopbackURL(parsed) {
		return models.ModelCompletion{}, errors.New("OLLAMA_URL должен указывать на локальный адрес")
	}
	log.Printf("LLM request provider=ollama model=%s endpoint=%s/api/chat messages=%d options=%s think=%t", modelID, c.baseURL, len(messages), mustJSON(options), think)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(encoded))
	if err != nil {
		return models.ModelCompletion{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := c.http.Do(request)
	if err != nil {
		log.Printf("LLM response provider=ollama model=%s error=%v", modelID, err)
		return models.ModelCompletion{}, fmt.Errorf("обращение к локальной модели Ollama: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		log.Printf("LLM response provider=ollama model=%s status=%d", modelID, response.StatusCode)
		return models.ModelCompletion{}, fmt.Errorf("Ollama вернула HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	var result chatResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 16<<20)).Decode(&result); err != nil {
		return models.ModelCompletion{}, fmt.Errorf("разобрать ответ Ollama: %w", err)
	}
	completion := models.ModelCompletion{Answer: result.Message.Content, FinishReason: result.DoneReason, Usage: models.ModelUsage{InputTokens: result.PromptEval, OutputTokens: result.Eval, TotalTokens: result.PromptEval + result.Eval}}
	for index, call := range result.Message.ToolCalls {
		args, _ := json.Marshal(call.Function.Arguments)
		id := call.ID
		if id == "" {
			id = fmt.Sprintf("ollama-call-%d", index)
		}
		completion.ToolCalls = append(completion.ToolCalls, models.ToolCall{ID: id, Type: "function", Function: models.ToolCallFunction{Name: call.Function.Name, Arguments: string(args)}})
	}
	log.Printf("LLM response provider=ollama model=%s done=%t reason=%s input_tokens=%d output_tokens=%d thinking_chars=%d final_chars=%d", modelID, result.Done, result.DoneReason, result.PromptEval, result.Eval, len([]rune(result.Message.Thinking)), len([]rune(result.Message.Content)))
	if !result.Done && result.DoneReason == "" {
		return models.ModelCompletion{}, errors.New("Ollama завершила ответ без признака done")
	}
	return completion, nil
}

func (c *Client) thinkingPreference(ctx context.Context, modelID string) (bool, error) {
	c.thinkingMu.RLock()
	value, ok := c.thinking[modelID]
	c.thinkingMu.RUnlock()
	if ok {
		return value, nil
	}
	parsed, _ := url.Parse(c.baseURL)
	if !isLoopbackURL(parsed) {
		return false, errors.New("OLLAMA_URL должен указывать на локальный адрес")
	}
	body, err := json.Marshal(showRequest{Model: modelID})
	if err != nil {
		return false, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return false, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := c.http.Do(request)
	if err != nil {
		return false, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return false, fmt.Errorf("Ollama /api/show вернула HTTP %d", response.StatusCode)
	}
	var metadata showResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&metadata); err != nil {
		return false, err
	}
	if metadata.RemoteHost != "" || metadata.RemoteModel != "" {
		return false, errors.New("Удалённая модель Ollama запрещена для локальной генерации")
	}
	preference := false
	hasTrue, hasFalse := false, false
	for _, supported := range metadata.Thinking.Values {
		if supported {
			hasTrue = true
		} else {
			hasFalse = true
		}
	}
	if hasTrue && !hasFalse {
		preference = true
	} else if len(metadata.Thinking.Values) == 0 && metadata.Thinking.Default != nil {
		preference = *metadata.Thinking.Default
	}
	// If metadata is unavailable or omits the thinking field, use the normal
	// non-thinking mode. A model that only supports thinking returns that channel
	// separately from its final answer.
	c.thinkingMu.Lock()
	c.thinking[modelID] = preference
	c.thinkingMu.Unlock()
	return preference, nil
}

func mustJSON(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}
