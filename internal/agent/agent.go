// Package agent contains the application-level LLM agent. It owns dialogue
// state and generation policy, while HTTP and provider concerns stay outside.
package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"unicode/utf8"

	"ai-challenge-app/internal/models"
)

const maxMessageCharacters = 32000

var ErrEmptyMessage = errors.New("Введите сообщение.")
var ErrMessageTooLong = errors.New("Сообщение слишком длинное.")
var ErrHistorySave = errors.New("Не удалось сохранить историю диалога.")
var ErrContextLimit = errors.New("Диалог не отправлен: история вместе с резервом для ответа превышает контекстное окно модели. Очистите историю или попросите краткое резюме в новом чате.")

type completer interface {
	CompleteMessages(context.Context, []models.ChatMessage, models.GenerationSettings) (models.ModelCompletion, error)
}

type conversation struct {
	mu       sync.Mutex
	messages []models.ChatMessage
	usages   []models.ModelUsage
}

type Agent struct {
	client    completer
	system    string
	settings  models.GenerationSettings
	store     Store
	mu        sync.Mutex
	persistMu sync.Mutex
	sessions  map[string]*conversation
}

func New(client completer) *Agent {
	return newAgent(client, nil, nil)
}

// NewPersistent restores previous sessions before the HTTP server starts.
func NewPersistent(client completer, store Store) (*Agent, error) {
	sessions, err := store.Load()
	if err != nil {
		return nil, err
	}
	return newAgent(client, store, sessions), nil
}

func newAgent(client completer, store Store, restored map[string][]models.ChatMessage) *Agent {
	temperature := 0.7
	value := &Agent{
		client:   client,
		system:   "Ты полезный диалоговый агент и продолжаешь текущий диалог. Перед ответом внимательно учитывай все предыдущие реплики в истории: не отрицай факты, которые в ней явно есть. Отвечай точно, дружелюбно и по-русски. Не раскрывай скрытые внутренние рассуждения.",
		settings: models.GenerationSettings{Temperature: &temperature, MaxTokens: 512},
		sessions: make(map[string]*conversation),
		store:    store,
	}
	for id, messages := range restored {
		value.sessions[id] = &conversation{messages: copyMessages(messages)}
	}
	return value
}

// Respond appends a user message, sends the entire dialogue to the LLM, and
// appends its answer only after a successful completion.
func (a *Agent) Respond(ctx context.Context, sessionID, input string) (models.AgentResponse, error) {
	message := strings.TrimSpace(input)
	if message == "" {
		return models.AgentResponse{}, ErrEmptyMessage
	}
	if utf8.RuneCountInString(message) > maxMessageCharacters {
		return models.AgentResponse{}, ErrMessageTooLong
	}

	// Serializing writes keeps the on-disk snapshot consistent across sessions.
	a.persistMu.Lock()
	defer a.persistMu.Unlock()
	conversation := a.conversation(sessionID)
	conversation.mu.Lock()
	defer conversation.mu.Unlock()
	conversation.messages = append(conversation.messages, models.ChatMessage{Role: "user", Content: message})
	request := make([]models.ChatMessage, 0, len(conversation.messages)+1)
	request = append(request, models.ChatMessage{Role: "system", Content: a.system})
	request = append(request, conversation.messages...)
	report := a.tokenReport(conversation, request, message, models.ModelUsage{})
	if report.EstimatedRequestTokens+report.ReservedOutputTokens > report.ContextLimitTokens {
		conversation.messages = conversation.messages[:len(conversation.messages)-1]
		return models.AgentResponse{}, ErrContextLimit
	}
	completion, err := a.client.CompleteMessages(ctx, request, a.settings)
	if err != nil {
		// Do not retain an unanswered user message: the next attempt is a clean retry.
		conversation.messages = conversation.messages[:len(conversation.messages)-1]
		return models.AgentResponse{}, err
	}
	conversation.messages = append(conversation.messages, models.ChatMessage{Role: "assistant", Content: completion.Answer})
	conversation.usages = append(conversation.usages, completion.Usage)
	if err := a.save(); err != nil {
		conversation.messages = conversation.messages[:len(conversation.messages)-2]
		conversation.usages = conversation.usages[:len(conversation.usages)-1]
		return models.AgentResponse{}, ErrHistorySave
	}
	report = a.tokenReport(conversation, request, message, completion.Usage)
	return models.AgentResponse{Answer: completion.Answer, Messages: copyMessages(conversation.messages), RequestMessages: copyMessages(request), Tokens: report}, nil
}

func (a *Agent) History(sessionID string) []models.ChatMessage {
	conversation := a.conversation(sessionID)
	conversation.mu.Lock()
	defer conversation.mu.Unlock()
	return copyMessages(conversation.messages)
}

// TokenReport is useful on page reloads too: it shows the estimated size of
// restored history even though historical provider usage is not persisted.
func (a *Agent) TokenReport(sessionID string) models.AgentTokenReport {
	conversation := a.conversation(sessionID)
	conversation.mu.Lock()
	defer conversation.mu.Unlock()
	request := append([]models.ChatMessage{{Role: "system", Content: a.system}}, conversation.messages...)
	return a.tokenReport(conversation, request, "", models.ModelUsage{})
}

func (a *Agent) tokenReport(conversation *conversation, request []models.ChatMessage, current string, usage models.ModelUsage) models.AgentTokenReport {
	// The system instruction is sent on every request, but it is agent
	// configuration rather than dialogue history. The displayed history is the
	// complete saved conversation, including the just-generated answer.
	history := conversation.messages
	report := models.AgentTokenReport{
		HistoryTokens: estimateDialogueHistoryTokens(history), CurrentMessageTokens: estimateMessageTokens(models.ChatMessage{Role: "user", Content: current}),
		EstimatedRequestTokens: estimateMessagesTokens(request), RequestTokens: usage.InputTokens, ResponseTokens: usage.OutputTokens,
		ContextLimitTokens: contextLimitTokens, ReservedOutputTokens: a.settings.MaxTokens, EstimateNote: estimateNote,
		CacheHitTokens: usage.CacheHitTokens, CacheMissTokens: usage.CacheMissTokens,
	}
	if current == "" {
		report.CurrentMessageTokens = 0
	}
	report.RemainingContextTokens = contextLimitTokens - report.EstimatedRequestTokens - report.ReservedOutputTokens
	if report.RemainingContextTokens < 0 {
		report.RemainingContextTokens = 0
	}
	for _, item := range conversation.usages {
		report.CumulativeInputTokens += item.InputTokens
		report.CumulativeOutputTokens += item.OutputTokens
		report.CumulativeCostUSD += usageCost(item)
	}
	report.EstimatedCostUSD = usageCost(usage)
	return report
}

// Clear removes one browser session from memory and durable storage.
func (a *Agent) Clear(sessionID string) error {
	a.persistMu.Lock()
	defer a.persistMu.Unlock()
	a.mu.Lock()
	previous, existed := a.sessions[sessionID]
	delete(a.sessions, sessionID)
	a.mu.Unlock()
	if err := a.save(); err != nil {
		if existed {
			a.mu.Lock()
			a.sessions[sessionID] = previous
			a.mu.Unlock()
		}
		return ErrHistorySave
	}
	return nil
}

func (a *Agent) conversation(sessionID string) *conversation {
	a.mu.Lock()
	defer a.mu.Unlock()
	if existing := a.sessions[sessionID]; existing != nil {
		return existing
	}
	created := &conversation{}
	a.sessions[sessionID] = created
	return created
}

func copyMessages(messages []models.ChatMessage) []models.ChatMessage {
	return append([]models.ChatMessage(nil), messages...)
}

func (a *Agent) save() error {
	if a.store == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	sessions := make(map[string][]models.ChatMessage, len(a.sessions))
	for id, conversation := range a.sessions {
		sessions[id] = copyMessages(conversation.messages)
	}
	return a.store.Save(sessions)
}
