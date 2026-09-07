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

type completer interface {
	CompleteMessages(context.Context, []models.ChatMessage, models.GenerationSettings) (models.ModelCompletion, error)
}

type conversation struct {
	mu       sync.Mutex
	messages []models.ChatMessage
}

type Agent struct {
	client   completer
	system   string
	settings models.GenerationSettings
	mu       sync.Mutex
	sessions map[string]*conversation
}

func New(client completer) *Agent {
	temperature := 0.7
	return &Agent{
		client:   client,
		system:   "Ты полезный диалоговый агент. Отвечай точно, дружелюбно и по-русски. Не раскрывай скрытые внутренние рассуждения.",
		settings: models.GenerationSettings{Temperature: &temperature, MaxTokens: 512},
		sessions: make(map[string]*conversation),
	}
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

	conversation := a.conversation(sessionID)
	conversation.mu.Lock()
	defer conversation.mu.Unlock()
	conversation.messages = append(conversation.messages, models.ChatMessage{Role: "user", Content: message})
	request := make([]models.ChatMessage, 0, len(conversation.messages)+1)
	request = append(request, models.ChatMessage{Role: "system", Content: a.system})
	request = append(request, conversation.messages...)
	completion, err := a.client.CompleteMessages(ctx, request, a.settings)
	if err != nil {
		// Do not retain an unanswered user message: the next attempt is a clean retry.
		conversation.messages = conversation.messages[:len(conversation.messages)-1]
		return models.AgentResponse{}, err
	}
	conversation.messages = append(conversation.messages, models.ChatMessage{Role: "assistant", Content: completion.Answer})
	return models.AgentResponse{Answer: completion.Answer, Messages: copyMessages(conversation.messages)}, nil
}

func (a *Agent) History(sessionID string) []models.ChatMessage {
	conversation := a.conversation(sessionID)
	conversation.mu.Lock()
	defer conversation.mu.Unlock()
	return copyMessages(conversation.messages)
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
