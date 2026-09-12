package agent

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sync"

	"ai-challenge-app/internal/models"
)

type Store interface {
	Load() (map[string]ConversationState, error)
	Save(map[string]ConversationState) error
}

// ConversationState is stored per browser session. Summary is deliberately a
// separate field, never a fake dialogue turn.
type ConversationState struct {
	Messages        []models.ChatMessage `json:"messages"`
	Summary         string               `json:"summary,omitempty"`
	RecentMessages  int                  `json:"recentMessages,omitempty"`
	CompressedCount int                  `json:"compressedCount,omitempty"`
}

// JSONStore writes through a temporary file, so an interrupted write never
// replaces a valid history file.
type JSONStore struct {
	path string
	mu   sync.Mutex
}

func NewJSONStore(path string) *JSONStore { return &JSONStore{path: path} }

func (s *JSONStore) Load() (map[string]ConversationState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return make(map[string]ConversationState), nil
	}
	if err != nil {
		return nil, err
	}
	var sessions map[string]ConversationState
	if err := json.Unmarshal(data, &sessions); err != nil {
		// Day 7/8 used a map of raw message arrays. Keep that local history
		// readable when upgrading to the compressed format.
		var legacy map[string][]models.ChatMessage
		if legacyErr := json.Unmarshal(data, &legacy); legacyErr != nil {
			return nil, err
		}
		sessions = make(map[string]ConversationState, len(legacy))
		for id, messages := range legacy {
			sessions[id] = ConversationState{Messages: messages}
		}
	}
	if sessions == nil {
		sessions = make(map[string]ConversationState)
	}
	return copySessions(sessions), nil
}

func (s *JSONStore) Save(sessions map[string]ConversationState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := json.MarshalIndent(sessions, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(s.path), ".agent-history-*.tmp")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryName, s.path)
}

func copySessions(sessions map[string]ConversationState) map[string]ConversationState {
	result := make(map[string]ConversationState, len(sessions))
	for id, state := range sessions {
		state.Messages = copyMessages(state.Messages)
		result[id] = state
	}
	return result
}
