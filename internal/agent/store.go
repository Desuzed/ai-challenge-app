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
	Load() (map[string][]models.ChatMessage, error)
	Save(map[string][]models.ChatMessage) error
}

// JSONStore writes through a temporary file, so an interrupted write never
// replaces a valid history file.
type JSONStore struct {
	path string
	mu   sync.Mutex
}

func NewJSONStore(path string) *JSONStore { return &JSONStore{path: path} }

func (s *JSONStore) Load() (map[string][]models.ChatMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return make(map[string][]models.ChatMessage), nil
	}
	if err != nil {
		return nil, err
	}
	var sessions map[string][]models.ChatMessage
	if err := json.Unmarshal(data, &sessions); err != nil {
		return nil, err
	}
	if sessions == nil {
		sessions = make(map[string][]models.ChatMessage)
	}
	return copySessions(sessions), nil
}

func (s *JSONStore) Save(sessions map[string][]models.ChatMessage) error {
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

func copySessions(sessions map[string][]models.ChatMessage) map[string][]models.ChatMessage {
	result := make(map[string][]models.ChatMessage, len(sessions))
	for id, messages := range sessions {
		result[id] = copyMessages(messages)
	}
	return result
}
