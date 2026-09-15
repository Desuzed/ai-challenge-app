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

// ConversationState is stored per browser session. Facts, checkpoints and
// branches remain structured state; no generated summary is persisted.
type ConversationState struct {
	Strategy       models.ContextStrategy `json:"strategy,omitempty"`
	Messages       []models.ChatMessage   `json:"messages"`
	Facts          map[string]string      `json:"facts,omitempty"`
	Usages         []models.ModelUsage    `json:"usages,omitempty"`
	RecentMessages int                    `json:"recentMessages,omitempty"`
	Branches       []BranchState          `json:"branches,omitempty"`
	ActiveBranchID string                 `json:"activeBranchId,omitempty"`
	Checkpoints    []CheckpointState      `json:"checkpoints,omitempty"`
	NextBranch     int                    `json:"nextBranch,omitempty"`
	NextCheckpoint int                    `json:"nextCheckpoint,omitempty"`
	NextMemoryItem int                    `json:"nextMemoryItem,omitempty"`
	Model          string                 `json:"model,omitempty"`
	// These fields intentionally do not share storage with Messages or Facts.
	// They make the three memory layers inspectable in the JSON file as well as
	// in the API response.
	WorkingMemory  map[string]string            `json:"workingMemory,omitempty"`
	LongTermMemory map[string]map[string]string `json:"longTermMemory,omitempty"`
}

type BranchState struct {
	ID                 string               `json:"id"`
	Name               string               `json:"name"`
	ParentCheckpointID string               `json:"parentCheckpointId,omitempty"`
	Messages           []models.ChatMessage `json:"messages"`
	Usages             []models.ModelUsage  `json:"usages,omitempty"`
}

type CheckpointState struct {
	ID       string               `json:"id"`
	Name     string               `json:"name"`
	BranchID string               `json:"branchId"`
	Messages []models.ChatMessage `json:"messages"`
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
		// readable when upgrading to structured context state.
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
		state.Facts = copyFactsMap(state.Facts)
		state.WorkingMemory = copyFactsMap(state.WorkingMemory)
		state.LongTermMemory = copyLongTermMemory(state.LongTermMemory)
		state.Usages = append([]models.ModelUsage(nil), state.Usages...)
		for i := range state.Branches {
			state.Branches[i].Messages = copyMessages(state.Branches[i].Messages)
			state.Branches[i].Usages = append([]models.ModelUsage(nil), state.Branches[i].Usages...)
		}
		for i := range state.Checkpoints {
			state.Checkpoints[i].Messages = copyMessages(state.Checkpoints[i].Messages)
		}
		result[id] = state
	}
	return result
}

func copyLongTermMemory(memory map[string]map[string]string) map[string]map[string]string {
	result := make(map[string]map[string]string, len(memory))
	for category, values := range memory {
		result[category] = copyFactsMap(values)
	}
	return result
}
