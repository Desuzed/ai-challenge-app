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
	Load() (PersistentState, error)
	Save(PersistentState) error
}

// PersistentState separates per-session conversations from data that belongs
// to a user across all of their sessions.
type PersistentState struct {
	Sessions map[string]ConversationState `json:"sessions"`
	Users    map[string]UserState         `json:"users"`
}

type UserState struct {
	Profiles       map[string]models.UserProfile `json:"profiles,omitempty"`
	LongTermMemory map[string]map[string]string  `json:"longTermMemory,omitempty"`
	NextProfile    int                           `json:"nextProfile,omitempty"`
}

// ConversationState is stored per browser session. Facts, checkpoints and
// branches remain structured state; no generated summary is persisted.
type ConversationState struct {
	Strategy        models.ContextStrategy `json:"strategy,omitempty"`
	Messages        []models.ChatMessage   `json:"messages"`
	Facts           map[string]string      `json:"facts,omitempty"`
	Usages          []models.ModelUsage    `json:"usages,omitempty"`
	RecentMessages  int                    `json:"recentMessages,omitempty"`
	Branches        []BranchState          `json:"branches,omitempty"`
	ActiveBranchID  string                 `json:"activeBranchId,omitempty"`
	Checkpoints     []CheckpointState      `json:"checkpoints,omitempty"`
	NextBranch      int                    `json:"nextBranch,omitempty"`
	NextCheckpoint  int                    `json:"nextCheckpoint,omitempty"`
	NextMemoryItem  int                    `json:"nextMemoryItem,omitempty"`
	Model           string                 `json:"model,omitempty"`
	UserID          string                 `json:"userId,omitempty"`
	ActiveProfileID string                 `json:"activeProfileId,omitempty"`
	// Profile and LongTermMemory are retained only to migrate the previous
	// single-profile, per-session format when it is read.
	Profile models.UserProfile `json:"profile,omitempty"`
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

func (s *JSONStore) Load() (PersistentState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return PersistentState{Sessions: make(map[string]ConversationState), Users: make(map[string]UserState)}, nil
	}
	if err != nil {
		return PersistentState{}, err
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return PersistentState{}, err
	}
	if _, ok := root["sessions"]; ok {
		var state PersistentState
		if err := json.Unmarshal(data, &state); err != nil {
			return PersistentState{}, err
		}
		return copyPersistentState(state), nil
	}
	// Earlier versions wrote a map keyed by session ID. Keep that history
	// readable, then migrate its long-term records to the matching user ID.
	var sessions map[string]ConversationState
	if err := json.Unmarshal(data, &sessions); err == nil {
		state := PersistentState{Sessions: sessions, Users: make(map[string]UserState)}
		for id, session := range sessions {
			user := UserState{Profiles: make(map[string]models.UserProfile), LongTermMemory: copyLongTermMemory(session.LongTermMemory)}
			if session.Profile.Name != "" || session.Profile.Style != "" || session.Profile.Format != "" || session.Profile.Constraints != "" {
				session.Profile.ID = "profile-1"
				user.Profiles[session.Profile.ID] = session.Profile
				user.NextProfile = 1
				session.ActiveProfileID = session.Profile.ID
			}
			session.UserID = id
			state.Sessions[id] = session
			state.Users[id] = user
		}
		return copyPersistentState(state), nil
	}
	var legacy map[string][]models.ChatMessage
	if err := json.Unmarshal(data, &legacy); err != nil {
		return PersistentState{}, err
	}
	state := PersistentState{Sessions: make(map[string]ConversationState, len(legacy)), Users: make(map[string]UserState)}
	for id, messages := range legacy {
		state.Sessions[id] = ConversationState{Messages: messages, UserID: id}
		state.Users[id] = UserState{Profiles: make(map[string]models.UserProfile), LongTermMemory: make(map[string]map[string]string)}
	}
	return state, nil
}

func (s *JSONStore) Save(state PersistentState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := json.MarshalIndent(copyPersistentState(state), "", "  ")
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

func copyPersistentState(state PersistentState) PersistentState {
	if state.Sessions == nil {
		state.Sessions = make(map[string]ConversationState)
	}
	if state.Users == nil {
		state.Users = make(map[string]UserState)
	}
	state.Sessions = copySessions(state.Sessions)
	users := make(map[string]UserState, len(state.Users))
	for id, user := range state.Users {
		profiles := make(map[string]models.UserProfile, len(user.Profiles))
		for profileID, profile := range user.Profiles {
			profiles[profileID] = profile
		}
		user.Profiles = profiles
		user.LongTermMemory = copyLongTermMemory(user.LongTermMemory)
		users[id] = user
	}
	state.Users = users
	return state
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
