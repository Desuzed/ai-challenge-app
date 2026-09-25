package models

type GenerationSettings struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"topP,omitempty"`
	MaxTokens   int      `json:"maxTokens"`
}

const (
	DeepSeekFlashModel = "deepseek-flash"
	DeepSeekProModel   = "deepseek-v4-pro"
)

// ChatMessage is a provider-neutral dialogue message. The agent owns the
// sequence of these messages; the API client only serializes it for the LLM.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolDefinition and ToolCall mirror the provider-neutral subset shared by
// MCP tools and OpenAI-compatible chat-completion function calling.
type ToolDefinition struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ContextStrategy selects the information that reaches the model. None of the
// strategies uses a generated dialogue summary.
type ContextStrategy string

const (
	StrategySlidingWindow ContextStrategy = "sliding_window"
	StrategyFacts         ContextStrategy = "facts"
	StrategyBranching     ContextStrategy = "branching"
)

type Fact struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// MemoryLayer names a deliberately separate part of the agent state. Short
// term is the dialogue window; working and long-term memory are only changed
// by an explicit memory command from the user.
type MemoryLayer string

const (
	MemoryShortTerm MemoryLayer = "short_term"
	MemoryWorking   MemoryLayer = "working"
	MemoryLongTerm  MemoryLayer = "long_term"
)

// MemoryItem is a visible key-value record. Category is used in long-term
// memory to keep a person profile, confirmed decisions and reusable knowledge
// distinct from each other.
type MemoryItem struct {
	Layer    MemoryLayer `json:"layer"`
	Category string      `json:"category,omitempty"`
	Key      string      `json:"key"`
	Value    string      `json:"value"`
}

// Invariant is a mandatory rule that is kept outside the dialogue. Its scope
// determines ownership: task rules live with one task, global rules live with
// a user, and state rules constrain task transitions.
type Invariant struct {
	ID    string `json:"id"`
	Scope string `json:"scope"`
	Rule  string `json:"rule"`
}

// MemoryLayers is returned with every agent response, so it is always clear
// which information is in which layer and will reach the model.
type MemoryLayers struct {
	ShortTerm []ChatMessage `json:"shortTerm"`
	Working   []MemoryItem  `json:"working"`
	LongTerm  []MemoryItem  `json:"longTerm"`
}

// UserProfile is a named expert perspective for a response. Profiles are
// reusable by one user; a conversation only stores the selected profile ID.
type UserProfile struct {
	ID          string `json:"id,omitempty"`
	Name        string `json:"name,omitempty"`
	Perspective string `json:"perspective,omitempty"`
	Style       string `json:"style,omitempty"`
	Format      string `json:"format,omitempty"`
	Constraints string `json:"constraints,omitempty"`
}

type ConversationBranch struct {
	ID                 string `json:"id"`
	Name               string `json:"name"`
	ParentCheckpointID string `json:"parentCheckpointId,omitempty"`
	MessageCount       int    `json:"messageCount"`
}

type ConversationCheckpoint struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	BranchID     string `json:"branchId"`
	MessageCount int    `json:"messageCount"`
}

type AgentRequest struct {
	Message        string          `json:"message"`
	RecentMessages int             `json:"recentMessages"`
	Strategy       ContextStrategy `json:"strategy,omitempty"`
	Model          string          `json:"model,omitempty"`
}

// ContextCommand changes context mode or the active branch without asking the
// model. It keeps branch operations explicit and easy to inspect in the UI.
type ContextCommand struct {
	Action       string          `json:"action"`
	Strategy     ContextStrategy `json:"strategy,omitempty"`
	Name         string          `json:"name,omitempty"`
	CheckpointID string          `json:"checkpointId,omitempty"`
	BranchID     string          `json:"branchId,omitempty"`
	Layer        MemoryLayer     `json:"layer,omitempty"`
	Category     string          `json:"category,omitempty"`
	Key          string          `json:"key,omitempty"`
	Value        string          `json:"value,omitempty"`
	Model        string          `json:"model,omitempty"`
	Profile      UserProfile     `json:"profile,omitempty"`
	ProfileID    string          `json:"profileId,omitempty"`
	Task         TaskState       `json:"task,omitempty"`
	Phase        string          `json:"phase,omitempty"`
	Invariant    Invariant       `json:"invariant,omitempty"`
	PlannerMode  string          `json:"plannerMode,omitempty"`
}

// TaskStatus describes whether a task can currently progress. Pausing is
// deliberately independent from its phase, so a task can be paused anywhere
// in its workflow and later continued from the exact same state.
type TaskStatus string

const (
	TaskActive TaskStatus = "active"
	TaskPaused TaskStatus = "paused"
	TaskDone   TaskStatus = "done"
)

// TaskState is the explicit finite-state model for one task. Phases form an
// ordered, user-configurable workflow; PhaseIndex is the source of truth for
// allowed next/previous transitions. CurrentStep and ExpectedAction make the
// hand-off to a later conversation turn concrete instead of requiring the
// user to repeat prior context.
type TaskState struct {
	Goal            string      `json:"goal,omitempty"`
	Phases          []string    `json:"phases"`
	Phase           string      `json:"phase,omitempty"`
	PhaseIndex      int         `json:"phaseIndex"`
	CurrentStep     string      `json:"currentStep,omitempty"`
	ExpectedAction  string      `json:"expectedAction,omitempty"`
	Status          TaskStatus  `json:"status"`
	Specification   string      `json:"specification,omitempty"`
	OpenQuestions   []string    `json:"openQuestions,omitempty"`
	Decisions       []string    `json:"decisions,omitempty"`
	NextSteps       []string    `json:"nextSteps,omitempty"`
	ArtifactTitle   string      `json:"artifactTitle,omitempty"`
	ArtifactContent string      `json:"artifactContent,omitempty"`
	TaskInvariants  []Invariant `json:"taskInvariants,omitempty"`
	StateInvariants []Invariant `json:"stateInvariants,omitempty"`
	// RequireApprovalForTransition makes an automatic phase change impossible
	// until the current user message explicitly approves it.
	RequireApprovalForTransition bool `json:"requireApprovalForTransition,omitempty"`
	// Transition evidence is written only by the server-side lifecycle
	// commands. It is deliberately separate from a model response, so a model
	// cannot complete a task merely by claiming that an earlier stage is done.
	PlanApproved            bool `json:"planApproved,omitempty"`
	ImplementationCompleted bool `json:"implementationCompleted,omitempty"`
	ValidationPassed        bool `json:"validationPassed,omitempty"`
}

type AgentResponse struct {
	Answer           string                   `json:"answer"`
	Messages         []ChatMessage            `json:"messages"`
	RequestMessages  []ChatMessage            `json:"requestMessages,omitempty"`
	Tokens           AgentTokenReport         `json:"tokens"`
	Strategy         ContextStrategy          `json:"strategy"`
	Facts            []Fact                   `json:"facts,omitempty"`
	Memory           MemoryLayers             `json:"memory"`
	Profile          UserProfile              `json:"profile"`
	Profiles         []UserProfile            `json:"profiles,omitempty"`
	ActiveProfileID  string                   `json:"activeProfileId,omitempty"`
	ActiveBranchID   string                   `json:"activeBranchId,omitempty"`
	Branches         []ConversationBranch     `json:"branches,omitempty"`
	Checkpoints      []ConversationCheckpoint `json:"checkpoints,omitempty"`
	RecentMessages   int                      `json:"recentMessages"`
	Model            string                   `json:"model"`
	Task             TaskState                `json:"task"`
	PendingMessage   string                   `json:"pendingMessage,omitempty"`
	GlobalInvariants []Invariant              `json:"globalInvariants,omitempty"`
	PlannerMode      string                   `json:"plannerMode"`
	ToolExecutions   []ToolExecution          `json:"toolExecutions,omitempty"`
}

// AgentModelsResponse is the safe, key-free catalogue used by the chat UI.
type AgentModelsResponse struct {
	Models []string `json:"models"`
}

// AgentTokenReport makes the cost of carrying a dialogue visible. Values named
// "estimated" are calculated locally before a request; the other token counts
// come from the provider's usage object after a successful request.
type AgentTokenReport struct {
	HistoryTokens          int     `json:"historyTokens"`
	CurrentMessageTokens   int     `json:"currentMessageTokens"`
	EstimatedRequestTokens int     `json:"estimatedRequestTokens"`
	RequestTokens          int     `json:"requestTokens"`
	ResponseTokens         int     `json:"responseTokens"`
	ContextLimitTokens     int     `json:"contextLimitTokens"`
	ReservedOutputTokens   int     `json:"reservedOutputTokens"`
	RemainingContextTokens int     `json:"remainingContextTokens"`
	CumulativeInputTokens  int     `json:"cumulativeInputTokens"`
	CumulativeOutputTokens int     `json:"cumulativeOutputTokens"`
	EstimatedCostUSD       float64 `json:"estimatedCostUsd"`
	CumulativeCostUSD      float64 `json:"cumulativeCostUsd"`
	CacheHitTokens         int     `json:"cacheHitTokens"`
	CacheMissTokens        int     `json:"cacheMissTokens"`
	EstimateNote           string  `json:"estimateNote"`
	FullHistoryEstimate    int     `json:"fullHistoryEstimate"`
	CompressionSavedTokens int     `json:"compressionSavedTokens,omitempty"`
}

// ContextStrategyDemoResult contains three genuine provider completions over
// one fixed 15-message specification-gathering scenario.
type ContextStrategyDemoResult struct {
	Title    string                   `json:"title"`
	Scenario ContextStrategyScenario  `json:"scenario"`
	Runs     []ContextStrategyDemoRun `json:"runs"`
}

// ContextStrategyScenario is the complete read-only input used by the
// educational comparison. GET returns it without a model invocation.
type ContextStrategyScenario struct {
	Title        string        `json:"title"`
	SystemPrompt string        `json:"systemPrompt"`
	Messages     []ChatMessage `json:"messages"`
	Facts        []Fact        `json:"facts"`
	WindowSize   int           `json:"windowSize"`
}

type ContextStrategyDemoRun struct {
	Strategy         ContextStrategy `json:"strategy"`
	Title            string          `json:"title"`
	Answer           string          `json:"answer"`
	InputTokens      int             `json:"inputTokens"`
	OutputTokens     int             `json:"outputTokens"`
	PromptPreview    string          `json:"promptPreview"`
	ContextNote      string          `json:"contextNote"`
	RetainedFacts    []Fact          `json:"retainedFacts,omitempty"`
	IncludedMessages int             `json:"includedMessages"`
}

// BranchingDemoScenario shows a shared checkpoint and two independent
// continuations. It is read-only until the user runs the comparison.
type BranchingDemoScenario struct {
	Title        string                `json:"title"`
	SystemPrompt string                `json:"systemPrompt"`
	Checkpoint   []ChatMessage         `json:"checkpoint"`
	Branches     []BranchingDemoBranch `json:"branches"`
}

type BranchingDemoBranch struct {
	ID       string        `json:"id"`
	Title    string        `json:"title"`
	Messages []ChatMessage `json:"messages"`
}

type BranchingDemoResult struct {
	Title    string                `json:"title"`
	Scenario BranchingDemoScenario `json:"scenario"`
	Runs     []BranchingDemoRun    `json:"runs"`
}

type BranchingDemoRun struct {
	BranchID           string `json:"branchId"`
	Title              string `json:"title"`
	Answer             string `json:"answer"`
	InputTokens        int    `json:"inputTokens"`
	OutputTokens       int    `json:"outputTokens"`
	PromptPreview      string `json:"promptPreview"`
	CheckpointMessages int    `json:"checkpointMessages"`
	BranchMessages     int    `json:"branchMessages"`
}

type TokenDemoRequest struct {
	Scenario       string `json:"scenario"`
	ForceCacheMiss bool   `json:"forceCacheMiss"`
}

// TokenDemoResult is an isolated, repeatable demonstration. It never changes
// the browser chat history.
type TokenDemoResult struct {
	Scenario         string  `json:"scenario"`
	Title            string  `json:"title"`
	Answer           string  `json:"answer,omitempty"`
	Blocked          bool    `json:"blocked"`
	InputTokens      int     `json:"inputTokens"`
	OutputTokens     int     `json:"outputTokens"`
	TotalTokens      int     `json:"totalTokens"`
	CacheHitTokens   int     `json:"cacheHitTokens"`
	CacheMissTokens  int     `json:"cacheMissTokens"`
	EstimatedCostUSD float64 `json:"estimatedCostUsd"`
	ForceCacheMiss   bool    `json:"forceCacheMiss"`
	Explanation      string  `json:"explanation"`
	PromptPreview    string  `json:"promptPreview"`
}

type ResponseMode string

const StopSequence = "<<END>>"

const (
	ModeUnrestricted ResponseMode = "unrestricted"
	ModeFormat       ResponseMode = "format"
	ModeLength       ResponseMode = "length"
	ModeFinish       ResponseMode = "finish"
	ModeAll          ResponseMode = "all"
)

type ChatRequest struct {
	Prompt   string             `json:"prompt"`
	Mode     ResponseMode       `json:"mode"`
	Settings GenerationSettings `json:"settings"`
}

type DebugInfo struct {
	Model            string             `json:"model"`
	PromptCharacters int                `json:"promptCharacters"`
	Settings         GenerationSettings `json:"settings"`
	HTTPStatus       int                `json:"httpStatus"`
	DurationMS       int64              `json:"durationMs"`
	AnswerCharacters int                `json:"answerCharacters,omitempty"`
	Mode             ResponseMode       `json:"mode"`
	FinishReason     string             `json:"finishReason,omitempty"`
	StopSequence     string             `json:"stopSequence,omitempty"`
}

type ChatResponse struct {
	Answer string    `json:"answer"`
	Debug  DebugInfo `json:"debug"`
}

type ErrorResponse struct {
	Error string     `json:"error"`
	Debug *DebugInfo `json:"debug,omitempty"`
}

// ReasoningApproach is one of the four comparable ways of solving the same task
// in the third lesson.
type ReasoningApproach string

const (
	ReasoningDirect         ReasoningApproach = "direct"
	ReasoningStepByStep     ReasoningApproach = "step_by_step"
	ReasoningPromptDesigner ReasoningApproach = "prompt_designer"
	ReasoningExpertPanel    ReasoningApproach = "expert_panel"
)

type ReasoningRequest struct {
	Task     string             `json:"task"`
	Approach ReasoningApproach  `json:"approach"`
	Settings GenerationSettings `json:"settings"`
}

type ReasoningDebugInfo struct {
	Model            string             `json:"model"`
	Settings         GenerationSettings `json:"settings"`
	HTTPStatus       int                `json:"httpStatus"`
	DurationMS       int64              `json:"durationMs"`
	TaskCharacters   int                `json:"taskCharacters"`
	AnswerCharacters int                `json:"answerCharacters,omitempty"`
	Requests         int                `json:"requests"`
	FinishReasons    []string           `json:"finishReasons,omitempty"`
}

type ReasoningResponse struct {
	Approach       ReasoningApproach  `json:"approach"`
	Answer         string             `json:"answer"`
	PreparedPrompt string             `json:"preparedPrompt,omitempty"`
	Debug          ReasoningDebugInfo `json:"debug"`
}

// ModelVersionRequest deliberately has no prompt field: the lesson always uses
// the fixed incident prompt below, so the compared requests remain identical.
type ModelVersionRequest struct{}

type ModelUsage struct {
	InputTokens     int `json:"inputTokens"`
	OutputTokens    int `json:"outputTokens"`
	TotalTokens     int `json:"totalTokens"`
	CacheHitTokens  int `json:"cacheHitTokens,omitempty"`
	CacheMissTokens int `json:"cacheMissTokens,omitempty"`
}

type ModelCompletion struct {
	Answer         string
	FinishReason   string
	Usage          ModelUsage
	ToolCalls      []ToolCall
	ToolExecutions []ToolExecution
}

type ToolExecution struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
	Result    string         `json:"result,omitempty"`
	IsError   bool           `json:"isError"`
}

type ModelVersionRun struct {
	Provider     string     `json:"provider"`
	Model        string     `json:"model"`
	Level        string     `json:"level"`
	Supported    bool       `json:"supported"`
	Answer       string     `json:"answer,omitempty"`
	Error        string     `json:"error,omitempty"`
	DurationMS   int64      `json:"durationMs,omitempty"`
	Usage        ModelUsage `json:"usage"`
	CostUSD      *float64   `json:"costUsd,omitempty"`
	CostNote     string     `json:"costNote"`
	FinishReason string     `json:"finishReason,omitempty"`
}

type ModelVersionsResponse struct {
	Prompt      string            `json:"prompt"`
	Catalog     []string          `json:"catalog"`
	CatalogNote string            `json:"catalogNote"`
	Runs        []ModelVersionRun `json:"runs"`
	Sources     []SourceLink      `json:"sources"`
}

type SourceLink struct {
	Title string `json:"title"`
	URL   string `json:"url"`
}
