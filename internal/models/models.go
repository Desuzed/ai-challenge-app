package models

type GenerationSettings struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"topP,omitempty"`
	MaxTokens   int      `json:"maxTokens"`
}

// ChatMessage is a provider-neutral dialogue message. The agent owns the
// sequence of these messages; the API client only serializes it for the LLM.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type AgentRequest struct {
	Message string `json:"message"`
}

type AgentResponse struct {
	Answer   string        `json:"answer"`
	Messages []ChatMessage `json:"messages"`
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
	Answer       string
	FinishReason string
	Usage        ModelUsage
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
