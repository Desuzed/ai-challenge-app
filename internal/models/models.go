package models

type GenerationSettings struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"topP,omitempty"`
	MaxTokens   int      `json:"maxTokens"`
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
