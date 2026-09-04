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
