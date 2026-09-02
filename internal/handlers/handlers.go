package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/models"
)

const (
	maxRequestBytes = 128 << 10
	defaultTokens   = 512
)

type completer interface {
	Complete(context.Context, string, models.GenerationSettings) (string, error)
}

type Handler struct {
	client completer
}

func New(client completer) *Handler {
	return &Handler{client: client}
}

func (h *Handler) Chat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "Используйте POST-запрос.", nil)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	defer r.Body.Close()
	var input models.ChatRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		writeError(w, http.StatusBadRequest, "Не удалось прочитать запрос.", nil)
		return
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		writeError(w, http.StatusBadRequest, "В запросе должен быть один JSON-объект.", nil)
		return
	}

	prompt := strings.TrimSpace(input.Prompt)
	if prompt == "" {
		writeError(w, http.StatusBadRequest, "Введите вопрос.", nil)
		return
	}
	if len(prompt) > 32000 {
		writeError(w, http.StatusBadRequest, "Вопрос слишком длинный.", nil)
		return
	}
	if err := validateSettings(&input.Settings); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}

	started := time.Now()
	debug := models.DebugInfo{Model: deepseek.ModelName(), PromptCharacters: utf8.RuneCountInString(prompt), Settings: input.Settings}
	ctx, cancel := context.WithTimeout(r.Context(), 50*time.Second)
	defer cancel()
	answer, err := h.client.Complete(ctx, prompt, input.Settings)
	debug.DurationMS = time.Since(started).Milliseconds()
	if err != nil {
		status, message := errorResponse(err)
		debug.HTTPStatus = status
		writeError(w, status, message, &debug)
		return
	}

	debug.HTTPStatus = http.StatusOK
	debug.AnswerCharacters = utf8.RuneCountInString(answer)
	writeJSON(w, http.StatusOK, models.ChatResponse{Answer: answer, Debug: debug})
}

func validateSettings(settings *models.GenerationSettings) error {
	if settings.MaxTokens == 0 {
		settings.MaxTokens = defaultTokens
	}
	if settings.MaxTokens < 1 || settings.MaxTokens > 8192 {
		return errors.New("Max tokens должен быть от 1 до 8192.")
	}
	if settings.Temperature != nil && (*settings.Temperature < 0 || *settings.Temperature > 2) {
		return errors.New("Temperature должен быть от 0 до 2.")
	}
	if settings.TopP != nil && (*settings.TopP < 0 || *settings.TopP > 1) {
		return errors.New("Top P должен быть от 0 до 1.")
	}
	if settings.Temperature != nil && settings.TopP != nil {
		return errors.New("Выберите temperature или top_p, но не оба параметра сразу.")
	}
	if settings.Temperature == nil && settings.TopP == nil {
		defaultTemperature := 0.7
		settings.Temperature = &defaultTemperature
	}
	return nil
}

func errorResponse(err error) (int, string) {
	switch {
	case errors.Is(err, deepseek.ErrNoAPIKey):
		return http.StatusServiceUnavailable, "API-ключ не настроен на сервере. Добавьте DEEPSEEK_API_KEY и перезапустите приложение."
	case errors.Is(err, deepseek.ErrUnauthorized):
		return http.StatusBadGateway, "DeepSeek не принял API-ключ. Проверьте его и перезапустите приложение."
	case errors.Is(err, deepseek.ErrRateLimited):
		return http.StatusTooManyRequests, "Слишком много запросов к DeepSeek. Повторите немного позже."
	default:
		return http.StatusBadGateway, "Не удалось получить ответ от DeepSeek. Попробуйте ещё раз."
	}
}

func writeError(w http.ResponseWriter, status int, message string, debug *models.DebugInfo) {
	writeJSON(w, status, models.ErrorResponse{Error: message, Debug: debug})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
