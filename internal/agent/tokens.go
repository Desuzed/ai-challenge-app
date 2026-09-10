package agent

import (
	"math"
	"unicode/utf8"

	"ai-challenge-app/internal/models"
)

const (
	// DeepSeek v4 Flash has a 1M-token context window. We reserve the requested
	// output budget as it occupies the same context window as the prompt.
	contextLimitTokens = 1_000_000
	inputMissUSDPerM   = 0.44
	inputHitUSDPerM    = 0.014
	outputUSDPerM      = 1.32
)

const estimateNote = "До ответа это консервативная оценка по символам; после ответа request/response — точные значения из usage API. Стоимость рассчитана по peak-тарифу DeepSeek V4 Flash и учитывает cache hit/miss."

func estimateMessageTokens(message models.ChatMessage) int {
	// Four service tokens approximate role/message framing. One token per three
	// Unicode symbols is deliberately conservative for mixed Russian text,
	// punctuation and code; only the API tokenizer can provide an exact count.
	return 4 + int(math.Ceil(float64(utf8.RuneCountInString(message.Content))/3))
}

func estimateMessagesTokens(messages []models.ChatMessage) int {
	tokens := 2 // assistant priming
	for _, message := range messages {
		tokens += estimateMessageTokens(message)
	}
	return tokens
}

func estimateDialogueHistoryTokens(messages []models.ChatMessage) int {
	if len(messages) == 0 {
		return 0
	}
	return estimateMessagesTokens(messages)
}

func usageCost(usage models.ModelUsage) float64 {
	hit := usage.CacheHitTokens
	miss := usage.CacheMissTokens
	if miss == 0 {
		miss = usage.InputTokens - hit
	}
	if miss < 0 {
		miss = 0
	}
	return float64(hit)*inputHitUSDPerM/1_000_000 + float64(miss)*inputMissUSDPerM/1_000_000 + float64(usage.OutputTokens)*outputUSDPerM/1_000_000
}
