package agent

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"ai-challenge-app/internal/models"
)

const taskMemoryMaxItems = 24
const taskMemoryMaxContextChars = 1200

func copyLegacyMiniChat(state MiniChatState) MiniChatState {
	state.Messages = copyMessages(state.Messages)
	state.Task = copyTaskMemory(state.Task)
	return state
}

// updateTaskMemory reads only the current user message. It keeps provenance in
// user-authored text and never infers durable facts from an assistant answer.
func updateTaskMemory(memory models.TaskMemory, message string, turn int) models.TaskMemory {
	trimmed := strings.TrimSpace(message)
	for _, rawLine := range strings.Split(trimmed, "\n") {
		line := strings.TrimSpace(rawLine)
		for _, field := range []string{"Цель:", "Уточнение:", "Ограничения:", "Термины:", "Термин:", "Отменить ограничение:"} {
			value, ok := prefixedValue(line, field)
			if !ok {
				continue
			}
			switch strings.ToLower(field) {
			case "цель:":
				memory.Goal, memory.GoalQuote, memory.GoalTurn = value, trimmed, turn
			case "уточнение:":
				memory.Clarifications = appendMemoryNote(memory.Clarifications, value, trimmed, turn)
			case "ограничения:":
				memory.Constraints = parseMemoryNotes(value, trimmed, turn)
			case "термины:":
				memory.Terms = parseMemoryNotes(value, trimmed, turn)
			case "термин:":
				memory.Terms = appendMemoryNote(memory.Terms, value, trimmed, turn)
			case "отменить ограничение:":
				memory.Constraints = removeMemoryConstraint(memory.Constraints, value)
			}
			break
		}
		lower := strings.ToLower(line)
		if strings.HasPrefix(lower, "ограничение больше не действует:") {
			value := strings.TrimSpace(line[len("ограничение больше не действует:"):])
			memory.Constraints = removeMemoryConstraint(memory.Constraints, value)
		}
	}
	if memory.Goal == "" && turn == 1 {
		memory.Goal, memory.GoalQuote, memory.GoalTurn = trimmed, trimmed, turn
	}
	memory.Clarifications = trimMemoryNotes(memory.Clarifications)
	memory.Constraints = trimMemoryNotes(memory.Constraints)
	memory.Terms = trimMemoryNotes(memory.Terms)
	return memory
}

func taskMemoryFromMessages(messages []models.ChatMessage) models.TaskMemory {
	var memory models.TaskMemory
	turn := 0
	for _, message := range messages {
		if message.Role != "user" {
			continue
		}
		turn++
		memory = updateTaskMemory(memory, message.Content, turn)
	}
	return memory
}

func removeMemoryConstraint(items []models.MemoryNote, requested string) []models.MemoryNote {
	needle := strings.ToLower(strings.TrimSpace(requested))
	filtered := make([]models.MemoryNote, 0, len(items))
	for _, item := range items {
		if needle != "" && (strings.Contains(strings.ToLower(item.Text), needle) || strings.Contains(needle, strings.ToLower(item.Text))) {
			continue
		}
		filtered = append(filtered, item)
	}
	return filtered
}

func prefixedValue(message, prefix string) (string, bool) {
	if len(message) < len(prefix) || !strings.EqualFold(message[:len(prefix)], prefix) {
		return "", false
	}
	return strings.TrimSpace(message[len(prefix):]), true
}

func parseMemoryNotes(value, quote string, turn int) []models.MemoryNote {
	parts := strings.FieldsFunc(value, func(r rune) bool { return r == ';' || r == '\n' })
	out := []models.MemoryNote{}
	for _, part := range parts {
		if v := strings.TrimSpace(part); v != "" {
			out = appendMemoryNote(out, v, quote, turn)
		}
	}
	return out
}

func appendMemoryNote(notes []models.MemoryNote, value, quote string, turn int) []models.MemoryNote {
	value = strings.TrimSpace(value)
	if value == "" || !strings.Contains(quote, value) {
		return notes
	}
	for i := range notes {
		if strings.EqualFold(memoryNoteKey(notes[i].Text), memoryNoteKey(value)) {
			notes[i] = models.MemoryNote{Text: value, SourceQuote: quote, Turn: turn}
			return notes
		}
	}
	return append(notes, models.MemoryNote{Text: value, SourceQuote: quote, Turn: turn})
}

func memoryNoteKey(value string) string {
	for _, separator := range []string{"=", ":", " — ", " – ", " - "} {
		if index := strings.Index(value, separator); index > 0 {
			return strings.ToLower(strings.TrimSpace(value[:index]))
		}
	}
	return strings.ToLower(strings.TrimSpace(value))
}

func trimMemoryNotes(notes []models.MemoryNote) []models.MemoryNote {
	if len(notes) > taskMemoryMaxItems {
		return append([]models.MemoryNote(nil), notes[len(notes)-taskMemoryMaxItems:]...)
	}
	return notes
}

func copyTaskMemory(memory models.TaskMemory) models.TaskMemory {
	memory.Clarifications = append([]models.MemoryNote(nil), memory.Clarifications...)
	memory.Constraints = append([]models.MemoryNote(nil), memory.Constraints...)
	memory.Terms = append([]models.MemoryNote(nil), memory.Terms...)
	return memory
}

func hasTaskMemory(memory models.TaskMemory) bool {
	return memory.Goal != "" || len(memory.Clarifications)+len(memory.Constraints)+len(memory.Terms) > 0
}

func taskMemoryPrompt(memory models.TaskMemory) string {
	return "Актуальная память задачи — только сведения, явно сказанные пользователем; номера ходов и цитаты показывают происхождение. Используй её как текущий источник намерений, ограничений и терминов. Для совпадающего пользовательского поля она имеет приоритет над старой перепиской, профилем, sticky facts, рабочей и долговременной памятью, а также над сохранённой спецификацией планировщика. Не меняй выбранный профиль, стратегию или настройки автоматически. Обязательные task/state/global invariants имеют приоритет над памятью пользователя; если они конфликтуют, прямо объясни конфликт и предложи совместимый вариант. Эти сведения не являются доказательствами о приложении или других внешних фактах.\n" + formatTaskMemory(memory)
}

func userTurnCount(messages []models.ChatMessage) int {
	count := 0
	for _, message := range messages {
		if message.Role == "user" {
			count++
		}
	}
	return count
}

func retrievalQuestion(message string) string {
	var question []string
	for _, rawLine := range strings.Split(message, "\n") {
		line := strings.TrimSpace(rawLine)
		lower := strings.ToLower(line)
		isMemoryLine := false
		for _, prefix := range []string{"цель:", "уточнение:", "уточнения:", "ограничение:", "ограничения:", "термин:", "термины:"} {
			if strings.HasPrefix(lower, prefix) {
				isMemoryLine = true
				break
			}
		}
		if line != "" && !isMemoryLine {
			question = append(question, line)
		}
	}
	if len(question) == 0 {
		return strings.TrimSpace(message)
	}
	joined := strings.Join(question, "\n")
	if detailedGroundedRequest(joined) {
		return retrievalTopicForDetailedRequest(joined)
	}
	return joined
}

func retrievalTopicForDetailedRequest(question string) string {
	query := strings.ToLower(question)
	for _, phrase := range []string{
		"подготовь итоговое объяснение", "подготовьте итоговое объяснение",
		"итоговое объяснение", "подробное итоговое объяснение", "подробного итогового объяснения",
		"для моего видео", "для видео", "для моего ролика",
		"учти актуальную просьбу о подробном ответе", "учти просьбу о подробном ответе",
		"о подробном ответе", "о подробного ответа", "и приложи источники", "приложи источники",
	} {
		query = strings.ReplaceAll(query, phrase, " ")
	}
	query = strings.NewReplacer(".", " ", ",", " ", ":", " ", ";", " ", "—", " ").Replace(query)
	return strings.Join(strings.Fields(query), " ")
}

func retrievalSupplementalQuery(message string, memory models.TaskMemory) string {
	words := strings.Fields(retrievalQuestion(message))
	if len(words) > 8 || !isShortAnaphoricFollowup(message) {
		return ""
	}
	context := []string{}
	if memory.Goal != "" {
		context = append(context, memory.Goal)
	}
	for _, item := range memory.Terms {
		context = append(context, item.Text)
	}
	joined := strings.Join(context, "; ")
	if utf8.RuneCountInString(joined) > taskMemoryMaxContextChars {
		joined = string([]rune(joined)[:taskMemoryMaxContextChars])
	}
	if joined == "" {
		return ""
	}
	return joined
}

func isShortAnaphoricFollowup(message string) bool {
	lower := strings.ToLower(strings.Join(strings.Fields(message), " "))
	for _, cue := range []string{"а это", "а этот", "а эта", "а там", "а тут", "а в нём", "а в нем", "а насчёт этого", "а насчет этого", "что насчёт этого", "что насчет этого", "где это", "как это"} {
		if strings.Contains(lower, cue) {
			return true
		}
	}
	return false
}

func formatTaskMemory(memory models.TaskMemory) string {
	var lines []string
	if memory.Goal != "" {
		lines = append(lines, fmt.Sprintf("Цель (ход %d): %s", memory.GoalTurn, memory.Goal))
	}
	for _, group := range []struct {
		name  string
		items []models.MemoryNote
	}{{"Уточнения", memory.Clarifications}, {"Ограничения", memory.Constraints}, {"Термины", memory.Terms}} {
		for _, note := range group.items {
			lines = append(lines, fmt.Sprintf("%s (ход %d): %s", group.name, note.Turn, note.Text))
		}
	}
	if len(lines) == 0 {
		return "пусто"
	}
	return strings.Join(lines, "\n")
}

func isTaskMemoryQuestion(message string) bool {
	lower := strings.ToLower(message)
	for _, cue := range []string{"вспомни", "напомни", "моя цель", "цель диалога", "мои ограничения", "актуальные ограничения", "значение слова", "согласованное значение", "какая аудитория", "мою аудиторию", "что я уже уточнил"} {
		if strings.Contains(lower, cue) {
			return true
		}
	}
	return false
}

func formatTaskMemorySources(memory models.TaskMemory) string {
	var lines []string
	if memory.Goal != "" {
		lines = append(lines, fmt.Sprintf("Цель: %s [память, ход %d]", memory.Goal, memory.GoalTurn))
	}
	for _, group := range []struct {
		label string
		items []models.MemoryNote
	}{{"Уточнение", memory.Clarifications}, {"Ограничение", memory.Constraints}, {"Термин", memory.Terms}} {
		for _, item := range group.items {
			lines = append(lines, fmt.Sprintf("%s: %s [память, ход %d]", group.label, item.Text, item.Turn))
		}
	}
	if len(lines) == 0 {
		return "Память задачи\nПодтверждённые пользовательские уточнения ещё не сохранены."
	}
	return "Память задачи и источники пользовательских сведений\n" + strings.Join(lines, "\n")
}
