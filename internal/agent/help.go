package agent

import (
	"context"
	"errors"
	"strings"
	"time"
	"unicode/utf8"

	"ai-challenge-app/internal/models"
)

// Help uses the ordinary provider, history and citation validator, but never
// calls MCP tools or advances the planner. Live git context belongs to /diff.
func (a *Agent) respondProjectHelpLocked(ctx context.Context, c *conversation, u *userState, message, question string, options models.RAGOptions) (models.AgentResponse, error) {
	if a.retriever == nil {
		return models.AgentResponse{}, errors.New("Поиск по документам не настроен.")
	}
	searchOptions := ragOptionsForRequest(options)
	searchOptions.DocumentationOnly = true
	searchOptions.ProjectOverview = projectOverviewQuestion(question)
	searchCtx, cancelSearch := context.WithTimeout(ctx, 30*time.Second)
	result, err := a.retriever.SearchWithOptions(searchCtx, question, searchOptions)
	cancelSearch()
	if err != nil {
		return models.AgentResponse{}, err
	}
	if c.task.Status == models.TaskPaused {
		return a.respondWhileTaskPausedLocked(c, u, message, true, result, options)
	}
	previous := copyMessages(c.activeMessagesLocked())
	c.setActiveMessagesLocked(append(previous, models.ChatMessage{Role: "user", Content: message}))
	request := a.requestMessagesForModeLocked(c, u, false)
	// Keep the chosen system/profile rules, but prevent older failed commands
	// and unrelated assistant refusals from becoming examples for help.
	var focused []models.ChatMessage
	for _, item := range request {
		if item.Role == "system" {
			focused = append(focused, item)
		}
	}
	request = append(focused, models.ChatMessage{Role: "user", Content: question})
	settings := a.settings
	if c.hasGenerationSettings {
		settings = c.generationSettings
	}
	compactLocal := models.IsOllamaModel(c.model) && !models.IsRemoteOllamaModel(c.model) && c.hasGenerationSettings && c.generationSettings.MaxTokens <= 1024
	request = addGroundedRAGContext(request, result.Matches, false, compactLocal)
	helpContract := "Объясни проект по найденной документации. Верни компактный полный JSON с claims и точными цитатами. Поле answer оставь пустым. Дай не более трёх коротких claims без повторов. Каждое утверждение оформи отдельным claim с короткой дословной цитатой из одного чанка; не объединяй разные строки таблицы в одну цитату. Не выполняй действия и не составляй план задачи. Документальные claims должны ссылаться только на чанки документации. Текущая git-ветка здесь не запрашивается: не добавляй её в ответ. Для просмотра фактической git-ветки используется команда /diff."
	if settings.MaxTokens > 0 && settings.MaxTokens <= 768 {
		helpContract += " Бюджет ответа мал: верни ровно один короткий claim, если ответ подтверждён. Текст утверждения — одно предложение до 180 символов; одна точная цитата до 120 символов. Не добавляй другие поля, пояснения и отступы. При отсутствии ответа верни claims:[]."
	}
	request = append([]models.ChatMessage{{Role: "system", Content: helpContract}}, request...)
	skipped := len(result.Matches) == 0
	if !models.IsRemoteOllamaModel(c.model) {
		if settings.MaxTokens == 0 {
			settings.MaxTokens = groundedOutputBudget(question)
		} else {
			settings.MaxTokens = min(settings.MaxTokens, groundedOutputBudget(question))
		}
	}
	budget := settings.MaxTokens
	report := reserveOutputTokens(a.tokenReportLocked(c, request, message, models.ModelUsage{}), budget)
	if skipped {
		report = reserveOutputTokens(a.tokenReportLocked(c, nil, message, models.ModelUsage{}), 0)
	}
	if report.ContextLimitTokens > 0 && report.EstimatedRequestTokens+report.ReservedOutputTokens > report.ContextLimitTokens {
		c.setActiveMessagesLocked(previous)
		return models.AgentResponse{}, ErrContextLimit
	}
	completion := models.ModelCompletion{Answer: `{"claims":[]}`}
	if !skipped {
		workCtx := c.startWork(ctx)
		completion, err = a.completeLocked(workCtx, c.model, request, false, false, question, false, &settings, true)
		if err == nil {
			firstEnvelope, _, firstParseErr := parseGroundedEnvelope(completion.Answer)
			firstClaims := validateGroundedEnvelopeClaims(firstEnvelope, result.Matches)
			retryReason := ""
			if firstParseErr != nil {
				retryReason = "invalid_structured_output"
				if isTruncatedFinishReason(completion.FinishReason) {
					retryReason = "structured_output_truncated"
				}
			} else if len(firstClaims) == 0 {
				retryReason = "model_no_claims"
				if len(firstEnvelope.Claims) > 0 {
					retryReason = "invalid_evidence"
				}
			} else if len(firstClaims) != len(firstEnvelope.Claims) {
				retryReason = "invalid_evidence"
			}
			if completion.CompletionAttempts == 0 {
				completion.CompletionAttempts = 1
			}
			if retryReason != "" && completion.CompletionAttempts < 2 {
				repair := models.ChatMessage{Role: "system", Content: "Предыдущая генерация не прошла проверку (" + retryReason + "). " + `Сформируй ответ заново: один полный JSON {"answer":"","claims":[{"text":"...","evidence":[{"chunkId":"ID из найденного фрагмента","quote":"короткая точная подстрока этого фрагмента"}]}]}. Не пересказывай цитату и не меняй её слова. Не более трёх коротких утверждений. Если источник не подтверждает ответ, верни claims:[]; не придумывай доказательства.`}
				retryRequest := append([]models.ChatMessage{}, request[:len(request)-1]...)
				retryRequest = append(retryRequest, repair, request[len(request)-1])
				estimate := reserveOutputTokens(a.tokenReportLocked(c, retryRequest, message, models.ModelUsage{}), budget)
				if estimate.ContextLimitTokens <= 0 || estimate.EstimatedRequestTokens+estimate.ReservedOutputTokens <= estimate.ContextLimitTokens {
					retry, retryErr := a.completeLocked(workCtx, c.model, retryRequest, false, false, question, false, &settings, true)
					if retryErr != nil {
						err = retryErr
					}
					if retryErr == nil {
						retryEnvelope, _, retryParseErr := parseGroundedEnvelope(retry.Answer)
						retryClaims := validateGroundedEnvelopeClaims(retryEnvelope, result.Matches)
						usage := addUsage(completion.Usage, retry.Usage)
						metrics := helpMetrics(completion.LocalMetrics, retry.LocalMetrics, usage)
						if retryParseErr == nil && len(retryClaims) >= len(firstClaims) {
							completion = retry
							request = retryRequest
						}
						completion.Usage = usage
						completion.LocalMetrics = metrics
						completion.CompletionAttempts = 2
						completion.CompletionRetryReason = retryReason
					}
				}
			}
		}
		if c.finishWork() {
			return a.respondAfterActivePauseLocked(c, u, previous, copyFactsMap(c.facts), message, true, result, options)
		}
		if err != nil {
			c.setActiveMessagesLocked(previous)
			return models.AgentResponse{}, err
		}
	}

	envelope, _, parseErr := parseGroundedEnvelope(completion.Answer)
	claims := validateGroundedEnvelopeClaims(envelope, result.Matches)
	sources := sourcesForGroundedClaims(claims, result.Matches)
	answer := formatGroundedAnswer(envelope.Answer, claims, sources, false, false, detailedGroundedRequest(question))
	abstained := parseErr != nil || len(claims) == 0
	reason := ""
	if abstained {
		answer = "Найденные фрагменты документации не подтверждают ответ на этот вопрос. Уточните формулировку."
		reason = "model_no_claims"
		if skipped {
			reason = "retrieval_empty"
			answer = "По этому вопросу подходящие фрагменты документации не найдены. Уточните вопрос или проверьте настройки поиска."
		}
		if parseErr != nil {
			reason = "invalid_structured_output"
			answer = "Документы найдены, но модель вернула некорректный структурированный ответ. Повторите вопрос."
			if isTruncatedFinishReason(completion.FinishReason) {
				reason = "structured_output_truncated"
				answer = "Документы найдены, но ответ модели обрезан до завершения JSON. Увеличьте лимит ответа или попросите краткий ответ."
			}
		}
		if parseErr == nil && len(envelope.Claims) > 0 {
			reason = "invalid_evidence"
			answer = "Документы найдены, но модель не подтвердила ответ точными цитатами. Непроверенные утверждения не показаны. Попробуйте уточнить вопрос."
		}
	}
	c.setActiveMessagesLocked(append(c.activeMessagesLocked(), models.ChatMessage{Role: "assistant", Content: answer}))
	c.lastFinishReason = completion.FinishReason
	c.appendUsageLocked(completion.Usage)
	c.trimWindowLocked()
	if err := a.save(); err != nil {
		c.setActiveMessagesLocked(previous)
		c.removeLastUsageLocked()
		return models.AgentResponse{}, ErrHistorySave
	}
	if skipped {
		request = nil
	}
	report = a.tokenReportLocked(c, request, message, completion.Usage)
	if completion.OutputTokenBudget > 0 {
		budget = completion.OutputTokenBudget
	}
	if skipped {
		budget = 0
	}
	report = reserveOutputTokens(report, budget)
	response := a.responseLocked(c, u, answer, request, report)
	response.RAGEnabled, response.RAGAbstained, response.RAGReason = true, abstained, reason
	response.RAGSources, response.RAGModelCallSkipped = nonNilSources(sources), skipped
	response.LocalMetrics = completion.LocalMetrics
	response.RAGTrace = ragTraceForResult(result, options)
	response.RAGTrace.ClaimedCount, response.RAGTrace.VerifiedClaimCount = len(envelope.Claims), len(claims)
	response.RAGTrace.StructuredOutput = !skipped && supportsStructuredOutput(a.client)
	response.RAGTrace.OutputTokenBudget = budget
	response.RAGTrace.CompletionFinishReason = completion.FinishReason
	response.RAGTrace.CompletionCharacters = utf8.RuneCountInString(completion.Answer)
	response.RAGTrace.CompletionAttempts = completion.CompletionAttempts
	response.RAGTrace.CompletionRetryReason = completion.CompletionRetryReason
	return response, nil
}

func projectOverviewQuestion(question string) bool {
	question = strings.Trim(strings.Join(strings.Fields(strings.ToLower(question)), " "), ".!?, ")
	switch question {
	case "расскажи о проекте", "расскажи про проект", "расскажи о приложении", "что это за проект", "что за проект", "что это за приложение", "опиши проект", "описание проекта", "обзор проекта", "для чего нужен проект", "что умеет проект":
		return true
	default:
		return false
	}
}

func helpMetrics(first, second *models.LocalMetrics, usage models.ModelUsage) *models.LocalMetrics {
	if first == nil {
		return second
	}
	if second == nil {
		return first
	}
	result := &models.LocalMetrics{TotalMS: first.TotalMS + second.TotalMS, LoadMS: first.LoadMS + second.LoadMS, PromptMS: first.PromptMS + second.PromptMS, GenerationMS: first.GenerationMS + second.GenerationMS}
	if result.GenerationMS > 0 {
		result.TokensPerSecond = float64(usage.OutputTokens) * 1000 / result.GenerationMS
	}
	return result
}
