package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/projectmcp"
)

const helpInstructions = "Задайте вопрос о проекте в команде: /help <вопрос>. Например: /help Как устроен проект? или /help Где хранится история? Помощь отвечает только на этот вызов; обычные сообщения остаются обычным чатом. /structure показывает структуру проекта, /diff — текущую git-ветку через MCP, /reindex пересобирает индексы документов."

func (a *Agent) projectCommandLocked(ctx context.Context, c *conversation, u *userState, message string, options models.RAGOptions) (response models.AgentResponse, handled bool, err error) {
	parts := strings.Fields(message)
	if len(parts) > 0 && strings.HasPrefix(parts[0], "/") {
		switch parts[0] {
		case "/help":
			question := strings.TrimSpace(strings.TrimPrefix(message, "/help"))
			if question == "" || question == "off" {
				response, err = a.commandReplyLocked(c, u, message, helpInstructions)
				return response, true, err
			}
			response, err = a.respondProjectHelpLocked(ctx, c, u, message, question, options)
			return response, true, err
		case "/structure":
			response, err = a.respondProjectHelpLocked(ctx, c, u, message, "За что отвечают каталоги internal, cmd и static?", options)
			return response, true, err
		case "/diff":
			if len(parts) != 1 {
				response, err = a.commandReplyLocked(c, u, message, "Используйте /diff без аргументов — команда покажет текущую git-ветку через MCP.")
				return response, true, err
			}
			git, execution, gitErr := a.projectGitContext(ctx)
			if gitErr != nil {
				response, err = a.commandReplyLocked(c, u, message, "Не удалось получить текущую git-ветку: "+gitErr.Error())
				return response, true, err
			}
			response, err = a.commandReplyLocked(c, u, message, gitBranchLine(git))
			response.ToolExecutions = []models.ToolExecution{execution}
			response.RAGSources = nonNilSources(sourcesForToolExecutions(response.ToolExecutions))
			return response, true, err
		case "/reindex":
			if len(parts) != 1 {
				response, err = a.commandReplyLocked(c, u, message, "Используйте /reindex без аргументов — команда пересоберёт индексы README, docs и выбранных исходников.")
				return response, true, err
			}
			response, err = a.reindexLocked(ctx, c, u, message)
			return response, true, err
		default:
			response, err = a.commandReplyLocked(c, u, message, fmt.Sprintf("Команда %s не реализована. Доступны /help <вопрос>, /structure, /diff и /reindex. /help показывает инструкцию.", parts[0]))
			return response, true, err
		}
	}
	return models.AgentResponse{}, false, nil
}

func (a *Agent) commandReplyLocked(c *conversation, u *userState, message, answer string) (models.AgentResponse, error) {
	previous := copyMessages(c.activeMessagesLocked())
	previousFinish := c.lastFinishReason
	c.setActiveMessagesLocked(append(previous, models.ChatMessage{Role: "user", Content: message}, models.ChatMessage{Role: "assistant", Content: answer}))
	c.lastFinishReason = "command"
	if err := a.save(); err != nil {
		c.setActiveMessagesLocked(previous)
		c.lastFinishReason = previousFinish
		return models.AgentResponse{}, ErrHistorySave
	}
	response := a.responseLocked(c, u, answer, nil, a.tokenReportLocked(c, nil, message, models.ModelUsage{}))
	response.RAGModelCallSkipped = true
	return response, nil
}

func (a *Agent) reindexLocked(ctx context.Context, c *conversation, u *userState, message string) (models.AgentResponse, error) {
	if a.indexer == nil {
		return a.commandReplyLocked(c, u, message, "Индексация не настроена на сервере. /reindex не выполнена.")
	}
	started := time.Now()
	report, err := a.indexer.Rebuild(ctx)
	if err != nil {
		return a.commandReplyLocked(c, u, message, "Индексация не выполнена: "+err.Error())
	}
	summary := &models.ReindexSummary{Files: len(report.CorpusFiles), EmbeddingModel: report.Model, DurationMS: time.Since(started).Milliseconds()}
	for _, s := range report.Strategies {
		if s.Strategy == "fixed" {
			summary.FixedChunks = s.Chunks
		}
		if s.Strategy == "structure" {
			summary.StructureChunks = s.Chunks
		}
	}
	answer := fmt.Sprintf("Индексация завершена: %d файлов; fixed — %d фрагментов; structure — %d фрагментов. Модель эмбеддингов: %s. Время: %.1f с.\n\nИндексы и отчёт сохранены. Неизменившиеся фрагменты могут использовать ранее полученные эмбеддинги; все исходные файлы перечитаны.", summary.Files, summary.FixedChunks, summary.StructureChunks, summary.EmbeddingModel, float64(summary.DurationMS)/1000)
	response, err := a.commandReplyLocked(c, u, message, answer)
	response.Reindex = summary
	return response, err
}

func (a *Agent) projectGitContext(ctx context.Context) (projectmcp.GitContext, models.ToolExecution, error) {
	if a.tools == nil {
		return projectmcp.GitContext{}, models.ToolExecution{}, errors.New("Project MCP не подключён.")
	}
	toolCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	content, isError, err := a.tools.CallForModel(toolCtx, "project_git_branch", map[string]any{})
	if err != nil {
		return projectmcp.GitContext{}, models.ToolExecution{}, fmt.Errorf("Project MCP: %w", err)
	}
	if isError {
		return projectmcp.GitContext{}, models.ToolExecution{}, errors.New("Project MCP не смог получить текущую git-ветку.")
	}
	var git projectmcp.GitContext
	if err := json.Unmarshal([]byte(content), &git); err != nil || git.Branch == "" && (!git.Detached || git.Commit == "") {
		return projectmcp.GitContext{}, models.ToolExecution{}, errors.New("Project MCP вернул некорректный контекст git.")
	}
	execution := models.ToolExecution{Name: "project_git_branch", Arguments: map[string]any{}, Result: content}
	return git, execution, nil
}

func gitBranchLine(git projectmcp.GitContext) string {
	branch := git.Branch
	if git.Detached {
		branch = "detached HEAD (" + git.Commit + ")"
	}
	return "Текущая git-ветка: " + branch + " (MCP)."
}
