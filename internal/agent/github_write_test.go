package agent

import (
	"strings"
	"testing"

	"ai-challenge-app/internal/models"
)

func TestAwaitingToolConfirmationRecognizesGitHubCommit(t *testing.T) {
	messages := []models.ChatMessage{{Role: "assistant", Content: "Подтвердите параметры коммита GitHub."}}
	if !awaitingToolConfirmation(messages) {
		t.Fatal("GitHub confirmation must start an MCP tool turn")
	}
}

func TestGroundedGitHubWriteAnswerRequiresCommitSHA(t *testing.T) {
	if answer := groundedGitHubWriteAnswer("Готово", nil, true); !strings.Contains(answer, "не вызвал") {
		t.Fatalf("answer=%q", answer)
	}
	executions := []models.ToolExecution{{Name: "github_put_file", Result: `{"path":"text_challenge.txt","branch":"main","commitSha":"abc123","url":"https://github.com/example/repo/blob/main/text_challenge.txt"}`}}
	answer := groundedGitHubWriteAnswer("", executions, true)
	if !strings.Contains(answer, "abc123") || !strings.Contains(answer, "text_challenge.txt") {
		t.Fatalf("answer=%q", answer)
	}
}
