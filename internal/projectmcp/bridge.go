// Package projectmcp exposes the current local checkout through read-only MCP.
package projectmcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"ai-challenge-app/internal/models"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Bridge struct {
	root   string
	client *mcp.ClientSession
	server *mcp.ServerSession
}

type GitContext struct {
	Branch   string `json:"branch"`
	Detached bool   `json:"detached"`
	Commit   string `json:"commit,omitempty"`
}

type ToolView struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
	ReadOnly    bool   `json:"readOnly"`
}

func New(ctx context.Context, root string) (*Bridge, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	b := &Bridge{root: abs}
	server := mcp.NewServer(&mcp.Implementation{Name: "local-project-mcp", Version: "1.0.0"}, nil)
	no := false
	mcp.AddTool(server, &mcp.Tool{Name: "project_git_branch", Title: "Текущая локальная git-ветка", Description: "Читает текущую ветку локального проекта. При detached HEAD возвращает detached=true и commit. Ветка диалога и ветка по умолчанию GitHub к этому результату не относятся.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true, DestructiveHint: &no, OpenWorldHint: &no}}, func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, GitContext, error) {
		out, err := b.gitContext(ctx)
		return nil, out, err
	})
	ct, st := mcp.NewInMemoryTransports()
	b.server, err = server.Connect(ctx, st, nil)
	if err != nil {
		return nil, err
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "project-context-client", Version: "1.0.0"}, nil)
	b.client, err = client.Connect(ctx, ct, nil)
	if err != nil {
		_ = b.server.Close()
		return nil, err
	}
	return b, nil
}

func (b *Bridge) gitContext(ctx context.Context) (GitContext, error) {
	// Arguments and checkout are fixed by the server, never supplied by the model.
	if _, err := b.git(ctx, "rev-parse", "--show-toplevel"); err != nil {
		return GitContext{}, errors.New("локальный проект не является доступным git-репозиторием")
	}
	branch, err := b.git(ctx, "branch", "--show-current")
	if err != nil {
		return GitContext{}, fmt.Errorf("получить git-ветку: %w", err)
	}
	commit, err := b.git(ctx, "rev-parse", "--verify", "HEAD")
	if err != nil && branch == "" {
		return GitContext{}, errors.New("git HEAD недоступен")
	}
	return GitContext{Branch: branch, Detached: branch == "", Commit: commit}, nil
}

func (b *Bridge) git(ctx context.Context, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", b.root}, args...)...)
	data, err := cmd.Output()
	return strings.TrimSpace(string(data)), err
}

func (b *Bridge) Tools(ctx context.Context) ([]ToolView, error) {
	result, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}
	views := make([]ToolView, 0, len(result.Tools))
	for _, t := range result.Tools {
		views = append(views, ToolView{Name: t.Name, Title: t.Title, Description: t.Description, InputSchema: t.InputSchema, ReadOnly: true})
	}
	return views, nil
}

func (b *Bridge) ToolsForModel(ctx context.Context) ([]models.ToolDefinition, error) {
	views, err := b.Tools(ctx)
	if err != nil {
		return nil, err
	}
	result := make([]models.ToolDefinition, 0, len(views))
	for _, t := range views {
		result = append(result, models.ToolDefinition{Type: "function", Function: models.ToolFunction{Name: t.Name, Description: t.Description, Parameters: t.InputSchema}})
	}
	return result, nil
}

func (b *Bridge) CallForModel(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	result, err := b.client.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		return "", false, err
	}
	if result.StructuredContent != nil {
		data, err := json.Marshal(result.StructuredContent)
		return string(data), result.IsError, err
	}
	var text []string
	for _, block := range result.Content {
		if item, ok := block.(*mcp.TextContent); ok {
			text = append(text, item.Text)
		}
	}
	return strings.Join(text, "\n"), result.IsError, nil
}

func (b *Bridge) Close() error {
	return errors.Join(b.client.Close(), b.server.Close())
}
