package mcpbridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"ai-challenge-app/internal/githubmcp"
	"ai-challenge-app/internal/models"
)

// Catalog presents multiple independent MCP servers as one agent tool runtime.
type Catalog struct {
	yandex *Bridge
	github *githubmcp.Bridge
}
type ServerView struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Connected  bool   `json:"connected"`
	Configured bool   `json:"configured"`
	Detail     string `json:"detail"`
	ToolCount  int    `json:"toolCount"`
}
type CatalogStatus struct {
	Connected                 bool         `json:"connected"`
	Servers                   []ServerView `json:"servers"`
	Tools                     []ToolView   `json:"tools"`
	EstimatedDefinitionTokens int          `json:"estimatedDefinitionTokens"`
	TokenExplanation          string       `json:"tokenExplanation"`
}

func NewCatalog(yandex *Bridge, github *githubmcp.Bridge) *Catalog {
	return &Catalog{yandex: yandex, github: github}
}
func (c *Catalog) Close() error { return errors.Join(c.yandex.Close(), c.github.Close()) }
func (c *Catalog) Status(ctx context.Context) (CatalogStatus, error) {
	ys, err := c.yandex.Status(ctx)
	if err != nil {
		return CatalogStatus{}, err
	}
	gs, err := c.github.Status(ctx)
	if err != nil {
		return CatalogStatus{}, err
	}
	tools := append([]ToolView{}, ys.Tools...)
	for _, tool := range gs.Tools {
		tools = append(tools, ToolView{Name: tool.Name, Title: tool.Title, Description: tool.Description, InputSchema: tool.InputSchema, ReadOnly: tool.ReadOnly})
	}
	encoded, _ := json.Marshal(tools)
	return CatalogStatus{Connected: true, Servers: []ServerView{{ID: "yandex-disk", Name: "Яндекс Диск", Connected: ys.Connected, Configured: ys.Configured, Detail: ys.RootFolder, ToolCount: len(ys.Tools)}, {ID: "github", Name: "GitHub", Connected: gs.Connected, Configured: gs.Configured, Detail: gs.Repository, ToolCount: len(gs.Tools)}}, Tools: tools, EstimatedDefinitionTokens: estimateTextTokens(len(encoded)), TokenExplanation: "tools/list выполняется локально; в модель передаются только схемы и текстовые результаты вызовов."}, nil
}
func (c *Catalog) ToolsForModel(ctx context.Context) ([]models.ToolDefinition, error) {
	y, err := c.yandex.ToolsForModel(ctx)
	if err != nil {
		return nil, err
	}
	g, err := c.github.ToolsForModel(ctx)
	if err != nil {
		return nil, err
	}
	return append(y, g...), nil
}
func (c *Catalog) CallForModel(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	if strings.HasPrefix(name, "github_") {
		return c.github.CallForModel(ctx, name, args)
	}
	return c.yandex.CallForModel(ctx, name, args)
}
func (c *Catalog) ToolsHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "Используйте GET-запрос."})
		return
	}
	status, err := c.Status(r.Context())
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, status)
}
func (c *Catalog) CallHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "Используйте POST-запрос."})
		return
	}
	var input CallRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32<<10)).Decode(&input); err != nil || strings.TrimSpace(input.Name) == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Некорректный JSON вызова MCP."})
		return
	}
	content, isError, err := c.CallForModel(r.Context(), input.Name, input.Arguments)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	status := http.StatusOK
	if isError {
		status = http.StatusBadRequest
	}
	writeJSON(w, status, map[string]any{"name": input.Name, "isError": isError, "content": content})
}
func (c *Catalog) String() string { return fmt.Sprintf("MCP catalog: yandex and github") }
