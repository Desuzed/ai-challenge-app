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
	"ai-challenge-app/internal/newsmcp"
	"ai-challenge-app/internal/projectmcp"
	"ai-challenge-app/internal/weathermcp"
)

// Catalog presents multiple independent MCP servers as one agent tool runtime.
type Catalog struct {
	yandex  *Bridge
	github  *githubmcp.Bridge
	weather *weathermcp.Bridge
	news    *newsmcp.Bridge
	project *projectmcp.Bridge
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

func NewCatalog(yandex *Bridge, github *githubmcp.Bridge, weather *weathermcp.Bridge, news *newsmcp.Bridge, project ...*projectmcp.Bridge) *Catalog {
	c := &Catalog{yandex: yandex, github: github, weather: weather, news: news}
	if len(project) > 0 {
		c.project = project[0]
	}
	return c
}
func (c *Catalog) Close() error {
	var projectErr error
	if c.project != nil {
		projectErr = c.project.Close()
	}
	return errors.Join(c.yandex.Close(), c.github.Close(), c.weather.Close(), c.news.Close(), projectErr)
}
func (c *Catalog) Status(ctx context.Context) (CatalogStatus, error) {
	ys, err := c.yandex.Status(ctx)
	if err != nil {
		return CatalogStatus{}, err
	}
	gs, err := c.github.Status(ctx)
	if err != nil {
		return CatalogStatus{}, err
	}
	ws, err := c.weather.Status(ctx)
	if err != nil {
		return CatalogStatus{}, err
	}
	ns, err := c.news.Status(ctx)
	if err != nil {
		return CatalogStatus{}, err
	}
	tools := append([]ToolView{}, ys.Tools...)
	for _, tool := range gs.Tools {
		tools = append(tools, ToolView{Name: tool.Name, Title: tool.Title, Description: tool.Description, InputSchema: tool.InputSchema, ReadOnly: tool.ReadOnly})
	}
	for _, tool := range ws.Tools {
		tools = append(tools, ToolView{Name: tool.Name, Title: tool.Title, Description: tool.Description, InputSchema: tool.InputSchema, ReadOnly: tool.ReadOnly})
	}
	for _, tool := range ns.Tools {
		tools = append(tools, ToolView{Name: tool.Name, Title: tool.Title, Description: tool.Description, InputSchema: tool.InputSchema, ReadOnly: tool.ReadOnly})
	}
	var projectServers []ServerView
	if c.project != nil {
		pt, err := c.project.Tools(ctx)
		if err != nil {
			return CatalogStatus{}, err
		}
		for _, t := range pt {
			tools = append(tools, ToolView{Name: t.Name, Title: t.Title, Description: t.Description, InputSchema: t.InputSchema, ReadOnly: t.ReadOnly})
		}
		projectServers = append(projectServers, ServerView{ID: "project", Name: "Локальный проект", Connected: true, Configured: true, Detail: "Текущая локальная git-ветка", ToolCount: len(pt)})
	}
	encoded, _ := json.Marshal(tools)
	weatherDetail := fmt.Sprintf("%d измерений · каждые %d с", ws.Measurements, ws.IntervalSeconds)
	if !ws.Scheduler.Enabled {
		weatherDetail = fmt.Sprintf("%d измерений · сбор остановлен", ws.Measurements)
	}
	return CatalogStatus{Connected: true, Servers: append(projectServers, []ServerView{{ID: "yandex-disk", Name: "Яндекс Диск", Connected: ys.Connected, Configured: ys.Configured, Detail: ys.RootFolder, ToolCount: len(ys.Tools)}, {ID: "github", Name: "GitHub", Connected: gs.Connected, Configured: gs.Configured, Detail: gs.Repository, ToolCount: len(gs.Tools)}, {ID: "weather", Name: "Погода Москвы", Connected: ws.Connected, Configured: true, Detail: weatherDetail, ToolCount: len(ws.Tools)}, {ID: "news", Name: "Новости", Connected: ns.Connected, Configured: true, Detail: "Google News RSS", ToolCount: len(ns.Tools)}}...), Tools: tools, EstimatedDefinitionTokens: estimateTextTokens(len(encoded)), TokenExplanation: "tools/list выполняется локально; в модель передаются только схемы и текстовые результаты вызовов."}, nil
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
	w, err := c.weather.ToolsForModel(ctx)
	if err != nil {
		return nil, err
	}
	n, err := c.news.ToolsForModel(ctx)
	if err != nil {
		return nil, err
	}
	tools := append(append(append(y, g...), w...), n...)
	if c.project != nil {
		p, err := c.project.ToolsForModel(ctx)
		if err != nil {
			return nil, err
		}
		tools = append(tools, p...)
	}
	return tools, nil
}
func (c *Catalog) CallForModel(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	if strings.HasPrefix(name, "project_") {
		if c.project == nil {
			return "", true, errors.New("Project MCP не подключён")
		}
		return c.project.CallForModel(ctx, name, args)
	}
	if strings.HasPrefix(name, "github_") {
		return c.github.CallForModel(ctx, name, args)
	}
	if isWeatherTool(name) {
		return c.weather.CallForModel(ctx, name, args)
	}
	if name == "search_news" {
		return c.news.CallForModel(ctx, name, args)
	}
	return c.yandex.CallForModel(ctx, name, args)
}

func isWeatherTool(name string) bool {
	return strings.HasPrefix(name, "weather_") || name == "collect_weather" || name == "get_city_weather" || name == "scheduler_status" || name == "set_weather_schedule" || name == "stop_weather_scheduler" || name == "clear_weather_history"
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
func (c *Catalog) String() string {
	return fmt.Sprintf("MCP catalog: project, yandex, github, weather and news")
}
