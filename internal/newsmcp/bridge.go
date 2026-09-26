// Package newsmcp exposes a small read-only MCP server for Google News RSS.
package newsmcp

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"html"
	"net/http"
	"net/url"
	"strings"
	"time"

	"ai-challenge-app/internal/models"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const defaultEndpoint = "https://news.google.com/rss/search"

type Config struct {
	Endpoint   string
	HTTPClient *http.Client
}

type Bridge struct {
	client *mcp.ClientSession
	server *mcp.ServerSession
	cfg    Config
}

type searchInput struct {
	City  string `json:"city" jsonschema:"city whose local news should be collected"`
	Query string `json:"query,omitempty" jsonschema:"optional topic; default is general city news"`
	From  string `json:"from" jsonschema:"inclusive date in YYYY-MM-DD"`
	To    string `json:"to" jsonschema:"inclusive date in YYYY-MM-DD"`
	Limit int    `json:"limit,omitempty" jsonschema:"1-10; default 8"`
}

type Article struct {
	Title       string `json:"title"`
	Source      string `json:"source"`
	PublishedAt string `json:"publishedAt"`
	URL         string `json:"url"`
	Snippet     string `json:"snippet,omitempty"`
}

type searchOutput struct {
	Query    string    `json:"query"`
	From     string    `json:"from"`
	To       string    `json:"to"`
	Articles []Article `json:"articles"`
}

type ToolView struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
	ReadOnly    bool   `json:"readOnly"`
}

type Status struct {
	Connected bool       `json:"connected"`
	Server    string     `json:"server"`
	Tools     []ToolView `json:"tools"`
}

func New(ctx context.Context, cfg Config) (*Bridge, error) {
	if cfg.Endpoint == "" {
		cfg.Endpoint = defaultEndpoint
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 75 * time.Second}
	}
	b := &Bridge{cfg: cfg}
	server := mcp.NewServer(&mcp.Implementation{Name: "news-search-mcp", Version: "1.0.0"}, &mcp.ServerOptions{Instructions: "Search raw local news through Google News RSS. Do not summarize or invent facts."})
	readOnly, notDestructive, openWorld := true, false, true
	mcp.AddTool(server, &mcp.Tool{Name: "search_news", Title: "Найти городские новости за период", Description: "Возвращает сырые новости из бесплатного Google News RSS по городу, теме и датам. Используй результат как единственный источник фактов для сводки.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &notDestructive, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, in searchInput) (*mcp.CallToolResult, searchOutput, error) {
		out, err := b.search(ctx, in)
		return nil, out, err
	})
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		return nil, err
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "ai-challenge-agent", Version: "1.0.0"}, nil)
	clientSession, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		_ = serverSession.Close()
		return nil, err
	}
	b.server, b.client = serverSession, clientSession
	return b, nil
}

func (b *Bridge) Close() error {
	var errs []error
	if b.client != nil {
		errs = append(errs, b.client.Close())
	}
	if b.server != nil {
		errs = append(errs, b.server.Close())
	}
	return errors.Join(errs...)
}

func (b *Bridge) search(ctx context.Context, in searchInput) (searchOutput, error) {
	in.City = strings.TrimSpace(in.City)
	in.Query = strings.TrimSpace(in.Query)
	if in.City == "" {
		return searchOutput{}, errors.New("city is required")
	}
	from, err := time.Parse("2006-01-02", in.From)
	if err != nil {
		return searchOutput{}, errors.New("from must be YYYY-MM-DD")
	}
	to, err := time.Parse("2006-01-02", in.To)
	if err != nil {
		return searchOutput{}, errors.New("to must be YYYY-MM-DD")
	}
	if to.Before(from) {
		return searchOutput{}, errors.New("to must not be earlier than from")
	}
	if to.Sub(from) > 31*24*time.Hour {
		return searchOutput{}, errors.New("news period must not exceed 31 days")
	}
	if in.Limit == 0 {
		in.Limit = 8
	}
	if in.Limit < 1 || in.Limit > 10 {
		return searchOutput{}, errors.New("limit must be from 1 to 10")
	}
	query := `"` + in.City + `"`
	if in.Query != "" {
		query += " " + in.Query
	}
	u, err := url.Parse(b.cfg.Endpoint)
	if err != nil {
		return searchOutput{}, err
	}
	values := u.Query()
	values.Set("q", query+" after:"+from.Format("2006-01-02")+" before:"+to.AddDate(0, 0, 1).Format("2006-01-02"))
	values.Set("hl", "ru")
	values.Set("gl", "RU")
	values.Set("ceid", "RU:ru")
	u.RawQuery = values.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return searchOutput{}, err
	}
	req.Header.Set("User-Agent", "ai-challenge-news-mcp/1.0")
	var resp *http.Response
	for attempt := 0; attempt < 3; attempt++ {
		resp, err = b.cfg.HTTPClient.Do(req)
		if err == nil {
			break
		}
		if ctx.Err() != nil {
			return searchOutput{}, ctx.Err()
		}
		if attempt < 2 {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}
	if err != nil {
		return searchOutput{}, fmt.Errorf("GDELT unavailable after 3 attempts: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return searchOutput{}, fmt.Errorf("news source returned HTTP %d", resp.StatusCode)
	}
	var feed struct {
		Channel struct {
			Items []struct {
				Title       string `xml:"title"`
				Link        string `xml:"link"`
				PubDate     string `xml:"pubDate"`
				Source      string `xml:"source"`
				Description string `xml:"description"`
			} `xml:"item"`
		} `xml:"channel"`
	}
	if err := xml.NewDecoder(resp.Body).Decode(&feed); err != nil {
		return searchOutput{}, err
	}
	out := searchOutput{Query: strings.TrimSpace(in.City + " " + in.Query), From: in.From, To: in.To, Articles: make([]Article, 0, min(in.Limit, len(feed.Channel.Items)))}
	for _, item := range feed.Channel.Items {
		if len(out.Articles) == in.Limit {
			break
		}
		published := item.PubDate
		if date, parseErr := time.Parse(time.RFC1123Z, item.PubDate); parseErr == nil {
			published = date.UTC().Format(time.RFC3339)
		}
		out.Articles = append(out.Articles, Article{Title: strings.TrimSpace(item.Title), Source: strings.TrimSpace(item.Source), PublishedAt: published, URL: strings.TrimSpace(item.Link), Snippet: stripHTML(item.Description)})
	}
	return out, nil
}

func stripHTML(value string) string {
	value = html.UnescapeString(value)
	for {
		start := strings.Index(value, "<")
		if start < 0 {
			break
		}
		end := strings.Index(value[start:], ">")
		if end < 0 {
			break
		}
		value = value[:start] + " " + value[start+end+1:]
	}
	return strings.Join(strings.Fields(value), " ")
}

func (b *Bridge) Status(ctx context.Context) (Status, error) {
	r, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return Status{}, err
	}
	tools := make([]ToolView, 0, len(r.Tools))
	for _, tool := range r.Tools {
		view := ToolView{Name: tool.Name, Title: tool.Title, Description: tool.Description, InputSchema: tool.InputSchema}
		if tool.Annotations != nil {
			view.ReadOnly = tool.Annotations.ReadOnlyHint
		}
		tools = append(tools, view)
	}
	return Status{Connected: true, Server: "news-search-mcp 1.0.0", Tools: tools}, nil
}

func (b *Bridge) ToolsForModel(ctx context.Context) ([]models.ToolDefinition, error) {
	r, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}
	tools := make([]models.ToolDefinition, 0, len(r.Tools))
	for _, tool := range r.Tools {
		tools = append(tools, models.ToolDefinition{Type: "function", Function: models.ToolFunction{Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema}})
	}
	return tools, nil
}

func (b *Bridge) CallForModel(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	r, err := b.client.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		return "", true, err
	}
	if r.StructuredContent != nil {
		encoded, marshalErr := json.Marshal(r.StructuredContent)
		return string(encoded), r.IsError, marshalErr
	}
	texts := make([]string, 0, len(r.Content))
	for _, content := range r.Content {
		if text, ok := content.(*mcp.TextContent); ok {
			texts = append(texts, text.Text)
		}
	}
	return strings.Join(texts, "\n"), r.IsError, nil
}
