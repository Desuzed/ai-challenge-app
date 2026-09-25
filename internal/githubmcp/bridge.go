// Package githubmcp exposes a small, auditable GitHub API surface as MCP tools.
package githubmcp

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"ai-challenge-app/internal/models"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Config struct {
	Token, Repository, Endpoint string
	HTTPClient                  *http.Client
}

type Bridge struct {
	client      *mcp.ClientSession
	server      *mcp.ServerSession
	cfg         Config
	owner, repo string
}
type ToolView struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
	ReadOnly    bool   `json:"readOnly"`
}
type Status struct {
	Connected  bool       `json:"connected"`
	Server     string     `json:"server"`
	Configured bool       `json:"configured"`
	Repository string     `json:"repository"`
	Tools      []ToolView `json:"tools"`
}
type CallResponse struct {
	Name              string `json:"name"`
	IsError           bool   `json:"isError"`
	Content           string `json:"content,omitempty"`
	StructuredContent any    `json:"structuredContent,omitempty"`
}

type repositoryInput struct{}
type repositoryOutput struct {
	FullName      string `json:"fullName"`
	Description   string `json:"description"`
	DefaultBranch string `json:"defaultBranch"`
	Private       bool   `json:"private"`
	URL           string `json:"url"`
}
type getFileInput struct {
	Path string `json:"path" jsonschema:"repository-relative text-file path"`
	Ref  string `json:"ref,omitempty" jsonschema:"branch, tag, or commit; default is repository default branch"`
}
type getFileOutput struct {
	Path    string `json:"path"`
	Ref     string `json:"ref"`
	SHA     string `json:"sha"`
	Content string `json:"content"`
}
type listFilesInput struct {
	Ref   string `json:"ref,omitempty" jsonschema:"branch, tag, or commit; default is repository default branch"`
	Limit int    `json:"limit,omitempty" jsonschema:"1-500; default 100"`
}
type repositoryFile struct {
	Path string `json:"path"`
	Type string `json:"type"`
	Size int64  `json:"size,omitempty"`
}
type listFilesOutput struct {
	Ref   string           `json:"ref"`
	Files []repositoryFile `json:"files"`
}
type listIssuesInput struct {
	State string `json:"state,omitempty" jsonschema:"open, closed, or all; default open"`
	Limit int    `json:"limit,omitempty" jsonschema:"1-100; default 20"`
}
type issue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	State  string `json:"state"`
	URL    string `json:"url"`
}
type listIssuesOutput struct {
	Issues []issue `json:"issues"`
}
type putFileInput struct {
	Path    string `json:"path" jsonschema:"repository-relative text-file path"`
	Content string `json:"content" jsonschema:"complete UTF-8 text to commit"`
	Message string `json:"message" jsonschema:"commit message"`
	Branch  string `json:"branch,omitempty" jsonschema:"target branch; default is repository default branch"`
	Confirm bool   `json:"confirm" jsonschema:"must be true after explicit user confirmation"`
}
type putFileOutput struct {
	Path      string `json:"path"`
	Branch    string `json:"branch"`
	CommitSHA string `json:"commitSha"`
	URL       string `json:"url"`
	Created   bool   `json:"created"`
}
type deleteFileInput struct {
	Path    string `json:"path" jsonschema:"repository-relative text-file path"`
	Message string `json:"message" jsonschema:"commit message"`
	Branch  string `json:"branch,omitempty" jsonschema:"target branch; default is repository default branch"`
	Confirm bool   `json:"confirm" jsonschema:"must be true after explicit user confirmation"`
}
type deleteFileOutput struct {
	Path      string `json:"path"`
	Branch    string `json:"branch"`
	CommitSHA string `json:"commitSha"`
}

func New(ctx context.Context, cfg Config) (*Bridge, error) {
	parts := strings.Split(strings.TrimSpace(cfg.Repository), "/")
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return nil, errors.New("GITHUB_REPOSITORY must be owner/repository")
	}
	if cfg.Endpoint == "" {
		cfg.Endpoint = "https://api.github.com"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	}
	b := &Bridge{cfg: cfg, owner: parts[0], repo: parts[1]}
	server := mcp.NewServer(&mcp.Implementation{Name: "github-repository-mcp", Version: "1.0.0"}, &mcp.ServerOptions{Instructions: "Read GitHub repository data and edit text files only after explicit confirmation."})
	readOnly, falseValue, openWorld := true, false, true
	destructive := true
	mcp.AddTool(server, &mcp.Tool{Name: "github_get_repository", Title: "Сведения о репозитории GitHub", Description: "Возвращает название, описание, ветку по умолчанию и ссылку подключённого репозитория.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &falseValue, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, _ repositoryInput) (*mcp.CallToolResult, repositoryOutput, error) {
		output, err := b.getRepository(ctx)
		return nil, output, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "github_get_file", Title: "Прочитать текстовый файл GitHub", Description: "Читает UTF-8 текстовый файл из подключённого GitHub-репозитория.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true, DestructiveHint: &falseValue, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, in getFileInput) (*mcp.CallToolResult, getFileOutput, error) {
		output, err := b.getFile(ctx, in.Path, in.Ref)
		return nil, output, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "github_list_files", Title: "Просмотреть структуру GitHub-репозитория", Description: "Возвращает пути файлов и папок в подключённом GitHub-репозитории.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true, DestructiveHint: &falseValue, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, in listFilesInput) (*mcp.CallToolResult, listFilesOutput, error) {
		output, err := b.listFiles(ctx, in)
		return nil, output, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "github_list_issues", Title: "Список GitHub Issues", Description: "Возвращает задачи (issues) подключённого GitHub-репозитория.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true, DestructiveHint: &falseValue, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, in listIssuesInput) (*mcp.CallToolResult, listIssuesOutput, error) {
		output, err := b.listIssues(ctx, in)
		return nil, output, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "github_put_file", Title: "Создать или обновить текстовый файл", Description: "Создаёт или обновляет UTF-8 текстовый файл коммитом в GitHub после confirm=true.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &falseValue, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, in putFileInput) (*mcp.CallToolResult, putFileOutput, error) {
		if !in.Confirm {
			return nil, putFileOutput{}, errors.New("запись в GitHub требует явного confirm=true")
		}
		output, err := b.putFile(ctx, in)
		return nil, output, err
	})
	mcp.AddTool(server, &mcp.Tool{Name: "github_delete_file", Title: "Удалить текстовый файл", Description: "Удаляет текстовый файл отдельным коммитом GitHub после confirm=true.", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &destructive, OpenWorldHint: &openWorld}}, func(ctx context.Context, _ *mcp.CallToolRequest, in deleteFileInput) (*mcp.CallToolResult, deleteFileOutput, error) {
		if !in.Confirm {
			return nil, deleteFileOutput{}, errors.New("удаление из GitHub требует явного confirm=true")
		}
		output, err := b.deleteFile(ctx, in)
		return nil, output, err
	})
	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	session, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		return nil, err
	}
	b.server = session
	client := mcp.NewClient(&mcp.Implementation{Name: "ai-challenge-agent", Version: "1.0.0"}, nil)
	clientSession, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		_ = session.Close()
		return nil, err
	}
	b.client = clientSession
	return b, nil
}
func (b *Bridge) Close() error { return errors.Join(b.client.Close(), b.server.Close()) }
func (b *Bridge) Status(ctx context.Context) (Status, error) {
	result, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return Status{}, err
	}
	tools := make([]ToolView, 0, len(result.Tools))
	for _, t := range result.Tools {
		v := ToolView{Name: t.Name, Title: t.Title, Description: t.Description, InputSchema: t.InputSchema}
		if t.Annotations != nil {
			v.ReadOnly = t.Annotations.ReadOnlyHint
		}
		tools = append(tools, v)
	}
	return Status{Connected: true, Server: "github-repository-mcp 1.0.0", Configured: strings.TrimSpace(b.cfg.Token) != "", Repository: b.cfg.Repository, Tools: tools}, nil
}
func (b *Bridge) ToolsForModel(ctx context.Context) ([]models.ToolDefinition, error) {
	result, err := b.client.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}
	tools := make([]models.ToolDefinition, 0, len(result.Tools))
	for _, t := range result.Tools {
		tools = append(tools, models.ToolDefinition{Type: "function", Function: models.ToolFunction{Name: t.Name, Description: t.Description, Parameters: t.InputSchema}})
	}
	return tools, nil
}
func (b *Bridge) CallForModel(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	result, err := b.Call(ctx, name, args)
	if err != nil {
		return "", true, err
	}
	if result.StructuredContent != nil {
		encoded, err := json.Marshal(result.StructuredContent)
		return string(encoded), result.IsError, err
	}
	return result.Content, result.IsError, nil
}
func (b *Bridge) Call(ctx context.Context, name string, args map[string]any) (CallResponse, error) {
	result, err := b.client.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		return CallResponse{}, err
	}
	texts := []string{}
	for _, c := range result.Content {
		if text, ok := c.(*mcp.TextContent); ok {
			texts = append(texts, text.Text)
		}
	}
	return CallResponse{Name: name, IsError: result.IsError, Content: strings.Join(texts, "\n"), StructuredContent: result.StructuredContent}, nil
}

func (b *Bridge) api(ctx context.Context, method, suffix string, body any, out any) (int, error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return 0, err
		}
		reader = bytes.NewReader(data)
	}
	request, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(b.cfg.Endpoint, "/")+suffix, reader)
	if err != nil {
		return 0, err
	}
	request.Header.Set("Accept", "application/vnd.github+json")
	request.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	request.Header.Set("User-Agent", "ai-challenge-mcp")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if token := strings.TrimSpace(b.cfg.Token); token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := b.cfg.HTTPClient.Do(request)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(response.Body, 8<<10))
		return response.StatusCode, fmt.Errorf("GitHub API %s %s: HTTP %d: %s", method, suffix, response.StatusCode, strings.TrimSpace(string(raw)))
	}
	if out != nil {
		if err := json.NewDecoder(response.Body).Decode(out); err != nil {
			return response.StatusCode, err
		}
	}
	return response.StatusCode, nil
}
func (b *Bridge) base() string {
	return "/repos/" + url.PathEscape(b.owner) + "/" + url.PathEscape(b.repo)
}
func safePath(value string) (string, error) {
	clean := path.Clean(strings.TrimSpace(value))
	if clean == "." || strings.HasPrefix(clean, "../") || strings.HasPrefix(clean, "/") {
		return "", errors.New("path must be a repository-relative file path")
	}
	return clean, nil
}
func (b *Bridge) contentPath(file string) (string, error) {
	clean, err := safePath(file)
	if err != nil {
		return "", err
	}
	return b.base() + "/contents/" + strings.ReplaceAll(url.PathEscape(clean), "%2F", "/"), nil
}
func (b *Bridge) getRepository(ctx context.Context) (repositoryOutput, error) {
	var data struct {
		FullName      string `json:"full_name"`
		Description   string `json:"description"`
		DefaultBranch string `json:"default_branch"`
		Private       bool   `json:"private"`
		HTMLURL       string `json:"html_url"`
	}
	_, err := b.api(ctx, http.MethodGet, b.base(), nil, &data)
	return repositoryOutput{FullName: data.FullName, Description: data.Description, DefaultBranch: data.DefaultBranch, Private: data.Private, URL: data.HTMLURL}, err
}
func (b *Bridge) getFile(ctx context.Context, file, ref string) (getFileOutput, error) {
	suffix, err := b.contentPath(file)
	if err != nil {
		return getFileOutput{}, err
	}
	if ref != "" {
		suffix += "?ref=" + url.QueryEscape(ref)
	}
	var data struct {
		Path     string `json:"path"`
		SHA      string `json:"sha"`
		Content  string `json:"content"`
		Encoding string `json:"encoding"`
	}
	_, err = b.api(ctx, http.MethodGet, suffix, nil, &data)
	if err != nil {
		return getFileOutput{}, err
	}
	if data.Encoding != "base64" {
		return getFileOutput{}, errors.New("GitHub returned a non-text file")
	}
	decoded, err := base64.StdEncoding.DecodeString(strings.ReplaceAll(data.Content, "\n", ""))
	if err != nil {
		return getFileOutput{}, err
	}
	return getFileOutput{Path: data.Path, Ref: ref, SHA: data.SHA, Content: string(decoded)}, nil
}

func (b *Bridge) listFiles(ctx context.Context, in listFilesInput) (listFilesOutput, error) {
	ref := strings.TrimSpace(in.Ref)
	if ref == "" {
		repository, err := b.getRepository(ctx)
		if err != nil {
			return listFilesOutput{}, err
		}
		ref = repository.DefaultBranch
	}
	limit := in.Limit
	if limit == 0 {
		limit = 100
	}
	if limit < 1 || limit > 500 {
		return listFilesOutput{}, errors.New("limit must be from 1 to 500")
	}
	var data struct {
		Tree []struct {
			Path string `json:"path"`
			Type string `json:"type"`
			Size int64  `json:"size"`
		} `json:"tree"`
	}
	_, err := b.api(ctx, http.MethodGet, b.base()+"/git/trees/"+url.PathEscape(ref)+"?recursive=1", nil, &data)
	if err != nil {
		return listFilesOutput{}, err
	}
	output := listFilesOutput{Ref: ref, Files: make([]repositoryFile, 0, min(limit, len(data.Tree)))}
	for _, entry := range data.Tree {
		output.Files = append(output.Files, repositoryFile{Path: entry.Path, Type: entry.Type, Size: entry.Size})
		if len(output.Files) == limit {
			break
		}
	}
	return output, nil
}
func (b *Bridge) listIssues(ctx context.Context, in listIssuesInput) (listIssuesOutput, error) {
	state := in.State
	if state == "" {
		state = "open"
	}
	if state != "open" && state != "closed" && state != "all" {
		return listIssuesOutput{}, errors.New("state must be open, closed, or all")
	}
	limit := in.Limit
	if limit == 0 {
		limit = 20
	}
	if limit < 1 || limit > 100 {
		return listIssuesOutput{}, errors.New("limit must be from 1 to 100")
	}
	var data []struct {
		Number      int             `json:"number"`
		Title       string          `json:"title"`
		State       string          `json:"state"`
		HTMLURL     string          `json:"html_url"`
		PullRequest json.RawMessage `json:"pull_request"`
	}
	_, err := b.api(ctx, http.MethodGet, b.base()+"/issues?state="+state+"&per_page="+fmt.Sprint(limit), nil, &data)
	if err != nil {
		return listIssuesOutput{}, err
	}
	result := listIssuesOutput{}
	for _, item := range data {
		if len(item.PullRequest) > 0 {
			continue
		}
		result.Issues = append(result.Issues, issue{Number: item.Number, Title: item.Title, State: item.State, URL: item.HTMLURL})
	}
	return result, nil
}
func (b *Bridge) putFile(ctx context.Context, in putFileInput) (putFileOutput, error) {
	if strings.TrimSpace(b.cfg.Token) == "" {
		return putFileOutput{}, errors.New("set GITHUB_TOKEN in .env to write to GitHub")
	}
	suffix, err := b.contentPath(in.Path)
	if err != nil {
		return putFileOutput{}, err
	}
	if strings.TrimSpace(in.Content) == "" {
		return putFileOutput{}, errors.New("content must not be empty")
	}
	if strings.TrimSpace(in.Message) == "" {
		return putFileOutput{}, errors.New("commit message is required")
	}
	branch := strings.TrimSpace(in.Branch)
	if branch == "" {
		repo, err := b.getRepository(ctx)
		if err != nil {
			return putFileOutput{}, err
		}
		branch = repo.DefaultBranch
	}
	var existing struct {
		SHA string `json:"sha"`
	}
	code, readErr := b.api(ctx, http.MethodGet, suffix+"?ref="+url.QueryEscape(branch), nil, &existing)
	if readErr != nil && code != http.StatusNotFound {
		return putFileOutput{}, readErr
	}
	payload := map[string]string{"message": in.Message, "content": base64.StdEncoding.EncodeToString([]byte(in.Content)), "branch": branch}
	created := code == http.StatusNotFound
	if !created {
		payload["sha"] = existing.SHA
	}
	var response struct {
		Content struct {
			HTMLURL string `json:"html_url"`
		}
		Commit struct {
			SHA string `json:"sha"`
		}
	}
	_, err = b.api(ctx, http.MethodPut, suffix, payload, &response)
	if err != nil {
		return putFileOutput{}, err
	}
	return putFileOutput{Path: in.Path, Branch: branch, CommitSHA: response.Commit.SHA, URL: response.Content.HTMLURL, Created: created}, nil
}

func (b *Bridge) deleteFile(ctx context.Context, in deleteFileInput) (deleteFileOutput, error) {
	if strings.TrimSpace(b.cfg.Token) == "" {
		return deleteFileOutput{}, errors.New("set GITHUB_TOKEN in .env to write to GitHub")
	}
	suffix, err := b.contentPath(in.Path)
	if err != nil {
		return deleteFileOutput{}, err
	}
	if strings.TrimSpace(in.Message) == "" {
		return deleteFileOutput{}, errors.New("commit message is required")
	}
	branch := strings.TrimSpace(in.Branch)
	if branch == "" {
		repository, err := b.getRepository(ctx)
		if err != nil {
			return deleteFileOutput{}, err
		}
		branch = repository.DefaultBranch
	}
	var existing struct {
		SHA string `json:"sha"`
	}
	_, err = b.api(ctx, http.MethodGet, suffix+"?ref="+url.QueryEscape(branch), nil, &existing)
	if err != nil {
		return deleteFileOutput{}, err
	}
	var response struct {
		Commit struct {
			SHA string `json:"sha"`
		} `json:"commit"`
	}
	_, err = b.api(ctx, http.MethodDelete, suffix, map[string]string{"message": in.Message, "sha": existing.SHA, "branch": branch}, &response)
	if err != nil {
		return deleteFileOutput{}, err
	}
	return deleteFileOutput{Path: in.Path, Branch: branch, CommitSHA: response.Commit.SHA}, nil
}
