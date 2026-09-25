package mcpbridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"ai-challenge-app/internal/models"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const gib = int64(1024 * 1024 * 1024)

var videoExtensions = map[string]bool{
	".avi": true, ".m4v": true, ".mkv": true, ".mov": true, ".mp4": true, ".webm": true,
}

type Config struct {
	OAuthToken  string
	RootFolder  string
	VideoDir    string
	Endpoint    string
	HTTPClient  *http.Client
	FFprobePath string
}

type Bridge struct {
	clientSession *mcp.ClientSession
	serverSession *mcp.ServerSession
	config        Config
}

type Video struct {
	Name       string    `json:"name"`
	SizeBytes  int64     `json:"sizeBytes"`
	ModifiedAt time.Time `json:"modifiedAt"`
}

type ListVideosInput struct{}
type ListVideosOutput struct {
	Videos []Video `json:"videos"`
}

type EstimateInput struct {
	VideoName string `json:"videoName" jsonschema:"filename returned by list_desktop_videos"`
}

type AnalyzeInput struct {
	VideoName string `json:"videoName" jsonschema:"filename returned by list_desktop_videos"`
}

type AnalyzeOutput struct {
	VideoName             string   `json:"videoName"`
	SizeBytes             int64    `json:"sizeBytes"`
	DurationSeconds       float64  `json:"durationSeconds,omitempty"`
	Container             string   `json:"container,omitempty"`
	VideoCodec            string   `json:"videoCodec,omitempty"`
	AudioCodec            string   `json:"audioCodec,omitempty"`
	Width                 int      `json:"width,omitempty"`
	Height                int      `json:"height,omitempty"`
	FramesPerSecond       float64  `json:"framesPerSecond,omitempty"`
	ContentAnalyzed       bool     `json:"contentAnalyzed"`
	VideoBytesSentToModel int64    `json:"videoBytesSentToModel"`
	Notes                 []string `json:"notes"`
}

type EstimateOutput struct {
	VideoName             string   `json:"videoName"`
	SizeBytes             int64    `json:"sizeBytes"`
	SizeGiB               float64  `json:"sizeGiB"`
	DiskTotalBytes        int64    `json:"diskTotalBytes,omitempty"`
	DiskUsedBytes         int64    `json:"diskUsedBytes,omitempty"`
	DiskFreeBytes         int64    `json:"diskFreeBytes,omitempty"`
	FitsAvailableSpace    *bool    `json:"fitsAvailableSpace,omitempty"`
	APIRequestsForUpload  int      `json:"apiRequestsForUpload"`
	VideoBytesSentToModel int64    `json:"videoBytesSentToModel"`
	Notes                 []string `json:"notes"`
}

type UploadInput struct {
	VideoName    string `json:"videoName" jsonschema:"filename returned by list_desktop_videos"`
	LessonFolder string `json:"lessonFolder" jsonschema:"existing subfolder inside AI Challenge, for example lession 16"`
	Confirm      bool   `json:"confirm" jsonschema:"must be true after explicit user confirmation"`
}

type UploadOutput struct {
	DiskPath       string `json:"diskPath"`
	SizeBytes      int64  `json:"sizeBytes"`
	AlreadyExisted bool   `json:"alreadyExisted,omitempty"`
}

type ToolView struct {
	Name        string `json:"name"`
	Title       string `json:"title,omitempty"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
	ReadOnly    bool   `json:"readOnly"`
}

type Status struct {
	Connected                    bool       `json:"connected"`
	Server                       string     `json:"server"`
	Configured                   bool       `json:"configured"`
	RootFolder                   string     `json:"rootFolder"`
	VideoDirectory               string     `json:"videoDirectory"`
	Tools                        []ToolView `json:"tools"`
	EstimatedDefinitionTokens    int        `json:"estimatedDefinitionTokens"`
	ModelTokensUsedByThisRequest int        `json:"modelTokensUsedByThisRequest"`
	TokenExplanation             string     `json:"tokenExplanation"`
}

type CallRequest struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type CallResponse struct {
	Name                         string `json:"name"`
	IsError                      bool   `json:"isError"`
	Content                      string `json:"content,omitempty"`
	StructuredContent            any    `json:"structuredContent,omitempty"`
	EstimatedPayloadTokens       int    `json:"estimatedPayloadTokens"`
	ModelTokensUsedByThisRequest int    `json:"modelTokensUsedByThisRequest"`
}

func New(ctx context.Context, cfg Config) (*Bridge, error) {
	if strings.TrimSpace(cfg.VideoDir) == "" {
		return nil, errors.New("video directory is required")
	}
	if strings.TrimSpace(cfg.RootFolder) == "" {
		cfg.RootFolder = "AI Challenge"
	}
	if strings.TrimSpace(cfg.Endpoint) == "" {
		cfg.Endpoint = "https://cloud-api.yandex.net/v1/disk"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 30 * time.Minute}
	}
	if strings.TrimSpace(cfg.FFprobePath) == "" {
		cfg.FFprobePath, _ = exec.LookPath("ffprobe")
	}

	server := mcp.NewServer(&mcp.Implementation{Name: "yandex-disk-video-mcp", Version: "1.0.0"}, &mcp.ServerOptions{
		Instructions: "List a Desktop recording first, inspect its size, and upload to Yandex Disk only after explicit confirmation.",
	})
	readOnly, notDestructive, closedWorld, openWorld := true, false, false, true

	mcp.AddTool(server, &mcp.Tool{
		Name: "analyze_desktop_video", Title: "Технически проанализировать видео",
		Description: "Возвращает размер, длительность, контейнер, кодеки, разрешение и FPS локального видео. Не анализирует содержание кадров или речи.",
		Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &notDestructive, OpenWorldHint: &closedWorld},
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in AnalyzeInput) (*mcp.CallToolResult, AnalyzeOutput, error) {
		video, path, err := resolveVideo(cfg.VideoDir, in.VideoName)
		if err != nil {
			return nil, AnalyzeOutput{}, err
		}
		return nil, analyzeVideo(ctx, cfg, video, path), nil
	})

	mcp.AddTool(server, &mcp.Tool{
		Name: "list_desktop_videos", Title: "Видео на Рабочем столе",
		Description: "Возвращает записи видео из разрешённой локальной папки.",
		Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true, DestructiveHint: &notDestructive, OpenWorldHint: &closedWorld},
	}, func(_ context.Context, _ *mcp.CallToolRequest, _ ListVideosInput) (*mcp.CallToolResult, ListVideosOutput, error) {
		videos, err := listVideos(cfg.VideoDir)
		return nil, ListVideosOutput{Videos: videos}, err
	})

	mcp.AddTool(server, &mcp.Tool{
		Name: "estimate_video_upload", Title: "Проверить размер и лимит Диска",
		Description: "Показывает размер видео, свободное место Яндекс Диска и влияние на токены без загрузки.",
		Annotations: &mcp.ToolAnnotations{ReadOnlyHint: readOnly, DestructiveHint: &notDestructive, OpenWorldHint: &openWorld},
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in EstimateInput) (*mcp.CallToolResult, EstimateOutput, error) {
		video, _, err := resolveVideo(cfg.VideoDir, in.VideoName)
		if err != nil {
			return nil, EstimateOutput{}, err
		}
		result := estimate(video)
		if strings.TrimSpace(cfg.OAuthToken) != "" {
			if quota, quotaErr := diskQuota(ctx, cfg); quotaErr == nil {
				result.DiskTotalBytes = quota.TotalSpace
				result.DiskUsedBytes = quota.UsedSpace
				result.DiskFreeBytes = max(0, quota.TotalSpace-quota.UsedSpace)
				fits := video.SizeBytes <= result.DiskFreeBytes
				result.FitsAvailableSpace = &fits
			} else {
				result.Notes = append(result.Notes, "Не удалось получить квоту Диска: "+quotaErr.Error())
			}
		}
		return nil, result, nil
	})

	mcp.AddTool(server, &mcp.Tool{
		Name: "upload_video_to_yandex", Title: "Загрузить видео на Яндекс Диск",
		Description: "Потоково загружает запись в AI Challenge/lession N после подтверждения.",
		Annotations: &mcp.ToolAnnotations{ReadOnlyHint: false, DestructiveHint: &notDestructive, OpenWorldHint: &openWorld},
	}, func(ctx context.Context, _ *mcp.CallToolRequest, in UploadInput) (*mcp.CallToolResult, UploadOutput, error) {
		if !in.Confirm {
			return nil, UploadOutput{}, errors.New("загрузка требует явного confirm=true")
		}
		result, err := upload(ctx, cfg, in)
		return nil, result, err
	})

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, serverTransport, nil)
	if err != nil {
		return nil, fmt.Errorf("connect MCP server: %w", err)
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "ai-challenge-agent", Version: "1.0.0"}, nil)
	clientSession, err := client.Connect(ctx, clientTransport, nil)
	if err != nil {
		_ = serverSession.Close()
		return nil, fmt.Errorf("connect MCP client: %w", err)
	}
	return &Bridge{clientSession: clientSession, serverSession: serverSession, config: cfg}, nil
}

func (b *Bridge) Close() error {
	var errs []error
	if b.clientSession != nil {
		errs = append(errs, b.clientSession.Close())
	}
	if b.serverSession != nil {
		errs = append(errs, b.serverSession.Close())
	}
	return errors.Join(errs...)
}

func (b *Bridge) Status(ctx context.Context) (Status, error) {
	result, err := b.clientSession.ListTools(ctx, nil)
	if err != nil {
		return Status{}, fmt.Errorf("list MCP tools: %w", err)
	}
	views := make([]ToolView, 0, len(result.Tools))
	for _, tool := range result.Tools {
		view := ToolView{Name: tool.Name, Title: tool.Title, Description: tool.Description, InputSchema: tool.InputSchema}
		if tool.Annotations != nil {
			view.ReadOnly = tool.Annotations.ReadOnlyHint
		}
		views = append(views, view)
	}
	definitionJSON, _ := json.Marshal(views)
	return Status{
		Connected: true, Server: "yandex-disk-video-mcp 1.0.0",
		Configured: strings.TrimSpace(b.config.OAuthToken) != "", RootFolder: b.config.RootFolder,
		VideoDirectory: b.config.VideoDir, Tools: views,
		EstimatedDefinitionTokens: estimateTextTokens(len(definitionJSON)), ModelTokensUsedByThisRequest: 0,
		TokenExplanation: "tools/list выполнен локально без обращения к модели; видео не передаётся модели и не расходует её токены.",
	}, nil
}

func (b *Bridge) Call(ctx context.Context, name string, arguments map[string]any) (CallResponse, error) {
	result, err := b.clientSession.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: arguments})
	if err != nil {
		return CallResponse{}, fmt.Errorf("call MCP tool: %w", err)
	}
	var texts []string
	for _, content := range result.Content {
		if text, ok := content.(*mcp.TextContent); ok {
			texts = append(texts, text.Text)
		}
	}
	payload, _ := json.Marshal(struct {
		Arguments map[string]any `json:"arguments"`
		Result    any            `json:"result"`
	}{arguments, result.StructuredContent})
	return CallResponse{
		Name: name, IsError: result.IsError, Content: strings.Join(texts, "\n"), StructuredContent: result.StructuredContent,
		EstimatedPayloadTokens: estimateTextTokens(len(payload)), ModelTokensUsedByThisRequest: 0,
	}, nil
}

// ToolsForModel obtains the live catalogue through MCP tools/list and converts
// it to the function-calling shape accepted by DeepSeek.
func (b *Bridge) ToolsForModel(ctx context.Context) ([]models.ToolDefinition, error) {
	result, err := b.clientSession.ListTools(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("list MCP tools for model: %w", err)
	}
	definitions := make([]models.ToolDefinition, 0, len(result.Tools))
	for _, tool := range result.Tools {
		definitions = append(definitions, models.ToolDefinition{
			Type:     "function",
			Function: models.ToolFunction{Name: tool.Name, Description: tool.Description, Parameters: tool.InputSchema},
		})
	}
	return definitions, nil
}

// CallForModel executes a validated tool through the same MCP client session
// used by the web diagnostics. The result is returned as JSON text for a tool
// message; binary video bytes never enter the model context.
func (b *Bridge) CallForModel(ctx context.Context, name string, arguments map[string]any) (string, bool, error) {
	result, err := b.Call(ctx, name, arguments)
	if err != nil {
		return "", true, err
	}
	if result.StructuredContent != nil {
		encoded, marshalErr := json.Marshal(result.StructuredContent)
		if marshalErr != nil {
			return "", true, marshalErr
		}
		return string(encoded), result.IsError, nil
	}
	return result.Content, result.IsError, nil
}

func listVideos(directory string) ([]Video, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, fmt.Errorf("прочитать папку с видео: %w", err)
	}
	videos := make([]Video, 0)
	for _, entry := range entries {
		if entry.IsDir() || !videoExtensions[strings.ToLower(filepath.Ext(entry.Name()))] {
			continue
		}
		info, err := entry.Info()
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		videos = append(videos, Video{Name: entry.Name(), SizeBytes: info.Size(), ModifiedAt: info.ModTime()})
	}
	sort.Slice(videos, func(i, j int) bool { return videos[i].ModifiedAt.After(videos[j].ModifiedAt) })
	return videos, nil
}

func resolveVideo(directory, name string) (Video, string, error) {
	name = strings.TrimSpace(name)
	if name == "" || filepath.Base(name) != name || !videoExtensions[strings.ToLower(filepath.Ext(name))] {
		return Video{}, "", errors.New("выберите имя видео из list_desktop_videos")
	}
	path := filepath.Join(directory, name)
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return Video{}, "", errors.New("выбранное видео не найдено")
	}
	return Video{Name: name, SizeBytes: info.Size(), ModifiedAt: info.ModTime()}, path, nil
}

func estimate(video Video) EstimateOutput {
	return EstimateOutput{
		VideoName: video.Name, SizeBytes: video.SizeBytes, SizeGiB: float64(video.SizeBytes) / float64(gib),
		APIRequestsForUpload: 2, VideoBytesSentToModel: 0,
		Notes: []string{
			"Загрузка использует два API-запроса: получение upload URL и отправку файла.",
			"Яндекс Диск ограничен доступным местом вашего тарифа, а не классами хранения Object Storage.",
			"Бинарные байты видео передаются напрямую на Яндекс Диск и не входят в контекст модели.",
		},
	}
}

func analyzeVideo(ctx context.Context, cfg Config, video Video, path string) AnalyzeOutput {
	result := AnalyzeOutput{
		VideoName: video.Name, SizeBytes: video.SizeBytes, ContentAnalyzed: false, VideoBytesSentToModel: 0,
		Notes: []string{"Анализируются только технические метаданные; кадры и речь не передаются модели."},
	}
	if cfg.FFprobePath == "" {
		result.Notes = append(result.Notes, "ffprobe не найден, поэтому доступны только имя и размер файла.")
		return result
	}
	command := exec.CommandContext(ctx, cfg.FFprobePath, "-v", "error", "-show_entries", "format=duration,format_name:stream=codec_type,codec_name,width,height,r_frame_rate", "-of", "json", path)
	output, err := command.Output()
	if err != nil {
		result.Notes = append(result.Notes, "ffprobe не смог прочитать технические метаданные: "+err.Error())
		return result
	}
	var probe struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
			CodecName string `json:"codec_name"`
			Width     int    `json:"width"`
			Height    int    `json:"height"`
			FrameRate string `json:"r_frame_rate"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
			Name     string `json:"format_name"`
		} `json:"format"`
	}
	if err := json.Unmarshal(output, &probe); err != nil {
		result.Notes = append(result.Notes, "Не удалось разобрать ответ ffprobe: "+err.Error())
		return result
	}
	result.Container = probe.Format.Name
	result.DurationSeconds, _ = strconv.ParseFloat(probe.Format.Duration, 64)
	for _, stream := range probe.Streams {
		switch stream.CodecType {
		case "video":
			result.VideoCodec, result.Width, result.Height = stream.CodecName, stream.Width, stream.Height
			result.FramesPerSecond = parseFrameRate(stream.FrameRate)
		case "audio":
			result.AudioCodec = stream.CodecName
		}
	}
	return result
}

func parseFrameRate(value string) float64 {
	parts := strings.Split(value, "/")
	if len(parts) != 2 {
		result, _ := strconv.ParseFloat(value, 64)
		return result
	}
	numerator, _ := strconv.ParseFloat(parts[0], 64)
	denominator, _ := strconv.ParseFloat(parts[1], 64)
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

type quotaResponse struct {
	TotalSpace int64 `json:"total_space"`
	UsedSpace  int64 `json:"used_space"`
}

func diskQuota(ctx context.Context, cfg Config) (quotaResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(cfg.Endpoint, "/"), nil)
	if err != nil {
		return quotaResponse{}, err
	}
	req.Header.Set("Authorization", "OAuth "+strings.TrimSpace(cfg.OAuthToken))
	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return quotaResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return quotaResponse{}, yandexAPIError("получение квоты", resp)
	}
	var result quotaResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return quotaResponse{}, fmt.Errorf("прочитать квоту Диска: %w", err)
	}
	return result, nil
}

func upload(ctx context.Context, cfg Config, in UploadInput) (UploadOutput, error) {
	if strings.TrimSpace(cfg.OAuthToken) == "" {
		return UploadOutput{}, errors.New("задайте YANDEX_DISK_OAUTH_TOKEN в .env")
	}
	video, localPath, err := resolveVideo(cfg.VideoDir, in.VideoName)
	if err != nil {
		return UploadOutput{}, err
	}
	lessonFolder, err := safeFolder(in.LessonFolder)
	if err != nil {
		return UploadOutput{}, err
	}
	diskPath := strings.Trim(cfg.RootFolder, "/") + "/" + lessonFolder + "/" + video.Name

	query := url.Values{"path": {diskPath}, "overwrite": {"false"}}
	uploadURL := strings.TrimRight(cfg.Endpoint, "/") + "/resources/upload?" + query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, uploadURL, nil)
	if err != nil {
		return UploadOutput{}, fmt.Errorf("создать запрос URL загрузки: %w", err)
	}
	req.Header.Set("Authorization", "OAuth "+strings.TrimSpace(cfg.OAuthToken))
	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return UploadOutput{}, fmt.Errorf("получить URL загрузки Яндекс Диска: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		if resp.StatusCode == http.StatusConflict && strings.Contains(string(message), "DiskResourceAlreadyExistsError") {
			matches, matchErr := diskResourceMatches(ctx, cfg, diskPath, video.SizeBytes)
			if matchErr == nil && matches {
				return UploadOutput{DiskPath: "disk:/" + diskPath, SizeBytes: video.SizeBytes, AlreadyExisted: true}, nil
			}
		}
		return UploadOutput{}, fmt.Errorf("Яндекс Диск: получение URL загрузки, HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(message)))
	}
	var link struct {
		Href string `json:"href"`
	}
	decodeErr := json.NewDecoder(resp.Body).Decode(&link)
	resp.Body.Close()
	if decodeErr != nil || strings.TrimSpace(link.Href) == "" {
		return UploadOutput{}, errors.New("Яндекс Диск не вернул URL для загрузки")
	}

	file, err := os.Open(localPath)
	if err != nil {
		return UploadOutput{}, fmt.Errorf("открыть видео: %w", err)
	}
	defer file.Close()
	put, err := http.NewRequestWithContext(ctx, http.MethodPut, link.Href, file)
	if err != nil {
		return UploadOutput{}, fmt.Errorf("создать запрос загрузки: %w", err)
	}
	put.ContentLength = video.SizeBytes
	put.Header.Set("Content-Type", contentType(video.Name))
	putResp, err := cfg.HTTPClient.Do(put)
	if err != nil {
		return UploadOutput{}, fmt.Errorf("загрузить видео на Яндекс Диск: %w", err)
	}
	defer putResp.Body.Close()
	if putResp.StatusCode < 200 || putResp.StatusCode >= 300 {
		return UploadOutput{}, yandexAPIError("загрузка файла", putResp)
	}
	return UploadOutput{DiskPath: "disk:/" + diskPath, SizeBytes: video.SizeBytes}, nil
}

func diskResourceMatches(ctx context.Context, cfg Config, diskPath string, localSize int64) (bool, error) {
	query := url.Values{"path": {diskPath}, "fields": {"size,type"}}
	endpoint := strings.TrimRight(cfg.Endpoint, "/") + "/resources?" + query.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return false, err
	}
	req.Header.Set("Authorization", "OAuth "+strings.TrimSpace(cfg.OAuthToken))
	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return false, yandexAPIError("проверка существующего файла", resp)
	}
	var resource struct {
		Size int64  `json:"size"`
		Type string `json:"type"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&resource); err != nil {
		return false, err
	}
	return resource.Type == "file" && resource.Size == localSize, nil
}

func safeFolder(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", errors.New("укажите подпапку, например lession 16")
	}
	if strings.ContainsAny(value, "/\\") || value == "." || value == ".." {
		return "", errors.New("подпапка должна быть одним именем, например lession 16")
	}
	return value, nil
}

func yandexAPIError(action string, resp *http.Response) error {
	message, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	detail := strings.TrimSpace(string(message))
	if detail == "" {
		detail = http.StatusText(resp.StatusCode)
	}
	return fmt.Errorf("Яндекс Диск: %s, HTTP %d: %s", action, resp.StatusCode, detail)
}

func contentType(name string) string {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".mov":
		return "video/quicktime"
	case ".webm":
		return "video/webm"
	case ".mkv":
		return "video/x-matroska"
	default:
		return "video/mp4"
	}
}

func estimateTextTokens(bytes int) int {
	if bytes == 0 {
		return 0
	}
	return (bytes + 3) / 4
}
