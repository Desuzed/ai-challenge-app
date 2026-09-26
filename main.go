package main

import (
	"context"
	"errors"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"ai-challenge-app/internal/agent"
	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/githubmcp"
	"ai-challenge-app/internal/handlers"
	"ai-challenge-app/internal/mcpbridge"
	"ai-challenge-app/internal/openrouter"
	"ai-challenge-app/internal/weathermcp"
)

const (
	localEnvFile     = ".env"
	apiKeyEnvVar     = "DEEPSEEK_API_KEY"
	agentHistoryFile = ".local/agent-history.json"
	// Flash can spend longer than a short HTTP timeout preparing a first
	// completion, while the lightweight /models endpoint remains fast.
	deepSeekRequestTimeout = 120 * time.Second
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	apiKey, err := loadDeepSeekAPIKey(os.Getenv, os.ReadFile)
	if err != nil {
		log.Print("warning: local API key file could not be read; API key may be unavailable")
	}

	client := deepseek.NewClient(apiKey, deepSeekRequestTimeout)
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(filepath.Join(".", "static"))))
	handler := handlers.New(client)
	persistentAgent, err := agent.NewPersistent(client, agent.NewJSONStore(agentHistoryFile))
	if err != nil {
		log.Fatalf("load agent history: %v", err)
	}
	handler.SetAgent(persistentAgent)
	handler.SetOpenRouterClient(openrouter.NewClient(os.Getenv("OPENROUTER_API_KEY"), 120*time.Second))
	homeDirectory, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("find home directory: %v", err)
	}
	videoDirectory := localSetting("VIDEO_SOURCE_DIR", os.Getenv, os.ReadFile)
	if videoDirectory == "" {
		videoDirectory = filepath.Join(homeDirectory, "Desktop")
	}
	yandexBridge, err := mcpbridge.New(context.Background(), mcpbridge.Config{
		OAuthToken: localSetting("YANDEX_DISK_OAUTH_TOKEN", os.Getenv, os.ReadFile),
		RootFolder: localSetting("YANDEX_DISK_ROOT", os.Getenv, os.ReadFile),
		VideoDir:   videoDirectory,
	})
	if err != nil {
		log.Fatalf("connect Yandex MCP: %v", err)
	}
	githubRepository := localSetting("GITHUB_REPOSITORY", os.Getenv, os.ReadFile)
	if githubRepository == "" {
		githubRepository = "Desuzed/ai-challenge-app"
	}
	githubBridge, err := githubmcp.New(context.Background(), githubmcp.Config{
		Token: localSetting("GITHUB_TOKEN", os.Getenv, os.ReadFile), Repository: githubRepository,
	})
	if err != nil {
		log.Fatalf("connect GitHub MCP: %v", err)
	}
	weatherBridge, err := weathermcp.New(context.Background(), weathermcp.Config{DataFile: localSetting("WEATHER_DATA_FILE", os.Getenv, os.ReadFile), Interval: weatherInterval()})
	if err != nil {
		log.Fatalf("connect Weather MCP: %v", err)
	}
	mcpCatalog := mcpbridge.NewCatalog(yandexBridge, githubBridge, weatherBridge)
	defer mcpCatalog.Close()
	persistentAgent.SetToolRuntime(mcpCatalog)
	mux.Handle("/api/chat", http.HandlerFunc(handler.Chat))
	mux.Handle("/api/reasoning", http.HandlerFunc(handler.Reasoning))
	mux.Handle("/api/model-versions", http.HandlerFunc(handler.ModelVersions))
	mux.Handle("/api/agent/chat", http.HandlerFunc(handler.AgentChat))
	mux.Handle("/api/agent/models", http.HandlerFunc(handler.AgentModels))
	mux.Handle("/api/agent/token-demo", http.HandlerFunc(handler.TokenDemo))
	mux.Handle("/api/agent/strategy-demo", http.HandlerFunc(handler.ContextStrategyDemo))
	mux.Handle("/api/agent/branching-demo", http.HandlerFunc(handler.BranchingDemo))
	mux.Handle("/api/mcp/tools", http.HandlerFunc(mcpCatalog.ToolsHTTP))
	mux.Handle("/api/mcp/call", http.HandlerFunc(mcpCatalog.CallHTTP))

	server := &http.Server{
		Addr:              "127.0.0.1:" + port,
		Handler:           securityHeaders(mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		// The prompt-designer mode makes two sequential API calls and can take
		// longer than the single-response lessons.
		// Lesson 5 can make two slower model calls in sequence (Flash and Pro).
		WriteTimeout: 30 * time.Minute,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("AI Challenge App listening on http://localhost:%s", port)
	log.Fatal(server.ListenAndServe())
}

func weatherInterval() time.Duration {
	seconds, err := strconv.Atoi(localSetting("WEATHER_COLLECTION_INTERVAL_SECONDS", os.Getenv, os.ReadFile))
	if err != nil || seconds < 1 {
		return time.Minute
	}
	return time.Duration(seconds) * time.Second
}

func loadDeepSeekAPIKey(lookupEnv func(string) string, readFile func(string) ([]byte, error)) (string, error) {
	if apiKey := strings.TrimSpace(lookupEnv(apiKeyEnvVar)); apiKey != "" {
		return apiKey, nil
	}

	data, err := readFile(localEnvFile)
	if errors.Is(err, fs.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return deepSeekAPIKeyFromEnv(data), nil
}

func deepSeekAPIKeyFromEnv(data []byte) string {
	return valueFromEnvFile(data, apiKeyEnvVar)
}

func localSetting(name string, lookupEnv func(string) string, readFile func(string) ([]byte, error)) string {
	if value := strings.TrimSpace(lookupEnv(name)); value != "" {
		return value
	}
	data, err := readFile(localEnvFile)
	if err != nil {
		return ""
	}
	return valueFromEnvFile(data, name)
}

func valueFromEnvFile(data []byte, wanted string) string {
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		name, value, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(name) == wanted {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The exercises are edited locally while the server keeps running. Do not
		// let a browser retain an older HTML/JS bundle and its stale demo cards.
		w.Header().Set("Cache-Control", "no-store, max-age=0")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
		next.ServeHTTP(w, r)
	})
}
