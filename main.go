package main

import (
	"errors"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/handlers"
)

const (
	localEnvFile = ".env"
	apiKeyEnvVar = "DEEPSEEK_API_KEY"
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

	client := deepseek.NewClient(apiKey, 45*time.Second)
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(filepath.Join(".", "static"))))
	handler := handlers.New(client)
	mux.Handle("/api/chat", http.HandlerFunc(handler.Chat))
	mux.Handle("/api/reasoning", http.HandlerFunc(handler.Reasoning))

	server := &http.Server{
		Addr:              "127.0.0.1:" + port,
		Handler:           securityHeaders(mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		// The prompt-designer mode makes two sequential API calls and can take
		// longer than the single-response lessons.
		WriteTimeout: 100 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("AI Challenge App listening on http://localhost:%s", port)
	log.Fatal(server.ListenAndServe())
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
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		name, value, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(name) == apiKeyEnvVar {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
		next.ServeHTTP(w, r)
	})
}
