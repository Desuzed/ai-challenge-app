package main

import (
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/handlers"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	client := deepseek.NewClient(os.Getenv("DEEPSEEK_API_KEY"), 45*time.Second)
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir(filepath.Join(".", "static"))))
	mux.Handle("/api/chat", http.HandlerFunc(handlers.New(client).Chat))

	server := &http.Server{
		Addr:              "127.0.0.1:" + port,
		Handler:           securityHeaders(mux),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	log.Printf("AI Challenge App listening on http://localhost:%s", port)
	log.Fatal(server.ListenAndServe())
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
