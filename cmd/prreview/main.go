// Command prreview reviews local Git revisions or a GitHub PR without starting
// the web app. Publishing is opt-in; the default is a private local report.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"ai-challenge-app/internal/deepseek"
	"ai-challenge-app/internal/githubpr"
	"ai-challenge-app/internal/models"
	"ai-challenge-app/internal/prreview"
	"ai-challenge-app/internal/ragindex"
)

func main() {
	if err := run(); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run() error {
	root := flag.String("root", ".", "Git repository containing the reviewed revisions")
	base := flag.String("base", "", "base revision; in PR mode also checks the expected base SHA")
	head := flag.String("head", "", "head revision; in PR mode also checks the expected head SHA")
	repo := flag.String("repo", os.Getenv("GITHUB_REPOSITORY"), "GitHub owner/repository")
	pr := flag.Int("pr", 0, "GitHub PR number; metadata is read via GitHub API")
	publish := flag.Bool("publish", false, "publish a COMMENT review to GitHub (requires -pr)")
	title := flag.String("title", "Code review", "change intent for local revision mode")
	output := flag.String("output", ".local/prreview", "directory for review.md and review.json")
	maxFiles := flag.Int("max-files", 20, "maximum reviewed files per run (1-100)")
	model := flag.String("model", models.DeepSeekFlashModel, "DeepSeek model (only deepseek-flash is supported)")
	embedModel := flag.String("embedding-model", "embeddinggemma", "local Ollama embedding model")
	ollamaURL := flag.String("ollama-url", "", "local Ollama URL (otherwise OLLAMA_URL or loopback default)")
	timeout := flag.Duration("timeout", 20*time.Minute, "whole review timeout")
	flag.Parse()
	if *model != models.DeepSeekFlashModel {
		return errors.New("PR review is restricted to deepseek-flash")
	}
	if flag.NArg() != 0 || *pr < 0 || (*publish && *pr == 0) || *timeout <= 0 {
		return errors.New("invalid arguments; use -help (publishing requires -pr)")
	}
	if *repo != "" && !githubpr.ValidRepository(*repo) {
		return errors.New("invalid -repo; expected owner/repository")
	}
	rootAbs, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	key := setting(rootAbs, "DEEPSEEK_API_KEY")
	if key == "" {
		return errors.New("DEEPSEEK_API_KEY is required in environment or local .env")
	}
	if *ollamaURL == "" {
		*ollamaURL = setting(rootAbs, "OLLAMA_URL")
		if *ollamaURL == "" {
			*ollamaURL = "http://127.0.0.1:11434"
		}
	}
	parent, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(parent, *timeout)
	defer cancel()
	gh := &githubpr.Client{Token: setting(rootAbs, "GITHUB_TOKEN")}
	var pull githubpr.Pull
	description := ""
	if *pr > 0 {
		pull, err = gh.Get(ctx, *repo, *pr)
		if err != nil {
			return err
		}
		if pull.State != "open" || pull.Draft || !strings.EqualFold(pull.Head.Repo.FullName, *repo) {
			return errors.New("only open, non-draft PRs from the same repository are supported")
		}
		if (*head != "" && *head != pull.Head.SHA) || (*base != "" && *base != pull.Base.SHA) {
			return errors.New("PR revisions changed since workflow event; refusing stale analysis")
		}
		*base, *head, *title, description = pull.Base.SHA, pull.Head.SHA, pull.Title, pull.Body
	}
	if *base == "" || *head == "" {
		return errors.New("use -base and -head for local review, or -repo and -pr for GitHub")
	}
	client := deepseek.NewClient(key, 120*time.Second)
	zero := 0.0
	r, err := prreview.Run(ctx, prreview.Options{
		Root: rootAbs, Base: *base, Head: *head, Title: *title, Description: description, Repository: *repo,
		Model: *model, EmbeddingModel: *embedModel, MaxFiles: *maxFiles,
		Embedder: ragindex.Ollama{URL: *ollamaURL, Model: *embedModel}, Logf: log.Printf,
		Complete: func(ctx context.Context, system, prompt string) (string, error) {
			completion, err := client.CompleteMessagesModelJSON(ctx, *model, []models.ChatMessage{
				{Role: "system", Content: system}, {Role: "user", Content: prompt},
			}, models.GenerationSettings{Temperature: &zero, MaxTokens: 4096})
			if err != nil {
				return "", err
			}
			if completion.FinishReason != "stop" {
				return "", fmt.Errorf("model did not finish normally (%s)", completion.FinishReason)
			}
			return completion.Answer, nil
		},
	})
	if err != nil {
		return err
	}
	markdown := prreview.Markdown(r)
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	out := *output
	if !filepath.IsAbs(out) {
		out = filepath.Join(rootAbs, out)
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		return err
	}
	for name, content := range map[string][]byte{"review.json": append(data, '\n'), "review.md": []byte(markdown)} {
		if err := os.WriteFile(filepath.Join(out, name), content, 0600); err != nil {
			return err
		}
	}
	fmt.Printf("Review saved: %s\nFindings: %d\n", filepath.Join(out, "review.md"), len(r.Findings))
	if *publish {
		published, err := gh.Publish(ctx, *repo, pull, markdown)
		if err != nil {
			return err
		}
		if published {
			fmt.Println("GitHub review published")
		} else {
			fmt.Println("This revision pair already has a workflow review; no duplicate published")
		}
	}
	return nil
}

// Read only the requested setting, without evaluating shell syntax. CI never
// reads .env from a checkout. Nonempty process environment always wins.
func setting(root, key string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value == "" {
		return value
	}
	if os.Getenv("CI") != "" {
		return ""
	}
	data, err := os.ReadFile(filepath.Join(root, ".env"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		name, value, ok := strings.Cut(strings.TrimSpace(line), "=")
		if ok && strings.TrimSpace(name) == key {
			return strings.Trim(strings.TrimSpace(value), "\"'")
		}
	}
	return ""
}
