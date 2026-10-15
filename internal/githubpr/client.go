// Package githubpr provides the narrow PR API needed by the review CLI. The
// model never receives the GitHub token or access to write tools.
package githubpr

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type Client struct {
	Token   string
	BaseURL string
	HTTP    *http.Client
}

type Revision struct {
	SHA  string `json:"sha"`
	Repo struct {
		FullName string `json:"full_name"`
	} `json:"repo"`
}

type Pull struct {
	Number int      `json:"number"`
	State  string   `json:"state"`
	Draft  bool     `json:"draft"`
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	Head   Revision `json:"head"`
	Base   Revision `json:"base"`
}

var repositoryPattern = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`)
var commitPattern = regexp.MustCompile(`^(?:[a-f0-9]{40}|[a-f0-9]{64})$`)

func ValidRepository(repo string) bool {
	return repositoryPattern.MatchString(repo) && !strings.Contains(repo, "..")
}

func prPath(repo string, number int) (string, error) {
	if !ValidRepository(repo) || number < 1 {
		return "", errors.New("expected repository owner/name and positive PR number")
	}
	return fmt.Sprintf("/repos/%s/pulls/%d", repo, number), nil
}

func (c *Client) request(ctx context.Context, method, path string, body, result any) error {
	var payload []byte
	var err error
	if body != nil {
		payload, err = json.Marshal(body)
		if err != nil {
			return err
		}
	}
	base := c.BaseURL
	if base == "" {
		base = "https://api.github.com"
	}
	client := &http.Client{Timeout: 45 * time.Second}
	if c.HTTP != nil {
		*client = *c.HTTP
	}
	client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return errors.New("GitHub API redirects are disabled")
	}
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(base, "/")+path, bytes.NewReader(payload))
		if err != nil {
			return err
		}
		req.Header.Set("Accept", "application/vnd.github+json")
		req.Header.Set("X-GitHub-Api-Version", "2026-03-10")
		req.Header.Set("User-Agent", "ai-challenge-app-prreview")
		if c.Token != "" {
			req.Header.Set("Authorization", "Bearer "+c.Token)
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := client.Do(req)
		if err != nil {
			return errors.New("GitHub API request failed")
		}
		data, readErr := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024+1))
		resp.Body.Close()
		if method == http.MethodGet && attempt < 2 && (resp.StatusCode == 429 || resp.StatusCode == 502 || resp.StatusCode == 503) {
			timer := time.NewTimer(time.Duration(attempt+1) * time.Second)
			select {
			case <-ctx.Done():
				timer.Stop()
				return ctx.Err()
			case <-timer.C:
				continue
			}
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			// Never log response bodies, which can contain credentials or source.
			return fmt.Errorf("GitHub API HTTP %d", resp.StatusCode)
		}
		if readErr != nil || len(data) > 4*1024*1024 {
			return errors.New("invalid or oversized GitHub response")
		}
		if result != nil {
			if err := json.Unmarshal(data, result); err != nil {
				return errors.New("invalid GitHub JSON response")
			}
		}
		return nil
	}
}

func (c *Client) Get(ctx context.Context, repo string, number int) (Pull, error) {
	var p Pull
	path, err := prPath(repo, number)
	if err != nil {
		return p, err
	}
	if err := c.request(ctx, http.MethodGet, path, nil, &p); err != nil {
		return p, err
	}
	if p.Number != number || !commitPattern.MatchString(p.Head.SHA) || !commitPattern.MatchString(p.Base.SHA) || !strings.EqualFold(p.Base.Repo.FullName, repo) {
		return p, errors.New("GitHub PR metadata does not match repository/revisions")
	}
	return p, nil
}

func marker(head, base string) string {
	return "<!-- ai-prreview:" + head + ":" + base + " -->"
}

// Publish rechecks both revisions, uses COMMENT (never approves or blocks a
// merge), and deduplicates the workflow bot's reviews for the same revision pair.
func (c *Client) Publish(ctx context.Context, repo string, original Pull, body string) (bool, error) {
	if c.Token == "" {
		return false, errors.New("GITHUB_TOKEN is required to publish")
	}
	if len(body) > 60000 || strings.TrimSpace(body) == "" {
		return false, errors.New("review body must contain 1 to 60000 bytes")
	}
	path, err := prPath(repo, original.Number)
	if err != nil {
		return false, err
	}
	p, err := c.Get(ctx, repo, original.Number)
	if err != nil {
		return false, err
	}
	if p.State != "open" || p.Draft || p.Head.SHA != original.Head.SHA || p.Base.SHA != original.Base.SHA {
		return false, errors.New("PR closed, became draft, or changed revisions; refusing stale review")
	}
	if !strings.EqualFold(p.Head.Repo.FullName, repo) {
		return false, errors.New("fork PR publication is not enabled")
	}
	tag := marker(p.Head.SHA, p.Base.SHA)
	for page := 1; page <= 10; page++ {
		var reviews []struct {
			Body string `json:"body"`
			User struct {
				Login string `json:"login"`
			} `json:"user"`
		}
		if err := c.request(ctx, http.MethodGet, fmt.Sprintf("%s/reviews?per_page=100&page=%d", path, page), nil, &reviews); err != nil {
			return false, err
		}
		for _, review := range reviews {
			if review.User.Login == "github-actions[bot]" && strings.HasPrefix(review.Body, tag) {
				return false, nil
			}
		}
		if len(reviews) < 100 {
			break
		}
		if page == 10 {
			return false, errors.New("review pagination limit reached; cannot verify deduplication")
		}
	}
	err = c.request(ctx, http.MethodPost, path+"/reviews", map[string]string{
		"commit_id": original.Head.SHA, "event": "COMMENT", "body": tag + "\n" + body,
	}, nil)
	return err == nil, err
}
