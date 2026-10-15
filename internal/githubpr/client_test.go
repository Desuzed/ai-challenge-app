package githubpr

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func testPull() Pull {
	p := Pull{Number: 7, State: "open"}
	p.Head.SHA, p.Base.SHA = strings.Repeat("a", 40), strings.Repeat("b", 40)
	p.Head.Repo.FullName, p.Base.Repo.FullName = "owner/repo", "owner/repo"
	return p
}

func clientWithHandler(handler http.HandlerFunc) *Client {
	return &Client{Token: "private-token", BaseURL: "https://github.example.test", HTTP: &http.Client{Transport: transportFunc(func(req *http.Request) (*http.Response, error) {
		w := httptest.NewRecorder()
		handler(w, req)
		return w.Result(), nil
	})}}
}

func TestPublishAnchorsToSHAAndUsesComment(t *testing.T) {
	p := testPull()
	posts := 0
	c := clientWithHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer private-token" || r.Header.Get("X-GitHub-Api-Version") == "" {
			t.Fatal("missing authenticated API headers")
		}
		if r.Method == http.MethodPost {
			posts++
			var body map[string]string
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body["event"] != "COMMENT" || body["commit_id"] != p.Head.SHA || !strings.HasPrefix(body["body"], marker(p.Head.SHA, p.Base.SHA)) {
				t.Fatalf("invalid review payload: %v", body)
			}
			w.WriteHeader(201)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/reviews") {
			io.WriteString(w, `[]`)
			return
		}
		json.NewEncoder(w).Encode(p)
	})
	posted, err := c.Publish(context.Background(), "owner/repo", p, "review body")
	if err != nil || !posted || posts != 1 {
		t.Fatalf("Publish=%v %v, posts=%d", posted, err, posts)
	}
}

func TestPublishRefusesStaleClosedDraftAndForkPR(t *testing.T) {
	for _, change := range []string{"head", "base", "closed", "draft", "fork"} {
		t.Run(change, func(t *testing.T) {
			original, current := testPull(), testPull()
			switch change {
			case "head":
				current.Head.SHA = strings.Repeat("c", 40)
			case "base":
				current.Base.SHA = strings.Repeat("c", 40)
			case "closed":
				current.State = "closed"
			case "draft":
				current.Draft = true
			case "fork":
				current.Head.Repo.FullName = "external/repo"
			}
			c := clientWithHandler(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet || strings.HasSuffix(r.URL.Path, "/reviews") {
					t.Fatal("attempted to publish invalid/stale review")
				}
				json.NewEncoder(w).Encode(current)
			})
			if _, err := c.Publish(context.Background(), "owner/repo", original, "review"); err == nil {
				t.Fatal("accepted invalid PR")
			}
		})
	}
}

func TestPublishDeduplicatesBotReviewsWithPagination(t *testing.T) {
	p := testPull()
	pages := 0
	c := clientWithHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Fatal("published a duplicate")
		}
		if !strings.HasSuffix(r.URL.Path, "/reviews") {
			json.NewEncoder(w).Encode(p)
			return
		}
		pages++
		if r.URL.Query().Get("page") == "1" {
			json.NewEncoder(w).Encode(make([]map[string]string, 100))
			return
		}
		json.NewEncoder(w).Encode([]any{map[string]any{"body": marker(p.Head.SHA, p.Base.SHA) + "\nold review", "user": map[string]string{"login": "github-actions[bot]"}}})
	})
	posted, err := c.Publish(context.Background(), "owner/repo", p, "review")
	if err != nil || posted || pages != 2 {
		t.Fatalf("deduplication failed: posted=%v err=%v pages=%d", posted, err, pages)
	}
}

func TestAPIErrorsNeverExposeBodiesOrToken(t *testing.T) {
	c := clientWithHandler(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(403)
		io.WriteString(w, "private-token private source body")
	})
	_, err := c.Get(context.Background(), "owner/repo", 7)
	if err == nil || strings.Contains(err.Error(), "private") {
		t.Fatalf("unsafe error: %v", err)
	}
	c.HTTP.Transport = transportFunc(func(*http.Request) (*http.Response, error) { return nil, errors.New("private-token") })
	_, err = c.Get(context.Background(), "owner/repo", 7)
	if err == nil || strings.Contains(err.Error(), "private") {
		t.Fatalf("unsafe transport error: %v", err)
	}
	if _, err := c.Get(context.Background(), "../repo", 7); err == nil {
		t.Fatal("accepted unsafe repository")
	}
}
