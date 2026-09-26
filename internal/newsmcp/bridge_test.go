package newsmcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSearchNewsReturnsRawArticlesForRequestedPeriod(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("q"); !strings.Contains(got, "Москва") || !strings.Contains(got, "ИИ") {
			t.Fatalf("query=%q", got)
		}
		_, _ = w.Write([]byte(`<?xml version="1.0"?><rss><channel><item><title>Первая новость</title><link>https://example.test/1</link><pubDate>Tue, 02 Sep 2026 10:00:00 +0000</pubDate><source>example.test</source><description>Фрагмент</description></item></channel></rss>`))
	}))
	defer server.Close()
	bridge, err := New(context.Background(), Config{Endpoint: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	defer bridge.Close()
	raw, failed, err := bridge.CallForModel(context.Background(), "search_news", map[string]any{"city": "Москва", "query": "ИИ", "from": "2026-09-01", "to": "2026-09-03"})
	if err != nil || failed {
		t.Fatalf("raw=%s failed=%v err=%v", raw, failed, err)
	}
	var out searchOutput
	if err := json.Unmarshal([]byte(raw), &out); err != nil || len(out.Articles) != 1 || out.Articles[0].Source != "example.test" {
		t.Fatalf("out=%#v err=%v", out, err)
	}
}
