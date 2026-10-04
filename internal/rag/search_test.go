package rag

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"ai-challenge-app/internal/ragindex"
)

type fakeEmbedder struct{ vector []float64 }

func (f fakeEmbedder) Embed(_ context.Context, _ []string) ([][]float64, error) {
	return [][]float64{f.vector}, nil
}

func TestSearchRanksIndexedChunksAndDetectsChanges(t *testing.T) {
	root := t.TempDir()
	text := "# First\n" + strings.Repeat("First section content. ", 12) + "\n# Second\n" + strings.Repeat("Second section content. ", 12)
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	docs, err := ragindex.Load(root, []string{"guide.md"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ragindex.ChunkDocuments(docs, "structure")
	if err != nil || len(chunks) != 2 {
		t.Fatalf("chunks=%d, err=%v", len(chunks), err)
	}
	chunks[0].Vector = []float64{0, 1}
	chunks[1].Vector = []float64{1, 0}
	index := ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks}
	data, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	indexDir := filepath.Join(root, "index")
	if err := os.Mkdir(indexDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(indexDir, "index-structure.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	searcher := &Searcher{Root: root, IndexDir: indexDir, Model: "test-model", Embedder: fakeEmbedder{[]float64{1, 0}}}
	matches, err := searcher.Search(context.Background(), "second section", 2)
	if err != nil {
		t.Fatal(err)
	}
	if matches[0].Chunk.Section != "Second" || matches[0].Score < .99 {
		t.Fatalf("top match=%#v", matches[0])
	}
	if !strings.Contains(Context(matches[:1]), "guide.md: Second") {
		t.Fatal("context omitted source metadata")
	}
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte(text+"\nchanged"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := searcher.Search(context.Background(), "second section", 2); !errors.Is(err, ErrIndexStale) {
		t.Fatalf("stale index error=%v", err)
	}
}

func TestSearchWithOptionsRewritesFiltersAndReranks(t *testing.T) {
	root := t.TempDir()
	history := "История агента хранится в agent-history.json. "
	weather := "Погода Москвы собирается каждую минуту. "
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte("# История\n"+strings.Repeat(history, 5)+"\n# Другое\n"+strings.Repeat(weather, 5)), 0600); err != nil {
		t.Fatal(err)
	}
	docs, err := ragindex.Load(root, []string{"guide.md"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ragindex.ChunkDocuments(docs, "structure")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks=%d", len(chunks))
	}
	// Vector search slightly prefers the weather chunk. The lexical second stage
	// must restore the history chunk because the rewritten query expands it.
	chunks[0].Vector, chunks[1].Vector = []float64{0.95, 0.05}, []float64{1, 0}
	data, _ := json.Marshal(ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks})
	indexDir := filepath.Join(root, "index")
	if err := os.Mkdir(indexDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(indexDir, "index-structure.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	s := &Searcher{Root: root, IndexDir: indexDir, Model: "test-model", Embedder: fakeEmbedder{[]float64{1, 0}}}
	result, err := s.SearchWithOptions(context.Background(), "Где история диалога?", SearchOptions{CandidateLimit: 2, ResultLimit: 1, MinSimilarity: 0.5, Rewrite: true, Rerank: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.RewrittenQuery, "agent") {
		t.Fatalf("rewrite=%q", result.RewrittenQuery)
	}
	if len(result.Candidates) != 2 || len(result.Matches) != 1 || result.Matches[0].Chunk.Section != "История" {
		t.Fatalf("result=%#v", result)
	}
	if result.Matches[0].RerankScore <= 0 {
		t.Fatal("reranker score is missing")
	}
}
