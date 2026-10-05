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

func TestQueryTermsKeepsYoInsideRussianWord(t *testing.T) {
	terms := queryTerms("Где сохраняются два индекса и отчёт?")
	if !containsString(terms, "отчёт") {
		t.Fatalf("query terms lost ё: %#v", terms)
	}
	if containsString(terms, "отч") {
		t.Fatalf("query terms truncated отчёт: %#v", terms)
	}
	if rewritten := RewriteQuery("Где сохраняются два индекса и отчёт?"); !strings.Contains(rewritten, "отчёт") {
		t.Fatalf("rewrite lost full word: %q", rewritten)
	}
}

func TestLexicalGateFiltersQuestionWordsNumbersAndKeepsHistoryWindowDistinct(t *testing.T) {
	wheel := lexicalFeatures("Какой размер колеса?")
	if len(wheel.Terms) != 1 {
		t.Fatalf("question words/size must not manufacture lexical evidence: %#v", wheel.Terms)
	}
	if _, ok := wheel.Terms["колес"]; !ok {
		t.Fatalf("expected only the wheel concept, got %#v", wheel.Terms)
	}

	oneTermAndNumber := lexicalFeatures("overlap 150")
	if len(oneTermAndNumber.Terms) != 1 || !oneTermAndNumber.HasNumber {
		t.Fatalf("numeric evidence must not become a lexical term: %#v", oneTermAndNumber)
	}
	if len(oneTermAndNumber.Terms) >= lexicalGateMinTerms {
		t.Fatal("one specific term plus a number must not pass the lexical gate")
	}

	historyWindow := lexicalFeatures("окно истории sliding window")
	fixedChunk := lexicalFeatures("фиксированные фрагменты fixed chunks")
	if historyWindow.Terms["window"] != true || historyWindow.Terms["chunk"] {
		t.Fatalf("history window was conflated with an indexed chunk: %#v", historyWindow.Terms)
	}
	if fixedChunk.Terms["chunk"] != true || fixedChunk.Terms["window"] {
		t.Fatalf("fixed chunk terms were conflated with a history window: %#v", fixedChunk.Terms)
	}
}

func TestSearchExactFixedChunkQuestionFindsLexicalEvidenceBelowCosineThreshold(t *testing.T) {
	root := t.TempDir()
	document := "# История окна\n" + strings.Repeat("История диалога хранится отдельно от окна запроса. ", 8) +
		"\n# Параметры фрагментации\n" + strings.Repeat("Фиксированные фрагменты имеют размер 1200 символов и перекрытие 150 символов. ", 8) +
		"\n# Размер колеса\n" + strings.Repeat("Размер колеса зависит от модели устройства. ", 8)
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte(document), 0600); err != nil {
		t.Fatal(err)
	}
	docs, err := ragindex.Load(root, []string{"guide.md"})
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := ragindex.ChunkDocuments(docs, "structure")
	if err != nil || len(chunks) < 3 {
		t.Fatalf("chunks=%d, err=%v", len(chunks), err)
	}
	for i := range chunks {
		// Deliberately make vector retrieval prefer unrelated material. The lexical
		// channel must discover the exact paragraph independently and expose that
		// it passed the lexical gate rather than the cosine threshold.
		chunks[i].Vector = []float64{1, 0}
		if strings.Contains(chunks[i].Text, "Фиксированные фрагменты имеют размер") {
			chunks[i].Vector = []float64{0, 1}
		}
	}
	data, err := json.Marshal(ragindex.Index{Strategy: "structure", Model: "test-model", Files: docs, Chunks: chunks})
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
	s := &Searcher{Root: root, IndexDir: indexDir, Model: "test-model", Embedder: fakeEmbedder{[]float64{1, 0}}}
	options := SearchOptions{
		CandidateLimit: 8, ResultLimit: 1, MinSimilarity: 0.9, Rewrite: false, Rerank: false,
	}
	result, err := s.SearchWithOptions(context.Background(), "Какой размер фиксированных фрагментов и их перекрытие?", options)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Matches) != 1 || !strings.Contains(result.Matches[0].Chunk.Text, "1200") || !strings.Contains(result.Matches[0].Chunk.Text, "150") {
		t.Fatalf("strong lexical match was not retained at result limit 1: %#v", result.Matches)
	}
	options.ResultLimit = 3
	result, err = s.SearchWithOptions(context.Background(), "Какой размер фиксированных фрагментов и их перекрытие?", options)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Matches) == 0 {
		t.Fatal("exact fact chunk was not retrieved")
	}
	var fact *Match
	for index := range result.Matches {
		match := &result.Matches[index]
		if strings.Contains(match.Chunk.Text, "1200") && strings.Contains(match.Chunk.Text, "150") {
			fact = match
			break
		}
	}
	if fact == nil {
		t.Fatalf("top results omitted exact numeric evidence: %#v", result.Matches)
	}
	if fact.CandidateSource != "lexical" || fact.LexicalTermMatches < lexicalGateMinTerms || fact.Score >= 0.9 {
		t.Fatalf("candidate channel/score metadata is misleading: %#v", fact)
	}

	negativeCases := []struct {
		query          string
		wantFixedChunk bool
	}{
		{"Какой размер колеса?", false},
		{"overlap 150", false},
		{"размер окна истории", false},
	}
	for _, test := range negativeCases {
		negative, err := s.SearchWithOptions(context.Background(), test.query, SearchOptions{CandidateLimit: 8, ResultLimit: 3, MinSimilarity: 0.9})
		if err != nil {
			t.Fatalf("query %q: %v", test.query, err)
		}
		for _, candidate := range negative.Candidates {
			if strings.Contains(candidate.Chunk.Text, "1200") && strings.Contains(candidate.Chunk.Text, "150") && candidate.CandidateSource == "lexical" {
				if !test.wantFixedChunk {
					t.Errorf("query %q incorrectly opened lexical gate for fixed chunk: %#v", test.query, candidate)
				}
			}
		}
		if (test.query == "Какой размер колеса?" || test.query == "overlap 150") && negative.LexicalGateMatchCount != 0 {
			t.Errorf("query %q passed lexical gate on question words/one term plus number: %#v", test.query, negative)
		}
	}
}

func containsString(items []string, value string) bool {
	for _, item := range items {
		if item == value {
			return true
		}
	}
	return false
}
