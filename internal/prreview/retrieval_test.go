package prreview

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"ai-challenge-app/internal/ragindex"
)

func TestPreselectionBoundsWorkAndRetainsCrossFileEvidence(t *testing.T) {
	var chunks []ragindex.Chunk
	for file := 0; file < 100; file++ {
		for chunk := 0; chunk < 20; chunk++ {
			chunks = append(chunks, ragindex.Chunk{Source: fmt.Sprintf("other/file%d.go", file), ChunkID: fmt.Sprintf("%03d-%03d", file, chunk), Text: "unrelated rendering implementation"})
		}
	}
	chunks = append(chunks, ragindex.Chunk{Source: "cmd/prreview/main_test.go", ChunkID: "target", Text: "TestSettingsPreferEnvironmentAndNeverReadDotenvInCI uses setting and DEEPSEEK_API_KEY"})
	queries := []corpusQuery{{"cmd/prreview/main.go", "setting environment process .env DEEPSEEK_API_KEY"}}
	selected := selectChunks(chunks, queries, "code")
	if len(selected) > 16 {
		t.Fatalf("single-file review embedded %d chunks out of %d", len(selected), len(chunks))
	}
	found := false
	counts := map[string]int{}
	for _, c := range selected {
		found = found || c.ChunkID == "target"
		counts[c.Source]++
		if counts[c.Source] > 2 {
			t.Fatal("one large file crowded out cross-file context")
		}
	}
	if !found {
		t.Fatal("lost the relevant caller/test while reducing embedding work")
	}
	if next := selectChunks(chunks, queries, "code"); !reflect.DeepEqual(selected, next) {
		t.Fatal("selection is not deterministic")
	}
	docs := make([]ragindex.Chunk, 100)
	for i := range docs {
		docs[i] = ragindex.Chunk{Source: fmt.Sprintf("docs/%d.md", i), ChunkID: fmt.Sprint(i), Text: "general project guide"}
	}
	docs[99].Text = "Nonempty process environment takes precedence over .env; setting must honor DEEPSEEK_API_KEY"
	selected = selectChunks(docs, queries, "documentation")
	found = false
	for _, c := range selected {
		found = found || c.ChunkID == "99"
	}
	if len(selected) > 8 || !found {
		t.Fatalf("documentation budget or contract retrieval failed: %+v", selected)
	}
}

func TestPreselectionGivesEveryChangedFileAChance(t *testing.T) {
	var chunks []ragindex.Chunk
	var queries []corpusQuery
	for i := 0; i < 20; i++ {
		symbol := fmt.Sprintf("uniqueSymbol%d", i)
		queries = append(queries, corpusQuery{fmt.Sprintf("file%d.go", i), symbol})
		chunks = append(chunks, ragindex.Chunk{Source: fmt.Sprintf("caller%d.go", i), ChunkID: symbol, Text: symbol})
	}
	for i := 0; i < 200; i++ {
		chunks = append(chunks, ragindex.Chunk{Source: fmt.Sprintf("noise%d.go", i), ChunkID: fmt.Sprintf("noise%d", i), Text: "unrelated"})
	}
	selected := selectChunks(chunks, queries, "code")
	seen := map[string]bool{}
	for _, c := range selected {
		seen[c.ChunkID] = true
	}
	if len(selected) > 48 {
		t.Fatalf("exceeded total embedding budget: %d", len(selected))
	}
	for i := 0; i < 20; i++ {
		if !seen[fmt.Sprintf("uniqueSymbol%d", i)] {
			t.Fatalf("changed file %d lost its relevant caller", i)
		}
	}
}

func TestEmbeddingSelectedChunksUsesBoundedBatches(t *testing.T) {
	chunks := make([]ragindex.Chunk, 17)
	opts := Options{Embedder: testEmbedder{}}
	if err := embedChunks(context.Background(), opts, "code", chunks); err != nil {
		t.Fatal(err)
	}
	for _, chunk := range chunks {
		if len(chunk.Vector) == 0 {
			t.Fatal("selected chunk missing vector")
		}
	}
}
