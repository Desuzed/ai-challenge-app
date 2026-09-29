package ragindex

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

type stubEmbedder struct{}

func (stubEmbedder) Embed(_ context.Context, texts []string) ([][]float64, error) {
	vectors := make([][]float64, len(texts))
	for i, text := range texts {
		vectors[i] = []float64{float64(utf8.RuneCountInString(text)), 1}
	}
	return vectors, nil
}

type rejectingEmbedder struct{}

func (rejectingEmbedder) Embed(_ context.Context, _ []string) ([][]float64, error) {
	return nil, os.ErrInvalid
}

func TestBuildAndSaveBothStrategies(t *testing.T) {
	root := t.TempDir()
	markdown := "# Overview\n" + strings.Repeat("A useful explanation. ", 90) + "\n## Storage\n" + strings.Repeat("Stored in JSON. ", 40)
	goText := "package demo\n\nfunc First() {\n" + strings.Repeat("// first function detail\n", 12) + "}\n\nfunc Second() {\n" + strings.Repeat("// second function detail\n", 12) + "}\n"
	if err := os.WriteFile(filepath.Join(root, "guide.md"), []byte(markdown), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "demo.go"), []byte(goText), 0600); err != nil {
		t.Fatal(err)
	}
	docs, err := Load(root, []string{"guide.md", "demo.go"})
	if err != nil {
		t.Fatal(err)
	}
	indexes, report, err := Build(context.Background(), docs, "test-model", stubEmbedder{})
	if err != nil {
		t.Fatal(err)
	}
	if len(indexes) != 2 || indexes[0].Strategy != "fixed" || indexes[1].Strategy != "structure" {
		t.Fatalf("wrong indexes: %#v", indexes)
	}
	if len(indexes[1].Chunks) <= len(indexes[0].Chunks) {
		t.Fatalf("structure did not split headings and declarations: %d <= %d", len(indexes[1].Chunks), len(indexes[0].Chunks))
	}
	sections := map[string]bool{}
	for _, index := range indexes {
		for _, chunk := range index.Chunks {
			if chunk.Source == "" || chunk.Title == "" || chunk.Section == "" || chunk.ChunkID == "" || len(chunk.Vector) != 2 {
				t.Fatalf("incomplete chunk: %#v", chunk)
			}
			if utf8.RuneCountInString(chunk.Text) > MaxChunkRunes {
				t.Fatalf("oversize chunk: %s", chunk.ChunkID)
			}
			sections[chunk.Section] = true
		}
	}
	if !sections["Overview"] || !sections["Storage"] || !sections["demo.go / First"] || !sections["Second"] {
		t.Fatalf("missing structure labels: %#v", sections)
	}
	output := filepath.Join(root, ".local", "rag")
	if err := Save(output, indexes, report); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"index-fixed.json", "index-structure.json", "report.json"} {
		data, err := os.ReadFile(filepath.Join(output, name))
		if err != nil {
			t.Fatal(err)
		}
		if !json.Valid(data) {
			t.Fatalf("invalid JSON in %s", name)
		}
	}
	cache, err := LoadEmbeddingCache(output, "test-model")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := BuildCached(context.Background(), docs, "test-model", rejectingEmbedder{}, cache); err != nil {
		t.Fatalf("unchanged chunks should reuse stored embeddings: %v", err)
	}
	otherModelCache, err := LoadEmbeddingCache(output, "different-model")
	if err != nil {
		t.Fatal(err)
	}
	if len(otherModelCache) != 0 {
		t.Fatal("reused embeddings across different models")
	}
}

func TestLoadRejectsOutsideProject(t *testing.T) {
	if _, err := Load(t.TempDir(), []string{"../secret.txt"}); err == nil {
		t.Fatal("outside source was accepted")
	}
}

func TestChangedDocumentChangesChunkIDs(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "guide.md")
	write := func(s string) []Chunk {
		if err := os.WriteFile(path, []byte(s), 0600); err != nil {
			t.Fatal(err)
		}
		docs, err := Load(root, []string{"guide.md"})
		if err != nil {
			t.Fatal(err)
		}
		chunks, err := ChunkDocuments(docs, "fixed")
		if err != nil {
			t.Fatal(err)
		}
		return chunks
	}
	before := write("# Guide\nFirst version")
	after := write("# Guide\nSecond version")
	if before[0].ChunkID == after[0].ChunkID {
		t.Fatal("changed source retained stale chunk ID")
	}
}
