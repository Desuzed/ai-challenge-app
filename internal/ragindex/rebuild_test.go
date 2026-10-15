package ragindex

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestFailedRebuildKeepsLastCompleteIndex(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "README.md")
	if err := os.WriteFile(path, []byte("# Project\nWorking documentation."), 0600); err != nil {
		t.Fatal(err)
	}
	b := &Rebuilder{Root: root, Output: "index", Model: "test", Sources: []string{"README.md"}, Embedder: stubEmbedder{}}
	if _, err := b.Rebuild(context.Background()); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(filepath.Join(root, "index", "index-structure.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("# Project\nChanged documentation."), 0600); err != nil {
		t.Fatal(err)
	}
	b.Embedder = rejectingEmbedder{}
	if _, err := b.Rebuild(context.Background()); err == nil {
		t.Fatal("embedding failure was ignored")
	}
	after, err := os.ReadFile(filepath.Join(root, "index", "index-structure.json"))
	if err != nil || string(after) != string(before) {
		t.Fatal("failed build changed the previous complete index")
	}
}
