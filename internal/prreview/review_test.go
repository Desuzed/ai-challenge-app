package prreview

import (
	"context"
	"encoding/json"
	"errors"
	"html"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type testEmbedder struct{ fail bool }

func (e testEmbedder) Embed(_ context.Context, input []string) ([][]float64, error) {
	if e.fail {
		return nil, errors.New("embedding service offline")
	}
	result := make([][]float64, len(input))
	for i := range input {
		result[i] = []float64{1, 1}
	}
	return result, nil
}

func testGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", root, "-c", "user.name=Review Test", "-c", "user.email=review@example.test", "-c", "commit.gpgsign=false"}, args...)...)
	data, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, data)
	}
	return strings.TrimSpace(string(data))
}

func writeTest(t *testing.T, root, name, text string) {
	t.Helper()
	file := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(file), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
}

func fixture(t *testing.T) (root, base, head string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("Git is required")
	}
	root = t.TempDir()
	testGit(t, root, "init", "--quiet", "--initial-branch=main")
	writeTest(t, root, "README.md", "# First\nFirst must return an empty string for an empty slice.\n")
	writeTest(t, root, "docs/contract.md", "# First contract\nFirst must handle empty values without panic.\n")
	writeTest(t, root, "first.go", "package sample\n\nfunc First(values []string) string {\n if len(values) == 0 { return \"\" }\n return values[0]\n}\n")
	writeTest(t, root, "caller_test.go", "package sample\n\nfunc EmptyCaller() string { return First(nil) }\n")
	writeTest(t, root, ".env", "private-secret-never-retrieve")
	if err := os.Symlink(".env", filepath.Join(root, "linked.go")); err != nil {
		t.Fatal(err)
	}
	testGit(t, root, "add", ".")
	testGit(t, root, "commit", "--quiet", "-m", "base")
	base = testGit(t, root, "rev-parse", "HEAD")
	writeTest(t, root, "first.go", "package sample\n\nfunc First(values []string) string {\n return values[0]\n}\n")
	testGit(t, root, "add", "first.go")
	testGit(t, root, "commit", "--quiet", "-m", "remove empty slice guard")
	head = testGit(t, root, "rev-parse", "HEAD")
	return
}

func TestReviewUsesPinnedCodeAndBothRAGPools(t *testing.T) {
	root, base, head := fixture(t)
	// A dirty working tree must not change the reviewed corpus or be modified.
	writeTest(t, root, "first.go", "uncommitted-work-must-not-be-reviewed")
	status := testGit(t, root, "status", "--porcelain")
	calls := 0
	r, err := Run(context.Background(), Options{Root: root, Base: base, Head: head, Title: "First empty values contract", EmbeddingModel: "test", Embedder: testEmbedder{},
		Complete: func(_ context.Context, system, prompt string) (string, error) {
			calls++
			if !strings.Contains(system, "UNTRUSTED DATA") || !strings.Contains(prompt, "@@") || strings.Contains(prompt, "uncommitted-work") || strings.Contains(prompt, "private-secret") {
				t.Fatal("missing diff/trust boundary or leaked working-tree data")
			}
			var input struct {
				Sources []struct {
					Source
					Text string `json:"text"`
				} `json:"sources_with_line_ranges_may_be_excerpts"`
			}
			if err := json.Unmarshal([]byte(prompt), &input); err != nil {
				t.Fatal(err)
			}
			var evidence []Evidence
			docs, caller := false, false
			for _, s := range input.Sources {
				if s.Kind == "documentation" {
					docs = true
				}
				if s.Kind == "code" && s.Path == "caller_test.go" {
					caller = true
					evidence = append(evidence, Evidence{s.ID, "return First(nil)"})
				}
			}
			if !docs || !caller {
				t.Fatal("retrieval did not include docs and cross-file code")
			}
			f := Finding{Category: "bug", Severity: "high", Path: "first.go", Line: 4, Side: "LEFT", Title: "Panic на пустом срезе", Scenario: "EmptyCaller вызывает First(nil)", Impact: "Обращение values[0] вызывает panic", Recommendation: "Вернуть проверку пустого среза", Evidence: evidence}
			data, _ := json.Marshal(map[string]any{"findings": []Finding{f}})
			return string(data), nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(r.Findings) != 1 || !r.Files[0].Reviewed || r.HeadSHA != head || r.MergeBaseSHA != base || r.CorpusFiles != 4 {
		t.Fatalf("unexpected report: %+v", r)
	}
	if testGit(t, root, "status", "--porcelain") != status {
		t.Fatal("review modified working tree")
	}
	encoded, _ := json.Marshal(r)
	if strings.Contains(string(encoded), "func EmptyCaller") {
		t.Fatal("report copied raw chunk text")
	}
	for _, title := range []string{"Потенциальные баги", "Архитектурные проблемы", "Рекомендации", "caller"} {
		if !strings.Contains(Markdown(r), title) {
			t.Fatalf("missing report section/evidence %q", title)
		}
	}
}

func TestEmbeddingAndModelFailuresDoNotProduceCleanReviews(t *testing.T) {
	root, base, head := fixture(t)
	for _, failure := range []string{"embedding", "model", "malformed"} {
		t.Run(failure, func(t *testing.T) {
			_, err := Run(context.Background(), Options{Root: root, Base: base, Head: head, EmbeddingModel: "test", Embedder: testEmbedder{fail: failure == "embedding"},
				Complete: func(context.Context, string, string) (string, error) {
					if failure == "model" {
						return "", errors.New("provider unavailable")
					}
					return `{}`, nil
				}})
			if err == nil {
				t.Fatal("failure reported as successful review")
			}
		})
	}
}

func TestMergeBaseExcludesUnrelatedBaseChanges(t *testing.T) {
	root, base, head := fixture(t)
	testGit(t, root, "checkout", "--quiet", "--detach", base)
	writeTest(t, root, "unrelated.go", "package sample\nconst Unrelated = 1\n")
	testGit(t, root, "add", ".")
	testGit(t, root, "commit", "--quiet", "-m", "unrelated base change")
	newBase := testGit(t, root, "rev-parse", "HEAD")
	r, err := Run(context.Background(), Options{Root: root, Base: newBase, Head: head, EmbeddingModel: "test", Embedder: testEmbedder{},
		Complete: func(context.Context, string, string) (string, error) { return `{"findings":[]}`, nil }})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Files) != 1 || r.Files[0].Path != "first.go" || r.MergeBaseSHA != base || len(r.Findings) != 0 {
		t.Fatalf("review used base..head instead of merge-base: %+v", r)
	}
}

func TestRenameAndRemovedFileDiscovery(t *testing.T) {
	root, _, base := fixture(t)
	testGit(t, root, "mv", "first.go", "renamed.go")
	testGit(t, root, "rm", "caller_test.go")
	testGit(t, root, "commit", "--quiet", "-m", "rename and remove")
	head := testGit(t, root, "rev-parse", "HEAD")
	files, err := changedFiles(context.Background(), root, base, head)
	if err != nil || len(files) != 2 {
		t.Fatalf("files: %+v, %v", files, err)
	}
	for _, f := range files {
		if f.Path == "renamed.go" && f.PreviousPath != "first.go" {
			t.Fatal("lost previous filename")
		}
	}
}

func TestFindingValidationRejectsInventedEvidenceAndAnchors(t *testing.T) {
	f := Finding{Category: "bug", Severity: "high", Path: "file.go", Side: "RIGHT", Line: 3, Title: "Valid title", Scenario: "Valid scenario", Impact: "Valid impact", Recommendation: "Valid recommendation", Evidence: []Evidence{{"s1", "return input[0]"}}}
	sources := []Source{{ID: "s1", Text: "return input[0]"}}
	for _, mutate := range []func(*Finding){
		func(f *Finding) { f.Path = "other.go" },
		func(f *Finding) { f.Line = 999 },
		func(f *Finding) { f.Side = "LEFT" },
		func(f *Finding) { f.Evidence = []Evidence{{"s1", "invented quote"}} },
		func(f *Finding) { f.Evidence = []Evidence{{"s2", "return input[0]"}} },
		func(f *Finding) { f.Category = "security-ish" },
	} {
		copy := f
		mutate(&copy)
		if validateFinding(copy, FileReview{Path: "file.go"}, nil, map[int]bool{3: true}, sources) == nil {
			t.Fatalf("accepted invalid finding %+v", copy)
		}
	}
	if err := validateFinding(f, FileReview{Path: "file.go"}, nil, map[int]bool{3: true}, sources); err != nil {
		t.Fatal(err)
	}
}

func TestDiffLinesAndSizeLimit(t *testing.T) {
	left, right := diffLines("--- a/file.go\n+++ b/file.go\n@@ -4,2 +4,2 @@\n-old\n+new\n context\n")
	if len(left) != 2 || len(right) != 2 || !left[4] || !right[4] || !left[5] || !right[5] || left[6] || right[6] {
		t.Fatalf("invalid anchors left=%v right=%v", left, right)
	}
	root, _, head := fixture(t)
	if _, err := git(context.Background(), root, 10, "show", head+":first.go"); !errors.Is(err, errOutputLimit) {
		t.Fatalf("missing output limit: %v", err)
	}
}

func TestBodyEscapesMentionsAndImages(t *testing.T) {
	text := plain("@everyone ![image](https://evil.test) <img src=x>")
	if strings.Contains(text, "@everyone") || strings.Contains(text, "![image]") || strings.Contains(text, "<img") {
		t.Fatalf("unsafe rendered text: %s", text)
	}
	if got := html.UnescapeString(plain(`return ""`)); got != `return ""` {
		t.Fatalf("quotes did not survive Markdown entity escaping: %s", got)
	}
	for _, input := range []string{`{}`, `{"findings":null}`, `{"findings":[]} {}`, "```json\n{}\n```"} {
		if _, err := decodeFindings(input); err == nil {
			t.Fatalf("accepted invalid response %q", input)
		}
	}
}
