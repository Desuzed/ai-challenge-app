package projectmcp

import (
	"context"
	"encoding/json"
	"os/exec"
	"testing"
)

func TestMCPReadsLiveBranchAndDetachedHEAD(t *testing.T) {
	root := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", append([]string{"-C", root}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	git("init", "--initial-branch=main")
	git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "initial")
	b, err := New(context.Background(), root)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	tools, err := b.Tools(context.Background())
	if err != nil || len(tools) != 1 || tools[0].Name != "project_git_branch" || !tools[0].ReadOnly {
		t.Fatalf("tools=%#v err=%v", tools, err)
	}
	read := func() GitContext {
		t.Helper()
		text, isError, err := b.CallForModel(context.Background(), "project_git_branch", map[string]any{})
		if err != nil || isError {
			t.Fatalf("call=%q error=%v err=%v", text, isError, err)
		}
		var result GitContext
		if err := json.Unmarshal([]byte(text), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	if got := read(); got.Branch != "main" || got.Detached {
		t.Fatalf("initial=%#v", got)
	}
	git("switch", "-c", "feature/live")
	if got := read(); got.Branch != "feature/live" || got.Detached {
		t.Fatalf("changed=%#v", got)
	}
	git("checkout", "--detach")
	if got := read(); got.Branch != "" || !got.Detached || len(got.Commit) != 40 {
		t.Fatalf("detached=%#v", got)
	}
}

func TestMCPDoesNotInventBranchOutsideGit(t *testing.T) {
	b, err := New(context.Background(), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	_, isError, err := b.CallForModel(context.Background(), "project_git_branch", map[string]any{})
	if err != nil || !isError {
		t.Fatalf("isError=%v err=%v", isError, err)
	}
}
