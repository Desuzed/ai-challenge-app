package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSettingsPreferEnvironmentAndNeverReadDotenvInCI(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".env"), []byte("DEEPSEEK_API_KEY=local-value\nGITHUB_TOKEN='$(never-execute)'\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CI", "")
	t.Setenv("DEEPSEEK_API_KEY", "process-value")
	if got := setting(root, "DEEPSEEK_API_KEY"); got != "process-value" {
		t.Fatalf("got %q", got)
	}
	t.Setenv("DEEPSEEK_API_KEY", "")
	if got := setting(root, "DEEPSEEK_API_KEY"); got != "local-value" {
		t.Fatalf("got %q", got)
	}
	t.Setenv("GITHUB_TOKEN", "")
	if got := setting(root, "GITHUB_TOKEN"); got != "$(never-execute)" {
		t.Fatalf("got %q", got)
	}
	t.Setenv("CI", "true")
	if got := setting(root, "DEEPSEEK_API_KEY"); got != "" {
		t.Fatalf("CI read checkout .env: %q", got)
	}
}
