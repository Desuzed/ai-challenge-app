package prreview

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	maxFileBytes   = 160 * 1024
	maxCorpusBytes = 3 * 1024 * 1024
	maxCorpusFiles = 300
	maxPatchBytes  = 32 * 1024
)

var shaPattern = regexp.MustCompile(`^(?:[a-f0-9]{40}|[a-f0-9]{64})$`)
var errOutputLimit = errors.New("Git output exceeds configured size limit")

type limitedBuffer struct {
	buffer bytes.Buffer
	limit  int
	hit    bool
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	if b.buffer.Len()+len(p) > b.limit {
		b.hit = true
		return 0, errOutputLimit
	}
	return b.buffer.Write(p)
}

func git(ctx context.Context, root string, limit int, args ...string) ([]byte, error) {
	argv := append([]string{"--literal-pathspecs", "-C", root}, args...)
	cmd := exec.CommandContext(ctx, "git", argv...)
	buf := &limitedBuffer{limit: limit}
	cmd.Stdout = buf
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if buf.hit || errors.Is(err, errOutputLimit) {
			return nil, errOutputLimit
		}
		return nil, fmt.Errorf("git %s: %w (revision must be available locally)", args[0], err)
	}
	return buf.buffer.Bytes(), nil
}

func resolve(ctx context.Context, root, ref string) (string, error) {
	if strings.TrimSpace(ref) == "" || strings.ContainsAny(ref, "\x00\r\n") {
		return "", errors.New("empty or invalid Git revision")
	}
	data, err := git(ctx, root, 256, "rev-parse", "--verify", "--end-of-options", ref+"^{commit}")
	sha := strings.TrimSpace(string(data))
	if err != nil {
		return "", err
	}
	if !shaPattern.MatchString(sha) {
		return "", errors.New("invalid resolved commit SHA")
	}
	return sha, nil
}

// allowedPath is deliberately independent of the working tree and .gitignore:
// only allowlisted tracked text is read; secrets, symlinks and submodules stay out.
func allowedPath(name string) bool {
	if name == "" || path.IsAbs(name) || path.Clean(name) != name || strings.ContainsAny(name, "\\\x00\r\n\t") || strings.HasPrefix(name, "../") {
		return false
	}
	for i, part := range strings.Split(name, "/") {
		switch strings.ToLower(part) {
		case "vendor", "node_modules", "dist", "build", "reports", "fixtures", "testdata", "secrets":
			return false
		}
		if strings.HasPrefix(part, ".") && !(i == 0 && part == ".github") {
			return false
		}
	}
	if name == "go.mod" {
		return true
	}
	base := strings.ToLower(path.Base(name))
	for _, suffix := range []string{".pb.go", ".gen.go", ".min.js", ".min.css"} {
		if strings.HasSuffix(base, suffix) {
			return false
		}
	}
	if strings.HasPrefix(base, "zz_generated") {
		return false
	}
	switch strings.ToLower(path.Ext(name)) {
	case ".go", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".md":
		return true
	case ".txt", ".json", ".yaml", ".yml":
		return strings.HasPrefix(name, "docs/") || strings.HasPrefix(name, ".github/workflows/")
	}
	return false
}

func documentation(name string) bool {
	return strings.HasSuffix(strings.ToLower(name), ".md") || strings.HasPrefix(name, "docs/")
}

func changedFiles(ctx context.Context, root, base, head string) ([]FileReview, error) {
	data, err := git(ctx, root, 2*1024*1024, "diff", "--no-ext-diff", "--no-textconv", "--name-status", "-z", "--find-renames", base, head, "--")
	if err != nil {
		return nil, err
	}
	parts := strings.Split(strings.TrimSuffix(string(data), "\x00"), "\x00")
	var files []FileReview
	for i := 0; i < len(parts) && parts[i] != ""; {
		status := parts[i]
		i++
		if i >= len(parts) {
			return nil, errors.New("malformed Git name-status output")
		}
		f := FileReview{Status: status, Path: parts[i]}
		i++
		if strings.HasPrefix(status, "R") || strings.HasPrefix(status, "C") {
			if i >= len(parts) {
				return nil, errors.New("malformed Git rename output")
			}
			f.PreviousPath, f.Path = f.Path, parts[i]
			i++
		}
		files = append(files, f)
	}
	return files, nil
}

// snapshot materializes regular blobs into a fresh private directory. Nothing
// from the PR is checked out or executed, including hooks and filter drivers.
func snapshot(ctx context.Context, root, head, dest string) (map[string]string, []string, error) {
	data, err := git(ctx, root, 8*1024*1024, "ls-tree", "-r", "-l", "-z", "--full-tree", head)
	if err != nil {
		return nil, nil, err
	}
	texts := map[string]string{}
	var warnings []string
	total := 0
	for _, record := range strings.Split(string(data), "\x00") {
		if record == "" {
			continue
		}
		header, name, ok := strings.Cut(record, "\t")
		fields := strings.Fields(header)
		if !ok || len(fields) != 4 {
			return nil, nil, errors.New("malformed Git tree")
		}
		if !allowedPath(name) || fields[1] != "blob" || (fields[0] != "100644" && fields[0] != "100755") {
			continue
		}
		size, err := strconv.Atoi(fields[3])
		if err != nil || size < 0 {
			return nil, nil, errors.New("invalid Git blob size")
		}
		if size > maxFileBytes || total+size > maxCorpusBytes || len(texts) >= maxCorpusFiles {
			warnings = append(warnings, "Не включён в RAG из-за лимита размера корпуса: "+name)
			continue
		}
		blob, err := git(ctx, root, maxFileBytes, "cat-file", "blob", fields[2])
		if err != nil {
			return nil, nil, err
		}
		if !utf8.Valid(blob) || bytes.ContainsRune(blob, '\x00') || len(bytes.TrimSpace(blob)) == 0 {
			continue
		}
		headerText := string(blob[:min(2048, len(blob))])
		if strings.Contains(headerText, "Code generated") && strings.Contains(headerText, "DO NOT EDIT") {
			continue
		}
		file := filepath.Join(dest, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(file), 0700); err != nil {
			return nil, nil, err
		}
		if err := os.WriteFile(file, blob, 0600); err != nil {
			return nil, nil, err
		}
		texts[name] = string(blob)
		total += len(blob)
	}
	return texts, warnings, nil
}

var hunkHeader = regexp.MustCompile(`^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@`)

// diffLines contains valid anchors on both sides of a unified diff. Looking at
// +/- prefixes before checking hunk headers would misread ---/+++ file headers.
func diffLines(patch string) (map[int]bool, map[int]bool) {
	left, right := map[int]bool{}, map[int]bool{}
	oldLine, newLine, inHunk := 0, 0, false
	for _, line := range strings.Split(patch, "\n") {
		if m := hunkHeader.FindStringSubmatch(line); m != nil {
			oldLine, _ = strconv.Atoi(m[1])
			newLine, _ = strconv.Atoi(m[2])
			inHunk = true
			continue
		}
		if !inHunk || line == "" || line[0] == '\\' {
			continue
		}
		switch line[0] {
		case '-':
			left[oldLine] = oldLine > 0
			oldLine++
		case '+':
			right[newLine] = newLine > 0
			newLine++
		case ' ':
			left[oldLine] = oldLine > 0
			right[newLine] = newLine > 0
			oldLine++
			newLine++
		}
	}
	return left, right
}
