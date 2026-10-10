// Package ragindex builds local document indexes with embedding metadata.
package ragindex

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"ai-challenge-app/internal/ollama"
)

const MaxChunkRunes = 1200
const OverlapRunes = 150

// DefaultFiles form a stable, reviewable corpus. Tests, secrets and generated
// files are intentionally excluded.
var DefaultFiles = []string{
	"README.md", "main.go", "internal/agent/agent.go",
	"internal/agent/store.go", "internal/deepseek/client.go",
	"internal/agent/task_memory.go", "internal/handlers/handlers.go", "internal/models/models.go",
	"docs/rag-task-chat-corpus.md",
}

type Document struct {
	Source string `json:"source"`
	Title  string `json:"title"`
	Text   string `json:"-"`
	Hash   string `json:"sha256"`
}

type Chunk struct {
	Source  string    `json:"source"`
	Title   string    `json:"title"`
	Section string    `json:"section"`
	ChunkID string    `json:"chunk_id"`
	Text    string    `json:"text"`
	Vector  []float64 `json:"embedding,omitempty"`
}

type Index struct {
	Strategy string     `json:"strategy"`
	Model    string     `json:"model"`
	Files    []Document `json:"files"`
	Chunks   []Chunk    `json:"chunks"`
}

type Summary struct {
	Strategy  string  `json:"strategy"`
	Files     int     `json:"files"`
	Chunks    int     `json:"chunks"`
	MinRunes  int     `json:"min_runes"`
	MeanRunes float64 `json:"mean_runes"`
	MaxRunes  int     `json:"max_runes"`
	Sample    []Chunk `json:"sample_chunks"`
}

type Report struct {
	CorpusFiles []string  `json:"corpus_files"`
	TotalRunes  int       `json:"total_runes"`
	Model       string    `json:"embedding_model"`
	Strategies  []Summary `json:"strategies"`
}

func Load(root string, paths []string) ([]Document, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	docs := make([]Document, 0, len(paths))
	seen := map[string]bool{}
	for _, path := range paths {
		clean := filepath.Clean(path)
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return nil, fmt.Errorf("source must be inside project: %q", path)
		}
		if seen[clean] {
			return nil, fmt.Errorf("duplicate source: %s", clean)
		}
		seen[clean] = true
		data, err := os.ReadFile(filepath.Join(root, clean))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", clean, err)
		}
		if !utf8.Valid(data) {
			return nil, fmt.Errorf("%s is not UTF-8 text", clean)
		}
		if len(bytes.TrimSpace(data)) == 0 {
			return nil, fmt.Errorf("%s is empty", clean)
		}
		hash := sha256.Sum256(data)
		docs = append(docs, Document{Source: filepath.ToSlash(clean), Title: filepath.Base(clean), Text: string(data), Hash: hex.EncodeToString(hash[:])})
	}
	return docs, nil
}

type section struct{ name, text string }

var markdownHeading = regexp.MustCompile(`^#{1,6}\s+(.+?)\s*$`)
var goDeclaration = regexp.MustCompile(`^(?:func|type|var|const)\s+(?:\([^)]*\)\s*)?([\pL_][\pL\pN_]*)`)

func sections(doc Document) []section {
	lines := strings.SplitAfter(doc.Text, "\n")
	result := []section{}
	name := doc.Title
	var body strings.Builder
	flush := func() {
		if strings.TrimSpace(body.String()) != "" {
			result = append(result, section{name, body.String()})
		}
		body.Reset()
	}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		var next string
		if strings.HasSuffix(doc.Source, ".md") {
			if match := markdownHeading.FindStringSubmatch(trimmed); match != nil {
				next = match[1]
			}
		} else if strings.HasSuffix(doc.Source, ".go") {
			if match := goDeclaration.FindStringSubmatch(strings.TrimRight(line, "\r\n")); match != nil {
				next = match[1]
			}
		}
		if next != "" {
			flush()
			name = next
		}
		body.WriteString(line)
	}
	flush()
	return mergeSmallSections(result)
}

func mergeSmallSections(input []section) []section {
	var result []section
	for _, current := range input {
		if len(result) > 0 && utf8.RuneCountInString(strings.TrimSpace(result[len(result)-1].text)) < 160 {
			previous := &result[len(result)-1]
			previous.name += " / " + current.name
			previous.text += current.text
			continue
		}
		result = append(result, current)
	}
	if len(result) > 1 && utf8.RuneCountInString(strings.TrimSpace(result[len(result)-1].text)) < 160 {
		last := result[len(result)-1]
		result = result[:len(result)-1]
		previous := &result[len(result)-1]
		previous.name += " / " + last.name
		previous.text += last.text
	}
	return result
}

func windows(text string, size, overlap int) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	var out []string
	for start := 0; start < len(runes); {
		end := start + size
		if end >= len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[start:end]))
		if end == len(runes) {
			break
		}
		start = end - overlap
	}
	return out
}

func ChunkDocuments(docs []Document, strategy string) ([]Chunk, error) {
	if strategy != "fixed" && strategy != "structure" {
		return nil, fmt.Errorf("unknown strategy %q", strategy)
	}
	var out []Chunk
	for _, doc := range docs {
		units := sections(doc)
		if strategy == "fixed" {
			units = []section{{doc.Title, doc.Text}}
		}
		ordinal := 0
		for _, unit := range units {
			for _, part := range windows(unit.text, MaxChunkRunes, OverlapRunes) {
				if strings.TrimSpace(part) == "" {
					continue
				}
				ordinal++
				idInput := fmt.Sprintf("%s\x00%s\x00%s\x00%d", strategy, doc.Source, doc.Hash, ordinal)
				hash := sha256.Sum256([]byte(idInput))
				out = append(out, Chunk{Source: doc.Source, Title: doc.Title, Section: unit.name,
					ChunkID: hex.EncodeToString(hash[:12]), Text: part})
			}
		}
	}
	return out, nil
}

// Embedder is shared by both strategies, so each unique chunk text is embedded
// only once during a build.
type Embedder interface {
	Embed(context.Context, []string) ([][]float64, error)
}

type Ollama struct {
	URL, Model string
	Client     *http.Client
	// Chat retrieval can release the embedding model before generation on
	// machines where keeping both runners resident causes memory pressure.
	KeepAlive string
}

func (o Ollama) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if !ollama.IsLocalURL(o.URL) {
		return nil, errors.New("OLLAMA_URL для эмбеддингов должен указывать на loopback-адрес")
	}
	url := strings.TrimRight(o.URL, "/") + "/api/embed"
	payload := map[string]any{"model": o.Model, "input": texts}
	if o.KeepAlive != "" {
		payload["keep_alive"] = o.KeepAlive
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := o.Client
	if client == nil {
		client = &http.Client{Timeout: 2 * time.Minute, Transport: &http.Transport{Proxy: nil}, CheckRedirect: func(req *http.Request, _ []*http.Request) error {
			if !ollama.IsLocalURL(req.URL.String()) {
				return errors.New("редирект эмбеддингов на удалённый адрес запрещён")
			}
			return nil
		}}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Ollama unavailable at %s: %w", o.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("Ollama HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(message)))
	}
	var result struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("Ollama returned %d embeddings for %d texts", len(result.Embeddings), len(texts))
	}
	for _, vector := range result.Embeddings {
		if len(vector) == 0 {
			return nil, errors.New("Ollama returned empty embedding")
		}
	}
	return result.Embeddings, nil
}

func Build(ctx context.Context, docs []Document, model string, embedder Embedder) ([]Index, Report, error) {
	return BuildCached(ctx, docs, model, embedder, nil)
}

// BuildCached reuses vectors for identical text embedded with the same model.
// Callers may load the cache from the previous index before rebuilding.
func BuildCached(ctx context.Context, docs []Document, model string, embedder Embedder, previous map[string][]float64) ([]Index, Report, error) {
	var indexes []Index
	var report Report
	report.Model = model
	for _, doc := range docs {
		report.CorpusFiles = append(report.CorpusFiles, doc.Source)
		report.TotalRunes += utf8.RuneCountInString(doc.Text)
	}
	cache := map[string][]float64{}
	for text, vector := range previous {
		if len(vector) > 0 {
			cache[text] = vector
		}
	}
	for _, strategy := range []string{"fixed", "structure"} {
		chunks, err := ChunkDocuments(docs, strategy)
		if err != nil {
			return nil, Report{}, err
		}
		for start := 0; start < len(chunks); start += 16 {
			end := start + 16
			if end > len(chunks) {
				end = len(chunks)
			}
			var pending []string
			for i := start; i < end; i++ {
				if _, ok := cache[chunks[i].Text]; !ok {
					pending = append(pending, chunks[i].Text)
				}
			}
			if len(pending) > 0 {
				vectors, err := embedder.Embed(ctx, pending)
				if err != nil {
					return nil, Report{}, fmt.Errorf("embed %s chunks %d-%d: %w", strategy, start+1, end, err)
				}
				if len(vectors) != len(pending) {
					return nil, Report{}, errors.New("embedder returned wrong vector count")
				}
				for i, text := range pending {
					cache[text] = vectors[i]
				}
			}
			for i := start; i < end; i++ {
				chunks[i].Vector = cache[chunks[i].Text]
			}
		}
		indexes = append(indexes, Index{Strategy: strategy, Model: model, Files: append([]Document(nil), docs...), Chunks: chunks})
		report.Strategies = append(report.Strategies, summarize(strategy, len(docs), chunks))
	}
	return indexes, report, nil
}

// LoadEmbeddingCache reads a previous local index. A model change forces a full
// rebuild, while unchanged chunk texts keep their vectors across code edits.
func LoadEmbeddingCache(output, model string) (map[string][]float64, error) {
	cache := map[string][]float64{}
	for _, strategy := range []string{"fixed", "structure"} {
		data, err := os.ReadFile(filepath.Join(output, "index-"+strategy+".json"))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		var index Index
		if err := json.Unmarshal(data, &index); err != nil {
			return nil, fmt.Errorf("read previous %s index: %w", strategy, err)
		}
		if index.Model != model {
			continue
		}
		for _, chunk := range index.Chunks {
			if len(chunk.Vector) > 0 {
				cache[chunk.Text] = chunk.Vector
			}
		}
	}
	return cache, nil
}

func summarize(strategy string, files int, chunks []Chunk) Summary {
	s := Summary{Strategy: strategy, Files: files, Chunks: len(chunks)}
	if len(chunks) == 0 {
		return s
	}
	s.MinRunes = utf8.RuneCountInString(chunks[0].Text)
	var total int
	for _, chunk := range chunks {
		n := utf8.RuneCountInString(chunk.Text)
		if n < s.MinRunes {
			s.MinRunes = n
		}
		if n > s.MaxRunes {
			s.MaxRunes = n
		}
		total += n
	}
	s.MeanRunes = float64(total) / float64(len(chunks))
	for _, chunk := range chunks {
		if len(s.Sample) == 3 {
			break
		}
		if len(s.Sample) == 0 || s.Sample[len(s.Sample)-1].Source != chunk.Source {
			s.Sample = append(s.Sample, Chunk{Source: chunk.Source, Title: chunk.Title, Section: chunk.Section, ChunkID: chunk.ChunkID, Text: preview(chunk.Text)})
		}
	}
	return s
}

func preview(s string) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) > 180 {
		r = r[:180]
	}
	return strings.ReplaceAll(string(r), "\n", " ")
}

func Save(output string, indexes []Index, report Report) error {
	if err := os.MkdirAll(output, 0700); err != nil {
		return err
	}
	for _, index := range indexes {
		if err := saveJSON(filepath.Join(output, "index-"+index.Strategy+".json"), index); err != nil {
			return err
		}
	}
	return saveJSON(filepath.Join(output, "report.json"), report)
}

func saveJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".rag-index-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// SortedSourceNames is used by tests and report consumers to check the corpus.
func SortedSourceNames(docs []Document) []string {
	names := make([]string, 0, len(docs))
	for _, doc := range docs {
		names = append(names, doc.Source)
	}
	sort.Strings(names)
	return names
}
