// Package prreview reviews immutable Git revisions using the application's RAG.
// It never checks out, builds, tests, or executes code from the reviewed tree.
package prreview

import (
	"context"
	"time"

	"ai-challenge-app/internal/ragindex"
)

type Options struct {
	Root, Base, Head, Title, Description, Repository string
	Model, EmbeddingModel                            string
	MaxFiles                                         int
	Embedder                                         ragindex.Embedder
	Complete                                         func(context.Context, string, string) (string, error)
	Logf                                             func(string, ...any)
}

type Evidence struct {
	SourceID string `json:"source_id"`
	Quote    string `json:"quote"`
}

type Finding struct {
	Category       string     `json:"category"`
	Severity       string     `json:"severity"`
	Path           string     `json:"path"`
	Line           int        `json:"line"`
	Side           string     `json:"side"`
	Title          string     `json:"title"`
	Scenario       string     `json:"scenario"`
	Impact         string     `json:"impact"`
	Recommendation string     `json:"recommendation"`
	Evidence       []Evidence `json:"evidence"`
}

// Source stores provenance in the report. Raw retrieved text is sent only to
// the model, not copied into the JSON report or GitHub review.
type Source struct {
	ID        string  `json:"id"`
	Kind      string  `json:"kind"`
	Path      string  `json:"path"`
	SHA       string  `json:"sha"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	ChunkID   string  `json:"chunk_id,omitempty"`
	Query     string  `json:"query,omitempty"`
	Score     float64 `json:"score,omitempty"`
	Channel   string  `json:"channel,omitempty"`
	Text      string  `json:"-"`
}

type FileReview struct {
	Path         string `json:"path"`
	PreviousPath string `json:"previous_path,omitempty"`
	Status       string `json:"status"`
	Reviewed     bool   `json:"reviewed"`
	SkipReason   string `json:"skip_reason,omitempty"`
}

type Report struct {
	GeneratedAt  time.Time    `json:"generated_at"`
	Repository   string       `json:"repository,omitempty"`
	BaseSHA      string       `json:"base_sha"`
	HeadSHA      string       `json:"head_sha"`
	MergeBaseSHA string       `json:"merge_base_sha"`
	Model        string       `json:"model"`
	CorpusFiles  int          `json:"corpus_files"`
	Files        []FileReview `json:"files"`
	Findings     []Finding    `json:"findings"`
	Sources      []Source     `json:"sources"`
	Warnings     []string     `json:"warnings"`
}
