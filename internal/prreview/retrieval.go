package prreview

import (
	"context"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"ai-challenge-app/internal/ragindex"
)

type corpusQuery struct {
	path string
	text string
}

func corpusQueries(ctx context.Context, opts Options, r Report) ([]corpusQuery, error) {
	var queries []corpusQuery
	for _, f := range r.Files {
		if f.SkipReason != "" || len(queries) >= opts.MaxFiles {
			continue
		}
		paths := []string{f.Path}
		if f.PreviousPath != "" {
			paths = append(paths, f.PreviousPath)
		}
		args := append([]string{"diff", "--no-ext-diff", "--no-textconv", "--find-renames", "--unified=8", r.MergeBaseSHA, r.HeadSHA, "--"}, paths...)
		patch, err := git(ctx, opts.Root, maxPatchBytes, args...)
		if err != nil {
			// The review loop records size exclusions and other errors itself.
			continue
		}
		queries = append(queries, corpusQuery{f.Path, opts.Title + " " + f.Path + " " + string(patch)})
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return queries, nil
}

var camelBoundary = regexp.MustCompile(`([\p{Ll}\pN])([\p{Lu}])`)
var corpusWord = regexp.MustCompile(`[\pL\pN]+`)

func corpusTerms(text string) map[string]bool {
	terms := map[string]bool{}
	text = camelBoundary.ReplaceAllString(text, "$1 $2")
	for _, term := range corpusWord.FindAllString(strings.ToLower(text), -1) {
		if utf8.RuneCountInString(term) < 2 {
			continue
		}
		switch term {
		case "if", "for", "return", "func", "var", "const", "type", "nil", "string", "true", "false", "go", "the", "and", "not", "test", "verify", "review", "ai", "do", "merge", "from", "with":
			continue
		}
		if len(term) > 4 && strings.HasSuffix(term, "s") {
			term = strings.TrimSuffix(term, "s")
		}
		terms[term] = true
	}
	return terms
}

// selectChunks performs a cheap lexical first stage over the entire snapshot.
// Round-robin selection gives each changed file a quota, with per-source
// diversity so a large implementation cannot crowd out callers and tests.
// Vector retrieval and reranking still happen afterwards in rag.Searcher.
func selectChunks(chunks []ragindex.Chunk, queries []corpusQuery, kind string) []ragindex.Chunk {
	perQuery, totalLimit, perSource := 16, 48, 2
	if kind == "documentation" {
		perQuery, totalLimit, perSource = 8, 24, 3
	}
	if len(chunks) <= min(totalLimit, perQuery*max(1, len(queries))) {
		return chunks
	}
	if len(queries) == 0 {
		return nil
	}
	terms := make([]map[string]bool, len(chunks))
	df := map[string]int{}
	for i, chunk := range chunks {
		terms[i] = corpusTerms(chunk.Source + " " + chunk.Section + " " + chunk.Text)
		for term := range terms[i] {
			df[term]++
		}
	}
	type ranked struct {
		index int
		score float64
	}
	lists := make([][]int, len(queries))
	for q, query := range queries {
		wanted := corpusTerms(query.text)
		var ordered []string
		for term := range wanted {
			ordered = append(ordered, term)
		}
		sort.Strings(ordered)
		order := make([]ranked, len(chunks))
		for i, chunk := range chunks {
			score := 0.0
			for _, term := range ordered {
				if terms[i][term] {
					score += math.Log(1 + float64(len(chunks))/float64(1+df[term]))
				}
			}
			if kind == "code" && chunk.Source == query.path {
				score *= 0.5 // Changed code already has mandatory direct context.
			}
			order[i] = ranked{i, score}
		}
		sort.Slice(order, func(i, j int) bool {
			if order[i].score == order[j].score {
				return chunks[order[i].index].ChunkID < chunks[order[j].index].ChunkID
			}
			return order[i].score > order[j].score
		})
		counts := map[string]int{}
		for _, item := range order {
			name := chunks[item.index].Source
			if counts[name] >= perSource {
				continue
			}
			lists[q] = append(lists[q], item.index)
			counts[name]++
			if len(lists[q]) == perQuery {
				break
			}
		}
	}
	var result []ragindex.Chunk
	seen := map[int]bool{}
	for n := 0; n < perQuery && len(result) < totalLimit; n++ {
		for _, list := range lists {
			if n < len(list) && !seen[list[n]] {
				seen[list[n]] = true
				result = append(result, chunks[list[n]])
			}
			if len(result) == totalLimit {
				break
			}
		}
	}
	return result
}

func embedChunks(ctx context.Context, opts Options, kind string, chunks []ragindex.Chunk) error {
	dimension := 0
	for start := 0; start < len(chunks); start += 8 {
		end := min(start+8, len(chunks))
		texts := make([]string, 0, end-start)
		for _, chunk := range chunks[start:end] {
			texts = append(texts, chunk.Text)
		}
		logf(opts, "Embedding %s chunks %d-%d/%d", kind, start+1, end, len(chunks))
		vectors, err := opts.Embedder.Embed(ctx, texts)
		if err != nil {
			return fmt.Errorf("embed %s: %w", kind, err)
		}
		if len(vectors) != len(texts) {
			return fmt.Errorf("embed %s: wrong vector count", kind)
		}
		for i, vector := range vectors {
			if dimension == 0 {
				dimension = len(vector)
			}
			if len(vector) == 0 || len(vector) != dimension {
				return fmt.Errorf("embed %s: invalid vector dimension", kind)
			}
			for _, value := range vector {
				if math.IsNaN(value) || math.IsInf(value, 0) {
					return fmt.Errorf("embed %s: non-finite vector", kind)
				}
			}
			chunks[start+i].Vector = vector
		}
	}
	return nil
}
