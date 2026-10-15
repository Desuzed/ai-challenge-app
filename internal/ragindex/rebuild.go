package ragindex

import (
	"context"
	"path/filepath"
	"sync"
)

// Rebuilder rereads the corpus and publishes complete indexes only after all
// embeddings have been built. Unchanged text may reuse its existing vectors.
type Rebuilder struct {
	Root, Output, Model string
	Sources             []string
	Embedder            Embedder
	mu                  sync.Mutex
}

func (r *Rebuilder) Rebuild(ctx context.Context) (Report, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	paths := r.Sources
	var err error
	if paths == nil {
		paths, err = DefaultSources(r.Root)
		if err != nil {
			return Report{}, err
		}
	}
	docs, err := Load(r.Root, paths)
	if err != nil {
		return Report{}, err
	}
	output := r.Output
	if !filepath.IsAbs(output) {
		output = filepath.Join(r.Root, output)
	}
	cache, err := LoadEmbeddingCache(output, r.Model)
	if err != nil {
		return Report{}, err
	}
	indexes, report, err := BuildCached(ctx, docs, r.Model, r.Embedder, cache)
	if err != nil {
		return Report{}, err
	}
	if err := ctx.Err(); err != nil {
		return Report{}, err
	}
	if err := Save(output, indexes, report); err != nil {
		return Report{}, err
	}
	return report, nil
}
