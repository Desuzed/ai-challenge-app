package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"ai-challenge-app/internal/ragindex"
)

func main() {
	root := flag.String("root", ".", "project directory containing source documents")
	output := flag.String("output", ".local/rag", "directory for the two indexes and report")
	model := flag.String("model", "embeddinggemma", "Ollama embedding model")
	url := flag.String("ollama-url", "http://127.0.0.1:11434", "local Ollama URL")
	files := flag.String("files", "", "comma-separated relative source paths (default: curated project corpus)")
	flag.Parse()
	paths := ragindex.DefaultFiles
	if strings.TrimSpace(*files) != "" {
		paths = nil
		for _, path := range strings.Split(*files, ",") {
			paths = append(paths, strings.TrimSpace(path))
		}
	}
	docs, err := ragindex.Load(*root, paths)
	if err != nil {
		log.Fatal(err)
	}
	indexDir := *output
	if !filepath.IsAbs(indexDir) {
		indexDir = filepath.Join(*root, indexDir)
	}
	cache, err := ragindex.LoadEmbeddingCache(indexDir, *model)
	if err != nil {
		log.Fatal(err)
	}
	indexes, report, err := ragindex.BuildCached(context.Background(), docs, *model, ragindex.Ollama{URL: *url, Model: *model}, cache)
	if err != nil {
		log.Fatal(err)
	}
	if err := ragindex.Save(indexDir, indexes, report); err != nil {
		log.Fatal(err)
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Fprintln(os.Stdout, string(data))
	fmt.Fprintf(os.Stdout, "Indexes saved in %s\n", indexDir)
}
