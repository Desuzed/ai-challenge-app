package agent

import (
	"strings"
	"testing"

	"ai-challenge-app/internal/rag"
	"ai-challenge-app/internal/ragindex"
)

func TestCanonicalSourceQuoteRecoversFormattingWithoutChangingFacts(t *testing.T) {
	chunking := "Индексатор строит два независимых индекса:\n\n- `fixed`: окна по 1200 символов Unicode с перекрытием 150 символов;\n- `structure`: разделы Markdown по заголовкам, Go по объявлениям функций и типов;\n  длинные разделы дополнительно делятся на окна того же размера."
	files := "Запуск создаёт `.local/rag/index-fixed.json`, `.local/rag/index-structure.json` и\n`.local/rag/report.json`."
	metadata := "Каждый чанк содержит `source`, `title`, `section`, `chunk_id`, исходный текст и\nвектор эмбеддинга."
	tests := []struct{ name, source, quote, want string }{
		{"literal-line-escapes", chunking, "Индексатор строит два независимых индекса:\\n- `fixed`: окна по 1200 символов Unicode с перекрытием 150 символов;\\n- `structure`: разделы Markdown по заголовкам, Go по объявлениям функций и типов; длинные разделы дополнительно делятся на окна того же размера.", chunking},
		{"wrapped-filenames", files, "Запуск создаёт .local/rag/index-fixed.json, .local/rag/index-structure.json и .local/rag/report.json.", files},
		{"metadata-code-markup", metadata, "Каждый чанк содержит source, title, section, chunk_id, исходный текст и вектор эмбеддинга.", metadata},
		{"partial-inline-code", metadata, "source, title, section, chunk_id", "`source`, `title`, `section`, `chunk_id`"},
		{"exact-code-escape", `fmt.Print("a\nb")`, `fmt.Print("a\nb")`, `fmt.Print("a\nb")`},
		{"changed-size", chunking, "fixed: окна по 1000 символов Unicode с перекрытием 150 символов;", ""},
		{"changed-path", files, "Запуск создаёт .local/rag/other.json, .local/rag/index-structure.json и .local/rag/report.json.", ""},
		{"paraphrase", metadata, "Чанки содержат название и идентификатор.", ""},
		{"ambiguous-span", "`source` и другой `source`", "source", "source"},
		{"ambiguous-repair", "`source`, `title` и ещё `source`, `title`", "source, title", ""},
		{"empty-markup", metadata, "` `", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := canonicalSourceQuote(tt.source, tt.quote)
			if got != tt.want || got != "" && !strings.Contains(tt.source, got) {
				t.Fatalf("quote=%q got=%q want=%q", tt.quote, got, tt.want)
			}
		})
	}
}

func TestGroundedClaimsRestoreSourceQuoteButRejectWrongChunkAndNumbers(t *testing.T) {
	source := "Окна `fixed` — 1200 символов\nс перекрытием 150 символов."
	matches := []rag.Match{{Chunk: ragindex.Chunk{ChunkID: "chunk-1", Text: source}}}
	for _, tt := range []struct {
		id, quote string
		valid     bool
	}{
		{"chunk-1", "Окна fixed — 1200 символов с перекрытием 150 символов.", true},
		{"unknown", "Окна fixed — 1200 символов с перекрытием 150 символов.", false},
		{"chunk-1", "Окна fixed — 1200 символов с перекрытием 100 символов.", false},
	} {
		claims := validateGroundedEnvelopeClaims(groundedEnvelope{Claims: []groundedClaim{{Text: "Фиксированные окна: 1200/150.", Evidence: []groundedEvidence{{ChunkID: tt.id, Quote: tt.quote}}}}}, matches)
		if (len(claims) == 1) != tt.valid {
			t.Fatalf("id=%s quote=%s claims=%+v", tt.id, tt.quote, claims)
		}
		if tt.valid && claims[0].Evidence[0].Quote != source {
			t.Fatal("published evidence must be the original source substring")
		}
	}
}

func TestGroundedEvidenceUsesCompleteRetrievedWindowOfSameSection(t *testing.T) {
	prefix := "Запуск создаёт `.local/rag/index-fixed.json`, `.local/rag/index-structure.json` и\n"
	complete := prefix + "`.local/rag/report.json`."
	chunks := map[string]rag.Match{
		"first": {Chunk: ragindex.Chunk{Source: "README.md", Section: "Индексация", ChunkID: "first", Text: prefix + "`.lo"}},
		"next":  {Chunk: ragindex.Chunk{Source: "README.md", Section: "Индексация", ChunkID: "next", Text: "Перед этим другой абзац.\n" + complete}},
	}
	quote := strings.ReplaceAll(complete, "\n", " ")
	got, ok := resolveGroundedEvidence(groundedEvidence{ChunkID: "first", Quote: quote}, chunks)
	if !ok || got.ChunkID != "next" || got.Quote != complete {
		t.Fatalf("overlapping window evidence = %+v, %v", got, ok)
	}
	for _, tt := range []struct{ source, section string }{{"other.md", "Индексация"}, {"README.md", "Другой раздел"}} {
		candidate := chunks["next"]
		candidate.Chunk.Source, candidate.Chunk.Section = tt.source, tt.section
		chunks["next"] = candidate
		if _, ok := resolveGroundedEvidence(groundedEvidence{ChunkID: "first", Quote: quote}, chunks); ok {
			t.Fatal("evidence must not move to another document or section")
		}
	}
	if _, ok := resolveGroundedEvidence(groundedEvidence{ChunkID: "invented", Quote: quote}, chunks); ok {
		t.Fatal("invented chunk ID must not be repaired")
	}
}
