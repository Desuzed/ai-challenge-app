package agent

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"ai-challenge-app/internal/rag"
)

// Overlapping windows may have the same section label. A model can copy a
// quote from the complete window but attribute it to its truncated neighbour.
// Resolve that case only among retrieved chunks of the same file and section;
// fabricated IDs, other documents and ambiguous matches remain invalid.
func resolveGroundedEvidence(evidence groundedEvidence, chunks map[string]rag.Match) (groundedEvidence, bool) {
	original, ok := chunks[evidence.ChunkID]
	if !ok {
		return groundedEvidence{}, false
	}
	if quote := canonicalSourceQuote(original.Chunk.Text, evidence.Quote); quote != "" {
		return groundedEvidence{ChunkID: evidence.ChunkID, Quote: quote}, true
	}
	var recovered groundedEvidence
	for id, candidate := range chunks {
		if id == evidence.ChunkID || candidate.Chunk.Source != original.Chunk.Source || candidate.Chunk.Section != original.Chunk.Section {
			continue
		}
		quote := canonicalSourceQuote(candidate.Chunk.Text, evidence.Quote)
		if quote == "" {
			continue
		}
		if recovered.ChunkID != "" {
			return groundedEvidence{}, false
		}
		recovered = groundedEvidence{ChunkID: id, Quote: quote}
	}
	return recovered, recovered.ChunkID != ""
}

// canonicalSourceQuote always returns a literal substring of the supplied
// chunk. Small models sometimes reflow Markdown, omit code backticks or
// double-escape line breaks. Recover only formatting differences, and only
// when the normalized quote identifies one unambiguous source span. Words,
// case, punctuation, numbers and paths are never corrected or paraphrased.
func canonicalSourceQuote(source, quote string) string {
	quote = strings.TrimSpace(quote)
	if quote == "" {
		return ""
	}
	if strings.Contains(source, quote) {
		return quote
	}
	text, starts, ends := normalizedQuoteText(source)
	variants := []string{quote}
	if strings.Contains(quote, `\n`) || strings.Contains(quote, `\r`) || strings.Contains(quote, `\t`) {
		variants = append(variants, strings.NewReplacer(`\r\n`, "\n", `\n`, "\n", `\r`, "\n", `\t`, "\t").Replace(quote))
	}
	var recovered string
	for _, variant := range variants {
		target, _, _ := normalizedQuoteText(variant)
		if target == "" {
			continue
		}
		index := strings.Index(text, target)
		if index < 0 {
			continue
		}
		// Include overlapping occurrences: partial identifiers must not be
		// silently assigned to the first of several matching spans.
		if strings.Contains(text[index+1:], target) {
			return ""
		}
		start, end := starts[index], ends[index+len(target)-1]
		for start > 0 && source[start-1] == '`' {
			start--
		}
		for end < len(source) && source[end] == '`' {
			end++
		}
		candidate := source[start:end]
		if recovered != "" && recovered != candidate {
			return ""
		}
		recovered = candidate
	}
	return recovered
}

// Byte offsets map a normalized match back onto the untouched source,
// including Unicode text and original line breaks.
func normalizedQuoteText(value string) (string, []int, []int) {
	var b strings.Builder
	var starts, ends []int
	space := false
	for index, r := range value {
		if r == '`' {
			continue
		}
		width := utf8.RuneLen(r)
		if unicode.IsSpace(r) {
			if b.Len() == 0 || space {
				continue
			}
			b.WriteByte(' ')
			starts = append(starts, index)
			ends = append(ends, index+width)
			space = true
			continue
		}
		b.WriteRune(r)
		for i := 0; i < width; i++ {
			starts = append(starts, index)
			ends = append(ends, index+width)
		}
		space = false
	}
	text := b.String()
	if space {
		text = text[:len(text)-1]
		starts, ends = starts[:len(starts)-1], ends[:len(ends)-1]
	}
	return text, starts, ends
}
