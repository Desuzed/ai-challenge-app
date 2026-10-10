package ollama

// A native schema prevents small models inventing different field names or
// endlessly emitting whitespace under Ollama's unconstrained JSON-object mode.
// The optional plan preserves the ordinary chat's RAG + planner envelope.
func groundedSchema(compact, planner bool) map[string]any {
	text := map[string]any{"type": "string"}
	claimText, quoteText := text, text
	chunkID := text
	answer := text
	if compact {
		answer = map[string]any{"type": "string", "enum": []string{""}}
		// Hard string limits force the grammar to close quotes mid-word.
		// Keep sentences intact; brevity is instructed in the prompt, and a
		// token-limit failure is handled by one complete regeneration.
		chunkID = map[string]any{"type": "string", "maxLength": 64}
	}
	evidence := map[string]any{"type": "array", "minItems": 1, "items": map[string]any{
		"type": "object", "required": []string{"chunkId", "quote"}, "additionalProperties": false,
		"properties": map[string]any{"chunkId": chunkID, "quote": quoteText},
	}}
	claims := map[string]any{"type": "array", "items": map[string]any{
		"type": "object", "required": []string{"text", "evidence"}, "additionalProperties": false,
		"properties": map[string]any{"text": claimText, "evidence": evidence},
	}}
	if compact {
		claims["maxItems"] = 2
		evidence["maxItems"] = 1
	}
	properties := map[string]any{"answer": answer, "claims": claims}
	if planner {
		properties["plan"] = map[string]any{"type": "object"}
	}
	return map[string]any{
		"type": "object", "required": []string{"answer", "claims"}, "additionalProperties": false,
		"properties": properties,
	}
}
