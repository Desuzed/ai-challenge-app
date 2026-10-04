package main

import "testing"

func TestSemanticProxySupportsAlternativesAndIgnoresCitationBlock(t *testing.T) {
	tc := testCase{ExpectedTerms: []string{"weather"}, ExpectedAny: [][]string{{"one minute", "1 minute"}}}
	if !semanticProxy("The weather collection runs every 1 minute.", tc) {
		t.Fatal("one accepted alternative should satisfy the proxy")
	}
	if semanticProxy("The weather is discussed.\n\nИсточники и цитаты\nQuote: 1 minute", tc) {
		t.Fatal("without a minute value in the answer, the proxy should fail")
	}
}

func TestResumeRowsReplaceByQuestionAndRejectDuplicates(t *testing.T) {
	rows := []row{{Number: 1, Question: cases[0].Question, Answer: "old"}}
	rows = upsertRow(rows, row{Number: 1, Question: cases[0].Question, Answer: "new"})
	if len(rows) != 1 || rows[0].Answer != "new" {
		t.Fatalf("upsert did not replace prior row: %#v", rows)
	}
	if !validSavedQuestions(rows) || validSavedQuestions(append(rows, rows[0])) {
		t.Fatal("resume validation accepted an invalid or duplicate question list")
	}
}
