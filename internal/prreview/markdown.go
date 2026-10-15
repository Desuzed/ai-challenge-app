package prreview

import (
	"fmt"
	"html"
	"net/url"
	"strings"
)

func plain(s string) string {
	s = strings.ReplaceAll(s, "@", "@\u200b")
	s = strings.NewReplacer("\\", "\\\\", "`", "\\`", "*", "\\*", "_", "\\_", "[", "\\[", "]", "\\]", "!", "\\!", "#", "\\#").Replace(s)
	return html.EscapeString(s)
}

func sourceLink(repo, sha, name string, line int) string {
	label := plain(fmt.Sprintf("%s:%d", name, line))
	if repo == "" {
		return label
	}
	parts := strings.Split(name, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return fmt.Sprintf("[%s](https://github.com/%s/blob/%s/%s#L%d)", label, repo, sha, strings.Join(parts, "/"), line)
}

// Markdown escapes generated text and mentions, and bounds the body below
// GitHub's review size limit without cutting a finding halfway through.
func Markdown(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## AI-ревью кода\n\nКоммит: `%s` · база сравнения: `%s`\n\n", r.HeadSHA, r.MergeBaseSHA)
	reviewed, docs, code := 0, 0, 0
	for _, f := range r.Files {
		if f.Reviewed {
			reviewed++
		}
	}
	for _, s := range r.Sources {
		if s.Kind == "documentation" {
			docs++
		} else if s.Kind == "code" {
			code++
		}
	}
	fmt.Fprintf(&b, "Проверено файлов: **%d из %d**. Корпус RAG: %d файлов; найдено фрагментов документации: %d, кода: %d.\n\n", reviewed, len(r.Files), r.CorpusFiles, docs, code)
	b.WriteString("Это автоматические замечания, требующие проверки человеком. Пустой список не доказывает отсутствие ошибок.\n\n")
	omitted := 0
	for _, category := range []struct{ key, title string }{{"bug", "Потенциальные баги"}, {"architecture", "Архитектурные проблемы"}, {"recommendation", "Рекомендации"}} {
		fmt.Fprintf(&b, "### %s\n\n", category.title)
		count := 0
		for _, f := range r.Findings {
			if f.Category != category.key {
				continue
			}
			count++
			var item strings.Builder
			sha, name := r.HeadSHA, f.Path
			if f.Side == "LEFT" {
				sha = r.MergeBaseSHA
				for _, file := range r.Files {
					if file.Path == f.Path && file.PreviousPath != "" {
						name = file.PreviousPath
					}
				}
			}
			fmt.Fprintf(&item, "**[%s] %s** — %s\n\n", plain(f.Severity), plain(f.Title), sourceLink(r.Repository, sha, name, f.Line))
			fmt.Fprintf(&item, "%s\n\nПоследствие: %s\n\nРекомендация: %s\n\n", plain(f.Scenario), plain(f.Impact), plain(f.Recommendation))
			for _, e := range f.Evidence {
				for _, s := range r.Sources {
					if s.ID == e.SourceID {
						line := s.StartLine
						if offset := strings.Index(s.Text, e.Quote); offset >= 0 {
							line += strings.Count(s.Text[:offset], "\n")
						}
						fmt.Fprintf(&item, "Источник: %s\n\n> %s\n\n", sourceLink(r.Repository, s.SHA, s.Path, line), strings.ReplaceAll(plain(e.Quote), "\n", "\n> "))
					}
				}
			}
			if b.Len()+item.Len() < 48000 {
				b.WriteString(item.String())
			} else {
				omitted++
			}
		}
		if count == 0 {
			b.WriteString("Подтверждённых замечаний не сформировано.\n\n")
		}
	}
	b.WriteString("### Покрытие и ограничения\n\n")
	if reviewed == len(r.Files) && len(r.Warnings) == 0 {
		b.WriteString("Пропущенных файлов и предупреждений нет.\n")
	}
	for _, f := range r.Files {
		if f.Reviewed {
			continue
		}
		item := fmt.Sprintf("- %s: %s.\n", plain(f.Path), plain(f.SkipReason))
		if b.Len()+len(item) < 56000 {
			b.WriteString(item)
		}
	}
	for _, warning := range r.Warnings {
		item := "- " + plain(warning) + ".\n"
		if b.Len()+len(item) < 56000 {
			b.WriteString(item)
		}
	}
	if omitted > 0 {
		fmt.Fprintf(&b, "\nИз-за лимита GitHub в текст не вошло замечаний: %d. Полный результат сохранён в локальном JSON-отчёте.\n", omitted)
	}
	return b.String()
}
