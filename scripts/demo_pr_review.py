#!/usr/bin/env python3
"""Run real embeddings + DeepSeek against an intentional regression and a clean refactor.

Only generated demo code is sent to DeepSeek. No GitHub publication is performed.
Requires Ollama/embeddinggemma and DEEPSEEK_API_KEY (environment or project .env).
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


PROJECT = Path(__file__).resolve().parents[1]


def run(args, cwd, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, check=True,
                          stdout=subprocess.PIPE).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / ".local/prreview-demo")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    repo = work / "repo"
    repo.mkdir()
    env = dict(os.environ)
    env.pop("GITHUB_REPOSITORY", None)
    # Never copy the key to the fixture or print it.
    if not env.get("DEEPSEEK_API_KEY") and (PROJECT / ".env").exists():
        for line in (PROJECT / ".env").read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "DEEPSEEK_API_KEY":
                env["DEEPSEEK_API_KEY"] = value.strip().strip("\"'")
                break
    if not env.get("DEEPSEEK_API_KEY"):
        parser.error("DEEPSEEK_API_KEY is not configured")
    binary = work / "prreview"
    run(["go", "build", "-o", str(binary), "./cmd/prreview"], PROJECT, env)

    def git(*argv):
        return run(["git", "-c", "user.name=AI Review Demo", "-c",
                    "user.email=demo@example.test", "-c", "commit.gpgsign=false", *argv], repo)

    def write(name, text):
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)

    git("init", "--quiet", "--initial-branch=main")
    write("README.md", "# First service\nFirst returns the first value, or an empty string when values is empty.\n")
    write("docs/contract.md", "# First empty input contract\nFirst(values) must accept a nil or empty slice without panic and return an empty string. Callers rely on this contract.\n")
    safe = 'package demo\n\nfunc First(values []string) string {\n if len(values) == 0 { return "" }\n return values[0]\n}\n'
    write("first.go", safe)
    write("first_test.go", 'package demo\n\nimport "testing"\n\nfunc TestFirstEmpty(t *testing.T) {\n if got := First(nil); got != "" { t.Fatalf("got %q", got) }\n}\n')
    git("add", ".")
    git("commit", "--quiet", "-m", "document empty input contract")
    base = git("rev-parse", "HEAD")
    cases = [
        ("buggy", safe.replace(' if len(values) == 0 { return "" }\n', "")),
        ("clean", safe.replace(" return values[0]", " first := values[0]\n return first")),
    ]
    results = []
    for name, text in cases:
        git("checkout", "--quiet", "--detach", base)
        write("first.go", text)
        git("add", ".")
        git("commit", "--quiet", "-m", name)
        head = git("rev-parse", "HEAD")
        report_dir = work / name
        # Empty -repo overrides any environment default and keeps demo links local.
        print(run([str(binary), "-root", str(repo), "-base", base, "-head", head,
                   "-repo", "", "-title", "First: preserve the empty input contract",
                   "-output", str(report_dir)], PROJECT, env), flush=True)
        report = json.loads((report_dir / "review.json").read_text())
        counts = {kind: sum(s["kind"] == kind for s in report["sources"])
                  for kind in ("documentation", "code")}
        defects = [f for f in report["findings"] if f["category"] in ("bug", "architecture")]
        results.append({"case": name, "defects": len(defects), "rag": counts,
                        "warnings": report["warnings"], "report": str(report_dir / "review.md")})
    summary = {"results": results, "passed": (
        results[0]["defects"] > 0 and results[1]["defects"] == 0
        and all(all(case["rag"].values()) for case in results))}
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit("Live demo expectations were not met; inspect reports and warnings")


if __name__ == "__main__":
    main()
