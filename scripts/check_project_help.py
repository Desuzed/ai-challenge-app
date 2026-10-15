#!/usr/bin/env python3
"""Verify explicit help commands, actual reindexing, citations and read-only Project MCP."""
import argparse
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
OPENER = build_opener(ProxyHandler({}))


def request(app, path, body=None, identity=None):
    identity = identity or uuid.uuid4().hex
    headers = {"Content-Type": "application/json", "Cookie": f"agent_user={identity}; agent_session={identity}"}
    req = Request(app + path, headers=headers,
                  data=None if body is None else json.dumps(body, ensure_ascii=False).encode())
    try:
        with OPENER.open(req, timeout=620) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def run(args):
    identity = uuid.uuid4().hex
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not branch:
        raise RuntimeError("Run on a named branch; detached HEAD is covered by Go tests")
    report = {"createdAt": datetime.now(timezone.utc).isoformat(), "model": args.model,
              "branch": branch, "controls": {}, "questions": []}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def send(message):
        return request(args.app, "/api/agent/chat", {
            "message": message, "recentMessages": 32, "strategy": "sliding_window",
            "model": args.model, "ragEnabled": False,
            "settings": {"temperature": args.temperature, "maxTokens": 2048,
                         **({"contextWindow": 8192, "localThink": False} if args.model.startswith("ollama/") else {})},
        }, identity)

    status, catalog = request(args.app, "/api/mcp/tools")
    assert status == 200, catalog
    project = [t for t in catalog["tools"] if t["name"] == "project_git_branch"]
    assert len(project) == 1 and project[0]["readOnly"], project
    assert len(catalog["servers"]) == 5, catalog["servers"]
    status, call = request(args.app, "/api/mcp/call", {"name": "project_git_branch", "arguments": {}})
    assert status == 200 and not call["isError"] and json.loads(call["content"])["branch"] == branch, call
    report["mcp"] = {"servers": len(catalog["servers"]), "tools": len(catalog["tools"]), "branchMatches": True}
    save()

    index_path = ROOT / ".local/rag/index-structure.json"
    before_time = index_path.stat().st_mtime_ns if index_path.exists() else 0
    print("RUN /reindex (actual index rebuild, no LLM)", flush=True)
    status, rebuilt = send("/reindex")
    report["reindexResponse"] = rebuilt
    report["controls"]["reindexActuallyRan"] = (status == 200 and bool(rebuilt.get("reindex")) and
                         rebuilt.get("ragModelCallSkipped") is True and index_path.stat().st_mtime_ns > before_time)
    save()
    assert report["controls"]["reindexActuallyRan"], rebuilt
    index = json.loads(index_path.read_text())
    chunks = {c["chunk_id"]: c for c in index["chunks"]}
    stale = [f["source"] for f in index["files"]
             if hashlib.sha256((ROOT / f["source"]).read_bytes()).hexdigest() != f["sha256"]]
    docs = {str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*") if p.is_file()
            and p.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}}
    indexed = {f["source"] for f in index["files"]}
    report["controls"]["freshIndexWithAllDocs"] = not stale and docs <= indexed and "README.md" in indexed
    save()
    assert report["controls"]["freshIndexWithAllDocs"], stale

    print("RUN /help (usage only, no persistent mode)", flush=True)
    status, welcome = send("/help")
    report["welcome"] = welcome
    report["controls"]["helpShowsUsageWithoutGeneration"] = (status == 200 and
            welcome.get("ragModelCallSkipped") is True and not welcome.get("ragEnabled") and
            not welcome.get("toolExecutions") and "/help <вопрос>" in welcome.get("answer", ""))
    status, state = request(args.app, "/api/agent/chat", identity=identity)
    report["controls"]["noPersistedHelpMode"] = status == 200 and "helpEnabled" not in state
    save()
    assert report["controls"]["helpShowsUsageWithoutGeneration"] and report["controls"]["noPersistedHelpMode"], welcome

    print("RUN /diff (live branch through MCP, no LLM or RAG)", flush=True)
    status, diff = send("/diff")
    tools = diff.get("toolExecutions", [])
    report["diff"] = diff
    report["controls"]["diffReadsBranchWithoutGeneration"] = (status == 200 and
            branch in diff.get("answer", "") and diff.get("ragModelCallSkipped") is True and
            not diff.get("ragEnabled") and len(tools) == 1 and
            tools[0]["name"] == "project_git_branch" and not tools[0]["isError"] and
            any(s["source"] == "MCP:project_git_branch" for s in diff.get("ragSources", [])))
    save()
    assert report["controls"]["diffReadsBranchWithoutGeneration"], diff

    cases = [
        ("/help расскажи о проекте", ["приложение"], False),
        ("/help За что отвечают каталоги internal, cmd и static?", ["internal", "cmd", "static"], False),
        ("/structure", ["internal", "cmd", "static"], False),
        ("/help Какие MCP-серверы подключены и как получается текущая локальная git-ветка?", ["mcp", "проект", "яндекс", "github", "погод", "новост", "project_git_branch"], False),
        ("/help Где хранится история диалога между перезапусками?", [".local/agent-history.json"], False),
        ("/help Какой цвет обложки у вымышленного документа «Туман-773»?", [], True),
    ]
    for question, terms, expect_refusal in cases:
        print(f"RUN {question}", flush=True)
        status, response = send(question)
        sources = response.get("ragSources", [])
        document_sources = [s for s in sources if not s["source"].startswith("MCP:")]
        tools = response.get("toolExecutions", [])
        exact = bool(document_sources) and all(s["chunkId"] in chunks and
                    chunks[s["chunkId"]]["source"] == s["source"] and
                    chunks[s["chunkId"]]["section"] == s["section"] and
                    s["quote"] and s["quote"] in chunks[s["chunkId"]]["text"] for s in document_sources)
        factual_answer = response.get("answer", "").split("\n\nИсточники", 1)[0].lower()
        checks = {
            "httpOK": status == 200,
            "ragForced": response.get("ragEnabled") is True,
            "noBranchPrefix": "Текущая git-ветка:" not in response.get("answer", ""),
            "noTools": not tools,
            "onlyDocumentSources": not any(s["source"].startswith("MCP:") for s in sources),
            "documentAnswer": (response.get("ragAbstained") is True and not document_sources) if expect_refusal
                              else (exact and not response.get("ragAbstained") and all(t in factual_answer for t in terms)),
        }
        report["questions"].append({"question": question, "status": status, "checks": checks, "response": response})
        save()
        print("PASS" if all(checks.values()) else f"REVIEW {checks}", flush=True)

    status, unknown = send("/not-a-command")
    report["controls"]["unknownCommandNeverGenerates"] = status == 200 and unknown.get("ragModelCallSkipped") is True and "не реализована" in unknown.get("answer", "")
    print("RUN ordinary chat after help commands", flush=True)
    status, plain = send("Ответь одним словом: готово.")
    report["ordinaryChat"] = plain
    report["controls"]["ordinaryChatDoesNotInvokeProjectHelp"] = (status == 200 and
            not plain.get("ragEnabled") and not plain.get("toolExecutions") and
            not plain.get("ragModelCallSkipped") and bool(plain.get("answer")))
    report["passed"] = all(report["controls"].values()) and all(all(q["checks"].values()) for q in report["questions"])
    save()
    print(f"Report: {output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="http://127.0.0.1:18081")
    parser.add_argument("--model", default="ollama/qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default=".local/project-check/help-results.json")
    raise SystemExit(run(parser.parse_args()))
