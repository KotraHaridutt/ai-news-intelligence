"""
Feature 2, Section F: review the proposed evaluation labels in the browser and save your version.

  python evals/review_labels.py            opens http://127.0.0.1:8765
  python evals/review_labels.py --port 9000 --no-browser

One page, two parts:
  Retrieval    per topic, every chunk: grade 2 (belongs in an answer) / 1 (useful background) /
               0 (off-topic, boilerplate), and whether it dates an event (timeline). The chunks
               the app actually retrieved for each view are marked.
  Fact check   per topic, what the reader was told and each story's excerpts (what the model
               reads): new or nothing new, and the new facts.
Each topic starts from Claude's proposal (with its reasons) or your earlier save. "Save" writes
.local/evals/labels/reviewed/<kind>_<topic>.json, which build_datasets.py then prefers over the
proposal. Local only: the server listens on 127.0.0.1 and reads/writes .local/evals/.
"""
import argparse
import json
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import data

CHUNKS_PER_ARTICLE = 6       # what the fact check reads of each article (fact_check_service.CHUNKS_PER_ARTICLE)


def _excerpt(text: str) -> str:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)   # rag_service.text_splitter
    return "\n\n".join(splitter.split_text(text or "")[:CHUNKS_PER_ARTICLE])


def payload() -> dict:
    retrieval = []
    for t in data.topics():
        labels, status = data.labels("retrieval", t["slug"])
        retrieval.append({"slug": t["slug"], "query": t["query"], "status": status,
                          "articles": [{k: a.get(k) for k in ("n", "title", "source_name", "published_at")} for a in t["articles"]],
                          "chunks": [{"id": c["id"], "article": c["article"], "text": c["text"]} for c in t["chunks"]],
                          "retrieval": t["retrieval"], "labels": (labels or {}).get("chunks", {})})
    fact_check = []
    for case in data.fact_check_cases():
        labels, status = data.labels("fact_check", case["slug"])
        fact_check.append({"slug": case["slug"], "query": case["query"], "status": status,
                           "told": "\n\n".join(t["text"] for t in case["told"]), "known": case["known_stories"],
                           "stories": [{"id": s["id"], "kind": s["kind"], "title": s.get("title"),
                                        "related_to": s.get("related_to"),
                                        "articles": [{"title": a.get("title"), "source_name": a.get("source_name"),
                                                      "published_at": a.get("published_at"), "excerpt": _excerpt(a.get("full_text"))}
                                                     for a in s["articles"]]} for s in case["stories"]],
                           "labels": (labels or {}).get("stories", {})})
    return {"retrieval": retrieval, "fact_check": fact_check}


def _valid(kind: str, slug: str, body: dict) -> str | None:
    """Why a save is refused, or None. Only known topics, only the expected shape."""
    known = {"retrieval": {t["slug"]: {c["id"] for c in t["chunks"]} for t in data.topics()},
             "fact_check": {c["slug"]: {s["id"] for s in c["stories"]} for c in data.fact_check_cases()}}
    if kind not in known or slug not in known[kind]:
        return "unknown topic"
    items = body.get("chunks" if kind == "retrieval" else "stories")
    if not isinstance(items, dict) or set(items) != known[kind][slug]:
        return "every item of the topic must be labelled, and nothing else"
    for item in items.values():
        if kind == "retrieval" and (item.get("grade") not in (0, 1, 2) or not isinstance(item.get("timeline"), bool)):
            return "a chunk needs grade 0, 1 or 2 and timeline true/false"
        if kind == "fact_check" and (item.get("expected") not in ("new", "nothing_new") or not isinstance(item.get("new_facts"), list)):
            return "a story needs expected new / nothing_new and a list of new facts"
    return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, ctype="application/json"):
        raw = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", f"{ctype}; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html")
        elif self.path == "/api/data":
            self._send(200, payload())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            kind, slug, labels = body["kind"], body["slug"], body["labels"]
        except Exception:
            return self._send(400, {"error": "the request wasn't valid JSON with kind, slug and labels"})
        problem = _valid(kind, slug, labels)
        if problem:
            return self._send(422, {"error": problem})
        record = {"topic": slug, "by": "reviewed", "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "started_from": data.labels(kind, slug)[1], **labels}
        path = data.save_reviewed(kind, slug, record)
        self._send(200, {"saved": str(path.relative_to(ROOT))})

    def log_message(self, *args):
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label review</title>
<style>
:root { --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --card:#fff; --line:#e7e5e4; --accent:#2563eb; --ok:#15803d; --warn:#b45309; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524; --line:#44403c; --accent:#60a5fa; --ok:#4ade80; --warn:#fbbf24; } }
body { margin:0; font:15px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
main { max-width:960px; margin:0 auto; padding:16px; }
h1 { font-size:22px; margin:8px 0 4px; } h2 { font-size:18px; margin:24px 0 8px; } .muted { color:var(--muted); font-size:13px; }
nav button, .save { border:1px solid var(--line); background:var(--card); color:var(--fg); padding:6px 12px; border-radius:6px; cursor:pointer; margin:2px; }
nav button.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
.save { background:var(--accent); color:#fff; border:none; } .status { margin-left:8px; font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:8px 0; }
.row { display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
.badge { font-size:11px; border:1px solid var(--accent); color:var(--accent); border-radius:10px; padding:0 6px; }
.changed { border-left:4px solid var(--warn); } .reason { color:var(--muted); font-size:13px; }
details pre { white-space:pre-wrap; font:13px/1.45 ui-monospace, monospace; margin:6px 0 0; max-height:320px; overflow:auto; }
textarea { width:100%; min-height:48px; font:13px system-ui; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; }
label { cursor:pointer; }
</style></head><body><main>
<h1>Label review</h1>
<p class="muted">Start from the proposal (reasons in grey), change what you disagree with, then Save. Changed items get an orange bar.
Grade 2 = belongs in a good answer · 1 = useful background · 0 = off-topic or boilerplate. Timeline = the chunk says when something happened.</p>
<nav id="nav"></nav><div id="view"></div>
</main><script>
let D, current;
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
async function load() { D = await (await fetch("/api/data")).json(); nav(); show(D.retrieval[0] ? ["retrieval", 0] : ["fact_check", 0]); }
function nav() {
  const n = document.getElementById("nav"); n.innerHTML = "";
  for (const [kind, list] of [["retrieval", D.retrieval], ["fact_check", D.fact_check]]) list.forEach((t, i) => {
    const b = el("button", current && current[0] === kind && current[1] === i ? "on" : "", (kind === "retrieval" ? "Chunks: " : "Fact check: ") + t.slug + (t.status === "reviewed" ? " ✓" : ""));
    b.onclick = () => show([kind, i]); n.appendChild(b);
  });
}
function show(sel) { current = sel; nav(); const [kind, i] = sel; (kind === "retrieval" ? showRetrieval : showFact)(D[kind][i]); }
function radio(name, value, checked, text, onchange) {
  const l = el("label"); const r = el("input"); r.type = "radio"; r.name = name; r.value = value; r.checked = checked; r.onchange = onchange; l.append(r, " " + text); return l;
}
function showRetrieval(t) {
  const v = document.getElementById("view"); v.innerHTML = "";
  v.appendChild(el("h2", null, `${t.query}: ${t.chunks.length} chunks (${t.status})`));
  const original = JSON.parse(JSON.stringify(t.labels)); const out = JSON.parse(JSON.stringify(t.labels));
  const inViews = id => Object.entries(t.retrieval).filter(([, ids]) => ids.includes(id)).map(([view]) => view);
  for (const c of t.chunks) {
    const a = t.articles.find(x => x.n === c.article) || {}; const lab = out[c.id] || (out[c.id] = { grade: 1, timeline: false, reason: "" });
    const card = el("div", "card"); const top = el("div", "row");
    top.append(el("strong", null, c.id), el("span", "muted", `${a.title || ""} · ${a.source_name || ""}`));
    for (const view of inViews(c.id)) top.appendChild(el("span", "badge", "retrieved: " + view));
    const ctl = el("div", "row"); const mark = () => card.classList.toggle("changed", JSON.stringify(out[c.id]) !== JSON.stringify(original[c.id]));
    for (const g of [2, 1, 0]) ctl.appendChild(radio("g" + c.id, g, lab.grade === g, String(g), () => { lab.grade = g; mark(); }));
    const tl = el("label"); const cb = el("input"); cb.type = "checkbox"; cb.checked = !!lab.timeline; cb.onchange = () => { lab.timeline = cb.checked; mark(); }; tl.append(cb, " timeline");
    ctl.appendChild(tl); ctl.appendChild(el("span", "reason", lab.reason || ""));
    const det = el("details"); det.appendChild(el("summary", "muted", c.text.slice(0, 140).replace(/\s+/g, " ") + "…")); det.appendChild(el("pre", null, c.text));
    card.append(top, ctl, det); v.appendChild(card);
  }
  v.appendChild(saveBar("retrieval", t, () => ({ query: t.query, chunks: out })));
}
function showFact(t) {
  const v = document.getElementById("view"); v.innerHTML = "";
  v.appendChild(el("h2", null, `${t.query}: ${t.stories.length} stories to judge (${t.status})`));
  v.appendChild(el("p", "muted", "Rule (the app's prompt): stories are answered in order; a story is \"nothing new\" if every fact in it is in what was told OR was already given by an earlier story in this list."));
  const told = el("details", "card"); told.appendChild(el("summary", null, `What the reader was told (${t.told.length.toLocaleString()} characters) · known stories: ` + t.known.map(k => k.id).join(", "))); told.appendChild(el("pre", null, t.told)); v.appendChild(told);
  const original = JSON.parse(JSON.stringify(t.labels)); const out = JSON.parse(JSON.stringify(t.labels));
  for (const s of t.stories) {
    const lab = out[s.id] || (out[s.id] = { expected: "new", new_facts: [], reason: "" });
    const card = el("div", "card"); const mark = () => card.classList.toggle("changed", JSON.stringify(out[s.id]) !== JSON.stringify(original[s.id]));
    const head = el("div", "row"); head.append(el("strong", null, s.id), el("span", "muted", s.kind === "update" ? `update of known story: ${s.title || ""}` : (s.related_to ? `new, related to ${s.related_to.id}: ${s.related_to.title}` : "new story")));
    const ctl = el("div", "row");
    for (const [val, text] of [["new", "new"], ["nothing_new", "nothing new"]]) ctl.appendChild(radio("e" + s.id, val, lab.expected === val, text, () => { lab.expected = val; mark(); }));
    ctl.appendChild(el("span", "reason", lab.reason || ""));
    const facts = el("textarea"); facts.value = (lab.new_facts || []).join("\n"); facts.placeholder = "New facts, one per line";
    facts.oninput = () => { lab.new_facts = facts.value.split("\n").map(x => x.trim()).filter(Boolean); mark(); };
    card.append(head, ctl, facts);
    for (const a of s.articles) { const d = el("details"); d.appendChild(el("summary", "muted", `${a.title} · ${a.source_name} · ${a.published_at}`)); d.appendChild(el("pre", null, a.excerpt)); card.appendChild(d); }
    v.appendChild(card);
  }
  v.appendChild(saveBar("fact_check", t, () => ({ stories: out })));
}
function saveBar(kind, t, labels) {
  const bar = el("div", "row"); const b = el("button", "save", "Save my review"); const st = el("span", "status");
  b.onclick = async () => {
    st.textContent = "Saving…";
    try {
      const r = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, slug: t.slug, labels: labels() }) });
      const j = await r.json();
      if (!r.ok) { st.textContent = "Not saved: " + (j.error || r.status); return; }
      t.status = "reviewed"; t.labels = labels()[kind === "retrieval" ? "chunks" : "stories"]; st.textContent = "Saved ✓ " + j.saved; nav();
    } catch (e) { st.textContent = "Not saved: the review server isn't running."; }
  };
  bar.append(b, st); return bar;
}
load();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Review and save the evaluation labels in the browser.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Label review at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
