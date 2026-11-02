import hashlib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from services import observability
from services.safety import clef, facts, questions, settings
from services.safety.engine import Decision
from services.safety.guard import check

# --- Does the answer say what its sources say? (feature 3, Section J #74-#80a) ---
# After an answer is written, each of its sentences is checked against the chunks it was written from:
#   Clef (#74)          one call per answer: the chunks in `state` as `articles`, each sentence under its
#                       own key (s00, s01, ...), noul("`s03` is supported by `articles`"), as calibrated
#                       (A2 §5.3, J §6). Catches changed facts: a reversed direction, a swapped name. Misses
#                       loose summary claims, so the note says the statements "don't match", nothing more.
#   figures/dates/names every figure, date and attributed source must be in the sources (facts.py, #76, #78)
#   embedding (#74a)    when Clef can't run: each sentence's closest chunk by meaning; also run next to Clef
#                       on 1 answer in SAMPLE_EVERY, so the two can be compared on the dashboard
# The same call asks what kind of news the articles are (#80, `topic`), for a note under the answer.
#
# Modes (rule 3): monitor records everything on the trace; enforce also shows the notes. Nothing in an
# answer is ever removed or changed (rule 1, decision #77: answer + note).

SUPPORT = questions.get("support")
TOPIC = questions.get("topic")
# Units of fewer words than this ("Key points:", "Conflicting reports") state no fact worth a question
MIN_WORDS = 3
SAMPLE_EVERY = 10
# The embedding method: a sentence whose closest chunk is less similar than this is unsupported (J §6)
EMBEDDING_UNSUPPORTED_BELOW = 0.70

TOPIC_NOTES = {
    "election": "Election news: for voting dates, rules and official results, check your election authority.",
    "health": "Health news, not medical advice: for your own situation, talk to a doctor or a health service.",
    "finance": "Market and money news, not financial advice.",
    "developing": "This story is still developing: details may change as more is confirmed.",
}

# Feature 2's units (evals/judges.py uses these): one per list item or timeline line, and per sentence
# inside a paragraph. The labels the thresholds were calibrated on are in the same units.
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"”')\]]*\s+(?=[A-Z\"“(\[])")
ABBREVIATION_END = re.compile(r"(?:\b(?:[A-Z]\.){2,}|\b(?:Mr|Mrs|Ms|Dr|St|Inc|Corp|Co|Ltd|Jr|Sr|vs|No|Gov|Sen|Rep|Gen)\.)$")


def units(answer: str) -> list[str]:
    """
    The answer as the checks (and feature 2's judges and labels) see it: one unit per list item or timeline
    line, and per sentence inside a paragraph. Headings and markdown emphasis stay in the text; empty lines go.
    """
    out = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if BULLET.match(line) or re.match(r"^\**\d{4}(-\d{2}){0,2}\**\s*:", line):
            out.append(line)
            continue
        for sentence in (s.strip() for s in SENTENCE_END.split(line) if s.strip()):
            # "U.S. Federal Reserve", "Mr. Altman": the split above cut after an abbreviation
            if out and ABBREVIATION_END.search(out[-1]) and not BULLET.match(out[-1]):
                out[-1] = f"{out[-1]} {sentence}"
            else:
                out.append(sentence)
    return out


def checkable(answer: str) -> list[str]:
    return [u for u in units(answer) if len(re.findall(r"[A-Za-z]{2,}", u)) >= MIN_WORDS]


def rows_of(documents) -> list[dict]:
    """The sources of a view's answer: one row per chunk it was written from."""
    return [{"title": d.metadata.get("title"), "source_name": d.metadata.get("source_name"),
             "published_at": d.metadata.get("published_at"), "text": d.page_content} for d in documents]


def articles_text(rows: list[dict]) -> str:
    """The sources as Clef reads them, the way A2 §5.3 calibrated: a [Source: ...] header per chunk."""
    return "\n\n".join(f"[Source: {r.get('title') or 'untitled'} | {r.get('source_name') or 'unknown outlet'} | "
                       f"Published: {r.get('published_at') or 'unknown date'}]\n{r.get('text') or ''}" for r in rows)


def key(i: int) -> str:
    return f"s{i:02d}"


@dataclass
class Sentence:
    text: str
    support: float | None = None        # Clef: P(supported by articles)
    similarity: float | None = None     # embedding: cosine to the closest chunk
    problems: facts.Problems = field(default_factory=facts.Problems)

    def unsupported(self, engine: str) -> bool:
        if self.problems:
            return True
        if engine == "clef" and self.support is not None:
            return SUPPORT.band(1.0 - self.support) == "block"
        if engine == "embedding" and self.similarity is not None:
            return self.similarity < EMBEDDING_UNSUPPORTED_BELOW
        return False


@dataclass
class Judgement:
    sentences: list[Sentence]
    engine: str                          # clef / embedding / none
    topic: dict | None = None            # Clef's topic answer {choice, probabilities}
    reason: str | None = None            # why the fallback decided, or why it failed open
    neurons: float = 0.0
    sampled: bool = False                # the embedding method also ran (1 in SAMPLE_EVERY)
    review: dict = field(default_factory=dict)       # K: Clef's answers to tone / leak / balance about `answer`
    sentence_vectors: list | None = None             # the embedding method's vectors (K's leak fallback reuses them)

    @property
    def unsupported(self) -> list[int]:
        return [i for i, s in enumerate(self.sentences) if s.unsupported(self.engine)]


# K's questions about the whole answer (tone #83, leak #84, balance #85), asked in the same call
REVIEW = {name: questions.get(name) for name in ("tone", "leak", "balance")}


async def ask(sentences: list[str], articles: str, *, support=True, topic=True, answer: str | None = None,
              review=()) -> dict:
    """
    One Clef call (split by clef.ask past 64 questions or the state limit), with the chunks as `articles`:
    P(supported) per sentence (s00, s01, ...), the topic, and K's questions about the whole `answer`.
    Returns {"support": [P | None], "topic": answer | None, "review": {name: answer}, "neurons"}.
    Raises clef.ClefError.
    """
    state = {"articles": articles, **({key(i): s for i, s in enumerate(sentences)} if support else {})}
    asked = {key(i): SUPPORT.payload(key(i)) for i in range(len(sentences))} if support else {}
    if topic:
        asked["topic"] = TOPIC.payload("articles")
    review = [name for name in review if answer]
    if review:
        state["answer"] = answer
        asked.update({name: REVIEW[name].payload("answer") for name in review})
    reply = await clef.ask(state, asked, timeout=SUPPORT.timeout_s, label="faithfulness")
    answers = reply["answers"]
    return {"support": [float(answers[key(i)]["noul"]) if support else None for i in range(len(sentences))],
            "topic": answers.get("topic") if topic else None, "review": {name: answers[name] for name in review},
            "neurons": reply.get("neurons", 0.0)}


async def ask_clef(sentences: list[str], articles: str, *, support=True, topic=True) -> tuple[list[float | None], dict | None, float]:
    """J's part of the call: (P(supported) per sentence, the topic answer, neurons). Raises clef.ClefError."""
    r = await ask(sentences, articles, support=support, topic=topic)
    return r["support"], r["topic"], r["neurons"]


def normalized(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


async def embed_sentences(sentences: list[str], rows: list[dict]) -> tuple[list[float], np.ndarray]:
    """#74a: (each sentence's cosine similarity to its closest source chunk, the sentences' unit vectors).
    One embedding call for the sentences; the chunks' vectors come from the pipeline's cache."""
    from services import rag_service      # rag_service imports the safety package
    chunks = [r.get("text") or "" for r in rows]
    m = normalized(await rag_service.get_vectors(chunks + sentences, purpose="safety"))
    c, s = m[:len(chunks)], m[len(chunks):]
    sims = [round(float(x), 4) for x in (s @ c.T).max(axis=1)] if len(chunks) else [0.0] * len(sentences)
    return sims, s


async def embedding_similarity(sentences: list[str], rows: list[dict]) -> list[float]:
    return (await embed_sentences(sentences, rows))[0]


def sampled(text: str) -> bool:
    return int(hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:8], 16) % SAMPLE_EVERY == 0


async def judge_answer(text: str, rows: list[dict], *, support=True, topic=True, review=()) -> Judgement:
    """Every check on one answer: figures/dates/names always (when support is on), then Clef, else the
    embedding method, else nothing (fail open). Never raises for an engine failure."""
    sentences = [Sentence(u) for u in checkable(text)]
    if support:
        src = facts.read_sources(rows)
        for s in sentences:
            s.problems = facts.check_sentence(s.text, src)
    texts = [s.text for s in sentences]
    j = Judgement(sentences, engine="none")
    if (support and texts) or topic or review:
        if settings.engine() == "clef":
            try:
                r = await ask(texts, articles_text(rows), support=support and bool(texts), topic=topic, answer=text, review=review)
                j.topic, j.neurons, j.review = r["topic"], r["neurons"], r["review"]
                for s, p in zip(sentences, r["support"]):
                    s.support = p
                j.engine = "clef"
            except clef.ClefError as e:
                j.reason = f"clef {e.kind}"
        else:
            j.reason = "SAFETY_ENGINE=embedding"
    run_embedding = support and texts and (j.engine != "clef" or sampled(text))
    if run_embedding:
        try:
            sims, j.sentence_vectors = await embed_sentences(texts, rows)
            for s, sim in zip(sentences, sims):
                s.similarity = sim
            if j.engine != "clef":
                j.engine = "embedding"
            else:
                j.sampled = True
        except Exception as e:
            j.reason = (f"{j.reason}; " if j.reason else "") + f"embedding failed: {type(e).__name__}"
    return j


def _label(j: Judgement) -> str:
    bad = j.unsupported
    n = len(j.sentences)
    if not bad:
        return f"0 of {n} sentences"
    by_engine = sum(1 for i in bad if not j.sentences[i].problems)
    problem_kinds = [k for i in bad for k in j.sentences[i].problems.labels()]
    parts = [f"{len(bad)} of {n} sentences"]
    if by_engine:
        parts.append(f"{by_engine} by {j.engine}")
    for kind in ("figure", "date", "source"):
        if problem_kinds.count(kind):
            parts.append(f"{problem_kinds.count(kind)} {kind}{'s' if problem_kinds.count(kind) > 1 else ''} not in sources")
    return "; ".join(parts)


def topic_note(topic: dict | None) -> tuple[str | None, float]:
    """(the note's option, its P) when the likeliest option isn't "other" and its P is at least TOPIC_NOTE_AT;
    (None, that P) otherwise. J §8: on 19 labeled sets, 0 wrong notes at 0.6."""
    if not topic or not isinstance(topic.get("probabilities"), dict):
        return None, 0.0
    choice = topic.get("choice")
    p = float(topic["probabilities"].get(choice, 0.0)) if choice != "other" else 0.0
    return (choice if choice in TOPIC_NOTES and p >= TOPIC.block_at else None), round(p, 4)


def unsupported_note(n: int) -> str:
    return (f"{n} statement{'s' if n != 1 else ''} in this answer {'don’t' if n != 1 else 'doesn’t'} match the cited "
            "articles. Check them against the sources below.")


async def check_answer(where: str, text: str, rows: list[dict]) -> dict:
    """
    The checks on one answer (a view's, or a Follow check's texts together; `where` is the view or
    "update"). Returns {"notes": [...]}: the notes to show (enforce only). Records the `faithfulness` and
    `sensitive_topic` events (J) and K's `toxicity`, `leak` and `balance` (conduct.py), and puts the
    per-sentence scores on the trace. Raises answer_withheld (502) only for K #83 (hateful wording of the
    model's own, toxicity in enforce); never for an engine failure.
    """
    from services.safety import conduct      # conduct imports this module's helpers
    support_on = settings.mode("faithfulness") != "off"
    topic_on = settings.mode("sensitive_topic") != "off"
    review = conduct.review_for()
    if not (support_on or topic_on or review) or not (text or "").strip():
        return {"notes": []}
    started = time.perf_counter()
    j = await judge_answer(text, rows, support=support_on, topic=topic_on, review=review)
    notes = []

    if support_on:
        async def run_support() -> Decision:
            bad = j.unsupported
            if j.engine == "none" and not bad:
                return Decision.failed_open(j.reason or "no engine", threshold=SUPPORT.block_at)
            # The worst sentence's block score (1 - P(supported)) when Clef judged; else 1 / 0
            clef_scores = [1.0 - s.support for s in j.sentences if s.support is not None]
            score = max(clef_scores) if clef_scores else (1.0 if bad else 0.0)
            return Decision(verdict="block" if bad else "allow", score=score,
                            threshold=SUPPORT.block_at, engine=j.engine if j.engine != "none" else "code",
                            label=_label(j), reason=j.reason)
        decision = await check("faithfulness", run_support, where=where, text=text)
        if decision is not None and decision.enforced:
            notes.append(unsupported_note(len(j.unsupported)))

    if topic_on:
        async def run_topic() -> Decision:
            if j.topic is None:
                return Decision.failed_open(j.reason or "no topic answer", threshold=TOPIC.block_at)
            choice, score = topic_note(j.topic)
            return Decision(verdict="flag" if choice else "allow", score=score, threshold=TOPIC.block_at,
                            engine="clef", label=j.topic.get("choice"), probabilities=j.topic.get("probabilities"))
        decision = await check("sensitive_topic", run_topic, where=where, text=text)
        if (decision is not None and decision.verdict == "flag" and decision.engine == "clef"
                and settings.mode("sensitive_topic") == "enforce" and decision.label in TOPIC_NOTES):
            notes.append(TOPIC_NOTES[decision.label])

    bad = j.unsupported
    observability.annotate(faithfulness={
        "engine": j.engine, "sentences": len(j.sentences), "unsupported": bad,
        "support": [s.support for s in j.sentences], "similarity": [s.similarity for s in j.sentences] if any(
            s.similarity is not None for s in j.sentences) else None,
        "problems": {i: {"figures": s.problems.figures, "dates": s.problems.dates, "names": s.problems.names}
                     for i, s in enumerate(j.sentences) if s.problems},
        "topic": j.topic, "neurons": j.neurons, "sampled": j.sampled, "reason": j.reason,
        "ms": round((time.perf_counter() - started) * 1000)})
    if support_on and j.sentences:
        observability.add_totals(faithfulness_unsupported=len(bad), faithfulness_sentences=len(j.sentences))
        observability.feedback("safety.faithfulness", round(1 - len(bad) / len(j.sentences), 4),
                               comment=_label(j) if bad else None)
    # K: the model's own wording, leaks and balance, from the same call (may withhold: K #83)
    if review or any(settings.mode(name) != "off" for name in conduct.CHECKS):
        await conduct.record(where, j, text, rows, view=where)
    return {"notes": notes}
