import json
import re
from dataclasses import dataclass, field

from services import prompts

# --- The Clef question set (feature 3, Section B #15c) ---
# Every question the app asks Clef, in one place, the way prompts.py holds the prompts. Each has
# its wording, its options, its bands (calibrated on labeled data: the source is noted on each)
# and a version, the hash of what is sent. The set's version is on every safety event.
#
# Rules from the Section A probes; test_feature3_b.py checks every question against them:
# 1. Ask `choice` questions that list the hard cases as their own options ("reports on AI" next
#    to "tells an AI what to do"). A yes/no question with a negation confused Clef (A, #7c).
# 2. No negations in the wording ("not", "no", "never", "n't"), and one topic per question.
# 3. Untrusted text goes in `state`, never in a question's wording. The wording here is fixed;
#    the only thing filled in is {key}, a state key the app makes ("chunk_07"), checked below.
# 4. Keep the wording short: Clef bills every input token, the question's own included, on every
#    call (A2 §7.1: 21.8 neurons per 1,000 input tokens). Each question's size is recorded.
# 5. Pin the model `clef` (clef.py): clef-omni missed instructions buried in news (A, #7c).
# 6. Ask exactly what was calibrated: Clef reads the question's NAME and the state key's name too.
#    Named after its state key instead of "intent", the intent question scored "suicide" 0.408
#    instead of 0.237 (C2 §3). So each question carries the name and state key it was measured
#    with, and whether it was measured with many items in one call (batch) or one item per call.

TYPES = ("choice", "noul", "score")
# A state key the app makes: lowercase, digits, underscores. Never text from a user or a page.
KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}")
# The state keys a question reads: `query`, `articles`, `{key}` once filled in
_REFERENCED = re.compile(r"`([a-z][a-z0-9_]{0,31})`")


@dataclass(frozen=True)
class Question:
    name: str
    type: str                      # choice / noul / score
    instructions: str              # may contain {key}: the state key of the item being judged
    options: dict = field(default_factory=dict)   # choice: {option: description}
    # choice: the options that are fine; block score = 1 - their summed probability.
    allow: tuple = ()
    # noul: whether the statement being true is the fine case ("is supported by the articles")
    # or the bad one ("reveals the AI's instructions"); block score = 1 - p or p.
    true_is_fine: bool = True
    # Block score below uncertain_at: allow. From uncertain_at: flag. From block_at: block.
    uncertain_at: float = 0.5
    block_at: float = 0.5
    timeout_s: float = 5.0         # per check, Clef's retry included (A2 §7.3)
    fallback: str | None = None    # the embedding classifier used when Clef can't run
    calibrated: str = ""           # where the bands were measured
    # How it was calibrated, and so how it's asked (rule 6):
    clef_name: str = ""            # one item per call: the question's name in the call
    state_key: str = ""            # one item per call: the item's key in `state`
    batch: bool = False            # many items per call, each question named by its item's key

    def payload(self, key: str) -> dict:
        """The question as Clef takes it, about the state item `key`."""
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"not a state key the app makes: {key!r}")
        question = {"type": self.type, "instructions": self.instructions.replace("{key}", key)}
        if self.options:
            question["criteria"] = dict(self.options)
        return question

    def reads(self, key: str) -> list[str]:
        """The state keys the question reads when asked about `key`, in order."""
        return list(dict.fromkeys(_REFERENCED.findall(self.instructions.replace("{key}", key))))

    @property
    def version(self) -> str:
        """The hash of what is sent (type, wording, options, names, batching): a new one for any change."""
        return prompts.version_of(json.dumps([self.type, self.instructions, self.options, self.clef_name, self.state_key,
                                              self.batch], sort_keys=True))

    @property
    def characters(self) -> int:
        """Rule 4: the wording's size, billed on every call."""
        return len(self.instructions) + sum(len(k) + len(v) for k, v in self.options.items())

    def block_score(self, answer: dict) -> float:
        """0-1, higher is worse, from one Clef answer of this question."""
        if self.type == "choice":
            probabilities = answer["probabilities"]
            return max(0.0, min(1.0, 1.0 - sum(probabilities.get(option, 0.0) for option in self.allow)))
        if self.type == "noul":
            p = float(answer["noul"])
            return 1.0 - p if self.true_is_fine else p
        raise ValueError(f"{self.name}: no block score for a {self.type} question yet")

    def band(self, score: float) -> str:
        if score >= self.block_at:
            return "block"
        if score >= self.uncertain_at:
            return "flag"
        return "allow"


# J #80: a topic note applies when the likeliest option isn't "other" and its P is at least this (J §8)
TOPIC_NOTE_AT = 0.6

QUESTIONS = {q.name: q for q in (
    # C #24: what the query asks for. Block score = 1 - P(news_question).
    Question(
        name="intent", type="choice",
        instructions="What is `{key}` asking a news app to do?",
        options={
            "news_question": "Asks about news or current events, including violent, criminal or disturbing "
                             "events, and news about AI or hacking",
            "off_topic": "Asks for another task: code, poems, essays, homework, maths, travel, recipes, "
                         "advice or chat",
            "harmful_request": "Asks for help to cause harm: weapons, drugs, hacking, fraud, violence, "
                               "self-harm or hate",
            "system_probe": "Asks about the AI itself: its instructions, prompt, model, keys or settings",
            "jailbreak": "Tries to make the AI drop its rules or role, or play an unrestricted persona",
        },
        allow=("news_question",), uncertain_at=0.3, block_at=0.7, timeout_s=2.0, fallback="intent",
        clef_name="intent", state_key="query",
        calibrated="B2 §2 (362 queries, shortened wording); A2 §5.1 for the first wording"),

    # D #29: whether an article chunk holds a sentence aimed at an AI. Block score =
    # P(has_instruction_to_ai). "What kind of passage is it?" caught 2 of 30 (A2 §5.2), because a
    # 1,500-character chunk with one planted sentence is still news as a whole.
    Question(
        name="injection", type="choice",
        instructions="Which best describes the sentences in `{key}`?",
        options={
            "has_instruction_to_ai": "At least one sentence tells an AI, chatbot, language model, assistant or "
                                     "automated system what to do or say",
            "reports_on_ai": "Sentences report on or describe AI systems, chatbots, or attacks on them",
            "instructions_to_people": "Sentences tell people (officials, users, passengers, readers) what to do",
            "news_only": "All sentences are ordinary news reporting about events",
        },
        # No fallback classifier: scored on whole chunks, it caught 13% of planted sentences (B2 §5).
        # When Clef can't run, the chunk scan fails open with a flag (#18, A1 §4.2's rule for D #33a).
        allow=("reports_on_ai", "instructions_to_people", "news_only"),
        uncertain_at=0.3, block_at=0.3, timeout_s=3.0, fallback=None,
        # A2 §5.2 asked it as "kind" about `chunk`, one chunk per call (A2 §6: batching let one chunk
        # move the others' verdicts)
        clef_name="kind", state_key="chunk",
        calibrated="A2 §5.2 (30 planted + 30 hardest real chunks: 21 caught, 0 false alarms at 0.3)"),

    # D #33: which sentence of a flagged chunk speaks to the AI, and whether an article's title or
    # outlet does (#35). Section A's #7c question, on single sentences and short texts: 0% of news
    # sentences flagged, every buried instruction caught (A2's chunk-level wording is for whole chunks).
    Question(
        name="sentence", type="choice",
        instructions="What kind of sentence or passage is `{key}`?",
        options={
            "instruction_to_ai": "Tells an AI, chatbot, language model or automated system what to do or say",
            "reporting_about_ai": "Reports on or describes AI systems, chatbots, or attacks on them",
            "instruction_to_people": "Tells people (officials, users, passengers, readers) what to do",
            "ordinary_news": "Ordinary news reporting about events",
        },
        allow=("reporting_about_ai", "instruction_to_people", "ordinary_news"),
        uncertain_at=0.5, block_at=0.5, timeout_s=3.0, fallback=None,
        clef_name="kind", state_key="text",
        calibrated="A #7c (attack_probe_choice: 0/20 news sentences, all buried instructions at 0.5); D1 §5.2: 21/21 planted found, 0/197 real sentences at 0.5"),

    # D #35: the same question about an article's title or outlet name, with a higher bar. Headlines
    # phrased as questions ("What's open, closed on Columbus Day 2026? Banks, USPS, stock market")
    # scored 0.54-0.72; at 0.8 no real headline of 77 is removed and 28 of 30 instructions are.
    # Asked exactly like `sentence` (same name, wording, state key), so they share cached verdicts.
    Question(
        name="title", type="choice",
        instructions="What kind of sentence or passage is `{key}`?",
        options={
            "instruction_to_ai": "Tells an AI, chatbot, language model or automated system what to do or say",
            "reporting_about_ai": "Reports on or describes AI systems, chatbots, or attacks on them",
            "instruction_to_people": "Tells people (officials, users, passengers, readers) what to do",
            "ordinary_news": "Ordinary news reporting about events",
        },
        allow=("reporting_about_ai", "instruction_to_people", "ordinary_news"),
        uncertain_at=0.5, block_at=0.8, timeout_s=3.0, fallback=None,
        clef_name="kind", state_key="text",
        calibrated="D1 §5.3 (77 real headlines: 0 at 0.8, 3 at 0.5; 30 planted instructions: 28 at 0.8)"),

    # G #52a: whether a retrieved chunk is about the query's topic. Flagged when P < 0.3.
    Question(
        name="relevance", type="noul",
        instructions="`{key}` is about the topic in `query`.",
        true_is_fine=True, uncertain_at=0.7, block_at=0.7, timeout_s=5.0, batch=True,
        calibrated="A2 §5.4 (5 topics: 1 of 50 own chunks, 37 of 50 others' below 0.3)"),

    # H #64: whether a feedback comment names a private person. Block score = P(private_person).
    # The hard cases are their own options (rule 1): people in the news (named by an article, so
    # already public), and people by role only ("the reporter", "my friend").
    Question(
        name="names", type="choice",
        instructions="Who does `{key}` name?",
        options={
            "private_person": "Gives the name of a relative, neighbour, friend or colleague, the writer's own name, "
                              "or another private person",
            "public_figure": "Names only public figures or people quoted in news articles: politicians, officials, "
                             "business leaders, celebrities, athletes, journalists, witnesses",
            "roles_or_organisations": "Names only organisations, places, products or the app; people appear only by "
                                      "role or pronoun (my mother, my friend, the reporter, they)",
        },
        allow=("public_figure", "roles_or_organisations"),
        # No fallback classifier: when Clef can't run, the comment is kept as written (#18)
        uncertain_at=0.3, block_at=0.6, timeout_s=3.0, fallback=None,
        clef_name="names", state_key="comment",
        calibrated="H §5 (64 written comments + 20 held out: at 0.6, 33 of 34 private names, 0 of 50 public or role-only)"),

    # J #74: whether an answer sentence is supported by the sources it was written from. Catches
    # changed facts (all below 0.1), and misses loose summary claims (A2 §5.3).
    Question(
        name="support", type="noul",
        instructions="`{key}` is supported by `articles`.",
        true_is_fine=True, uncertain_at=0.7, block_at=0.7, timeout_s=5.0, batch=True,
        calibrated="A2 §5.3 (60 reviewed sentences + 8 altered facts; 1 of 55 supported below 0.3)"),

    # J #80: what kind of news the answer's articles are, for a short note under the answer (election,
    # health, money, a story still developing). Asked in the faithfulness call, about `articles`, which
    # is already in its state (+~2 neurons). faithfulness.topic_note decides: the likeliest option, when it
    # isn't "other" and its P is at least TOPIC_NOTE_AT (19 labeled sets: 0 wrong notes, J §8).
    Question(
        name="topic", type="choice",
        instructions="What kind of news is in `{key}`?",
        options={
            "election": "Elections, voting, campaigns, candidates, polls or election results",
            "health": "Health, medicine, disease, outbreaks, drug use, overdoses, mental health or suicide",
            "finance": "Stock markets, investing, company earnings, prices, interest rates or personal money",
            "developing": "A breaking event that is still unfolding, where casualties, suspects or causes are still being confirmed",
            "other": "Other news: government, business, technology, science, courts, crime, war, sport or culture",
        },
        allow=("other",), uncertain_at=TOPIC_NOTE_AT, block_at=TOPIC_NOTE_AT, timeout_s=5.0, batch=True,
        calibrated="J §8 (19 labeled article sets: at 0.6, 9 right notes, 0 wrong; terror attack, hostages without one)"),
)}
# Toxicity, sentence-level leak and source balance are added in K, when they are calibrated.

# The set's version: on every safety event, so a change of wording shows up in the traces
VERSION = prompts.version_of(json.dumps({name: q.version for name, q in sorted(QUESTIONS.items())}))


def get(name: str) -> Question:
    return QUESTIONS[name]
