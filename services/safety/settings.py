import os
import re
from dataclasses import dataclass

# --- Safety settings (feature 3, Section B #11, #15b) ---
# Every guard has three modes, set per guard in the environment as SAFETY_<CHECK>:
#   off      the check doesn't run
#   monitor  it runs and records what it decided on the trace and in the log, but never refuses
#   enforce  it also acts: refuses the request, or removes what it found
# The default is monitor, so a new guard is watched on real traffic before anyone is refused
# (the plan's rule 3). A guard is switched from Render's environment settings, no code change.

MODES = ("off", "monitor", "enforce")
DEFAULT_MODE = "monitor"

# Which engine answers the meaning-based checks (C, D, G, H, J, K). With clef, a check falls back
# to the embedding classifier by itself when Clef is off, failing or over its neuron budget.
ENGINES = ("clef", "embedding")
DEFAULT_ENGINE = "clef"


@dataclass(frozen=True)
class Check:
    section: str
    what: str
    # False for a check that only ever adds a note or a flag: enforce is then treated as monitor
    can_block: bool = True
    # Set when the section that runs the check is done; the startup line lists only these
    built: bool = False


# Every guard of the plan, by the name its SAFETY_<NAME> setting and its events use
CHECKS = {
    "intent": Check("C", "what the query asks for: news, off-topic, harmful, probing, jailbreak", built=True),
    "text_shape": Check("C", "scrambled or encoded query text (leetspeak, base64)", built=True),
    "comment": Check("C", "feedback comments (a rating is never refused for its comment)", can_block=False, built=True),
    "injection": Check("D", "article text written to give the AI instructions", built=True),
    "canary": Check("D", "the request's secret string showing up in an answer", built=True),
    "relevance": Check("G", "retrieved chunks that aren't about the query", built=True),
    "pii": Check("H", "personal data in queries, comments, answers, traces and logs", built=True),
    "names": Check("H", "feedback comments that name a private person (masked before saving)", built=True),
    "output": Check("I", "links to pages that aren't sources and images in answers; cut or malformed answers", built=True),
    # J: enforce shows a note under the answer; nothing in an answer is ever removed (decision #77)
    "faithfulness": Check("J", "answer sentences that don't match their sources (note under the answer)", built=True),
    "sensitive_topic": Check("J", "a note under election, health, money and developing-story answers", built=True),
    "coverage": Check("J", "retrieved chunks that are all off-topic: 'not enough coverage' before Gemini", built=True),
    # K: enforce withholds only hateful wording of the model's own (decision #83); leak and balance flag only
    "toxicity": Check("K", "loaded, insulting or hateful wording of the model's own (hateful: withheld)", built=True),
    "leak": Check("K", "an answer that repeats or describes the AI's instructions", can_block=False, built=True),
    "balance": Check("K", "answers that present one side of what the articles say", can_block=False, built=True),
    "rate_limit": Check("L", "per-IP limits and daily shares"),
}

# Not a check: sites the operator decided never to use (feature 3, G #54). Exact domains, comma- or
# space-separated; a domain also covers its subdomains ("example.com" covers "www.example.com").
# Empty by default. A domain is an identifier, not language, so this isn't a keyword list.
BLOCKED_DOMAINS_ENV = "SAFETY_BLOCKED_DOMAINS"
_DOMAIN = re.compile(r"(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}")

# What configure() read, for mode() and the startup line
_modes: dict[str, str] = {}
_engine = DEFAULT_ENGINE
_blocked_domains: frozenset[str] = frozenset()
_warnings: list[str] = []


def env_name(check: str) -> str:
    return f"SAFETY_{check.upper()}"


def configure(environ=None) -> dict:
    """
    Reads SAFETY_ENGINE and every SAFETY_<CHECK>. A value that isn't a mode is ignored with a
    warning (the default applies), so a typo in Render's settings can't stop the app from starting.
    Called once on import; tests call it again with their own environment. Returns the modes.
    """
    global _engine, _blocked_domains
    environ = os.environ if environ is None else environ
    _warnings.clear()
    _modes.clear()
    for name, check in CHECKS.items():
        raw = (environ.get(env_name(name)) or "").strip().lower()
        mode = raw or DEFAULT_MODE
        if mode not in MODES:
            _warnings.append(f"{env_name(name)}={raw!r} isn't one of {', '.join(MODES)}; using {DEFAULT_MODE}")
            mode = DEFAULT_MODE
        if mode == "enforce" and not check.can_block:
            _warnings.append(f"{env_name(name)}=enforce: this check never refuses anything; running it as monitor")
            mode = "monitor"
        _modes[name] = mode

    raw = (environ.get("SAFETY_ENGINE") or "").strip().lower()
    _engine = raw or DEFAULT_ENGINE
    if _engine not in ENGINES:
        _warnings.append(f"SAFETY_ENGINE={raw!r} isn't one of {', '.join(ENGINES)}; using {DEFAULT_ENGINE}")
        _engine = DEFAULT_ENGINE

    _blocked_domains = frozenset(_read_domains(environ.get(BLOCKED_DOMAINS_ENV) or ""))

    # A SAFETY_ setting that matches no check is most likely a typo of one
    known = {env_name(name) for name in CHECKS} | {"SAFETY_ENGINE", BLOCKED_DOMAINS_ENV}
    for key in sorted(k for k in environ if k.startswith("SAFETY_") and k not in known):
        _warnings.append(f"{key} matches no safety check; ignored")
    for line in _warnings:
        print(f"[WARN] Safety: {line}.")
    return dict(_modes)


def _read_domains(raw: str) -> list[str]:
    """'Example.com, https://www.spam.example/' -> ['example.com', 'spam.example']. An entry that isn't a
    domain is skipped with a warning."""
    domains = []
    for entry in re.split(r"[\s,]+", raw.strip().lower()):
        if not entry:
            continue
        domain = re.sub(r"^[a-z]+://", "", entry).split("/", 1)[0].split(":", 1)[0].removeprefix("www.").strip(".")
        if _DOMAIN.fullmatch(domain):
            domains.append(domain)
        else:
            _warnings.append(f"{BLOCKED_DOMAINS_ENV}: {entry!r} isn't a domain; skipped")
    return domains


def blocked_domains() -> frozenset[str]:
    return _blocked_domains


def mode(check: str) -> str:
    """The check's mode. An unknown name is a bug in the caller, so it raises KeyError."""
    return _modes[check]


def engine() -> str:
    return _engine


def summary() -> str:
    """The checks part of the startup line: the built checks with their modes."""
    built = [f"{name}={_modes[name]}" for name, check in CHECKS.items() if check.built]
    waiting = sum(1 for check in CHECKS.values() if not check.built)
    if not built:
        return f"no checks built yet ({waiting} planned)"
    line = ", ".join(built) + (f" ({waiting} more planned)" if waiting else "")
    if _blocked_domains:
        line += f"; {len(_blocked_domains)} blocked domain{'s' if len(_blocked_domains) != 1 else ''}"
    return line


configure()
