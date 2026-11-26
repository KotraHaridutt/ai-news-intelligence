"""
Pins requirements.txt to the installed versions (feature 3, Section N #103).

    .local/venv/Scripts/python.exe scripts/pin_requirements.py            # rewrite requirements.txt
    .local/venv/Scripts/python.exe scripts/pin_requirements.py --check    # exit 1 if it isn't what the venv has

Run it with the Python whose packages the tests pass on (.local/venv). Before this, requirements.txt
named packages without versions, so every Render build installed whatever was newest that day: a
release that broke the app, or a hijacked one, would reach prod on the next deploy without anyone
choosing it.

requirements.txt has two parts:
1. What the code imports, one per line, with its comment. Edit this part by hand: add a package
   there, `pip install` it into .local/venv, run the tests, then run this script.
2. Everything those packages install, pinned too, each with the package that needs it. Written by
   this script; don't edit it by hand.

Part 2 is worked out from the installed packages' own metadata, with the environment markers
evaluated for Render (Linux, the Python version in render.yaml), not for this Windows machine. So a
package only Windows needs (colorama for click) isn't pinned, and a package only Linux needs is
reported, because it can't be pinned from here (none today). Every pinned version is checked to
support Render's Python. Test-only packages in the venv (fakeredis) are left out: they aren't
reached from part 1.

    .local/venv/Scripts/python.exe scripts/pin_requirements.py --releases  # yanked or brand-new pins (PyPI)

--releases asks PyPI about every pin (names and versions only) and lists any that is yanked, or
was released less than COOLDOWN_DAYS ago. Hijacked releases are usually found and pulled within
days, so an upgrade waits a week (Dependabot is told the same, .github/dependabot.yml). Found in
N §3: soupsieve 3.0.1 was yanked.
"""
import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
RENDER = ROOT / "render.yaml"
MARK = "# --- Installed by the packages above. Written by scripts/pin_requirements.py: don't edit by hand. ---"
HEADER = """\
# Feature 3 N #103: exact versions, the ones the tests pass on (.local/venv). Render installs exactly these,
# so a new release reaches prod only when someone pins it here. To add or upgrade a package: pip install it into
# .local/venv, run the tests, then `.local/venv/Scripts/python.exe scripts/pin_requirements.py`.
# Checked by scripts/audit_deps.ps1 (pip-audit, #104) and by GitHub Dependabot alerts (.github/dependabot.yml, #105).
#
# --- What the code imports (edit by hand, with a comment when the reason isn't obvious) ---"""


def render_python(path: Path = RENDER) -> str:
    """The Python version Render builds with (render.yaml's PYTHON_VERSION)."""
    match = re.search(r"key:\s*PYTHON_VERSION\s*\n\s*value:\s*['\"]?([0-9.]+)", path.read_text(encoding="utf-8"))
    if not match:
        sys.exit("render.yaml has no PYTHON_VERSION: can't tell which Python the pins must support")
    return match.group(1)


def marker_env(python: str) -> dict:
    """Environment markers as Render evaluates them (Linux, x86_64, CPython `python`)."""
    return {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
            "implementation_name": "cpython", "platform_python_implementation": "CPython",
            "python_version": ".".join(python.split(".")[:2]), "python_full_version": python}


def read_direct(path: Path = REQUIREMENTS) -> list[tuple[list[str], str]]:
    """Part 1: [(comment lines above it, package name)], in file order. The header is dropped (rewritten)."""
    entries, comments, header_done = [], [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == MARK:
            break
        if not stripped:
            comments = [] if not header_done else comments
            continue
        if stripped.startswith("#"):
            if header_done:
                comments.append(stripped)
            continue
        header_done = True
        entries.append((comments, Requirement(stripped.split(" #")[0].strip()).name))
        comments = []
    return entries


def closure(direct: list[str], env: dict) -> tuple[dict[str, str], list[str]]:
    """{installed name: the package that first needs it} for every package reached from `direct` on
    Render, and the names reached that aren't installed here."""
    needed_by, extras_seen, missing = {}, {}, []
    stack = [(canonicalize_name(name), frozenset(), "") for name in direct]
    while stack:
        name, extras, parent = stack.pop(0)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(f"{name} (needed by {parent or 'requirements.txt'})")
            continue
        key = canonicalize_name(dist.metadata["Name"])
        if key in extras_seen and extras <= extras_seen[key]:
            continue
        extras_seen[key] = extras_seen.get(key, frozenset()) | extras
        needed_by.setdefault(key, parent)
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not any(req.marker.evaluate({**env, "extra": e}) for e in (extras or {""})):
                continue
            stack.append((canonicalize_name(req.name), frozenset(req.extras), dist.metadata["Name"]))
    return needed_by, missing


def pinned(name: str) -> str:
    dist = metadata.distribution(name)
    return f"{dist.metadata['Name']}=={dist.version}"


def unsupported(names, python: str) -> list[str]:
    """Pins whose Requires-Python excludes Render's Python."""
    bad = []
    for name in names:
        dist = metadata.distribution(name)
        spec = dist.metadata.get("Requires-Python")
        if spec and python not in SpecifierSet(spec):
            bad.append(f"{dist.metadata['Name']}=={dist.version} needs Python {spec}")
    return bad


def build(path: Path = REQUIREMENTS, python: str | None = None) -> tuple[str, list[str]]:
    """The pinned file's text, and the problems that stop it from being written."""
    python = python or render_python()
    direct = read_direct(path)
    needed_by, missing = closure([name for _, name in direct], marker_env(python))
    problems = [f"not installed: {m}" for m in missing]
    problems += [f"{p} (Render runs {python})" for p in unsupported(needed_by, python)]
    if problems:
        return "", problems
    lines = [HEADER]
    direct_keys = set()
    for comments, name in direct:
        lines += comments
        lines.append(pinned(name))
        direct_keys.add(canonicalize_name(name))
    lines += ["", MARK]
    rest = sorted((key for key in needed_by if key not in direct_keys), key=lambda k: metadata.distribution(k).metadata["Name"].lower())
    width = max((len(pinned(key)) for key in rest), default=0) + 2
    lines += [f"{pinned(key).ljust(width)}# via {needed_by[key]}" for key in rest]
    return "\n".join(lines) + "\n", []


COOLDOWN_DAYS = 7


def pins_of(path: Path = REQUIREMENTS) -> dict[str, str]:
    """{name: version} of every pin in the file."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def release_problems(pins: dict[str, str], releases: dict[str, dict], today, cooldown_days: int = COOLDOWN_DAYS) -> list[str]:
    """Pins that are yanked, unknown, or younger than the cooldown. `releases` is {name: PyPI's
    "releases" map}; kept apart from the download so it can be tested without the network."""
    from datetime import datetime
    problems = []
    for name, version in sorted(pins.items(), key=lambda kv: kv[0].lower()):
        files = (releases.get(name) or {}).get(version) or []
        if not files:
            problems.append(f"{name}=={version}: PyPI has no files for this version")
            continue
        if all(f.get("yanked") for f in files):
            reason = next((f.get("yanked_reason") for f in files if f.get("yanked_reason")), "") or "no reason given"
            problems.append(f"{name}=={version}: yanked ({reason[:120]})")
        uploaded = min(datetime.fromisoformat(f["upload_time"][:10]).date() for f in files)
        age = (today - uploaded).days
        if age < cooldown_days:
            problems.append(f"{name}=={version}: released {uploaded} ({age} days ago), under the {cooldown_days}-day wait")
    return problems


def fetch_releases(names) -> dict[str, dict]:
    import httpx
    out = {}
    # PyPI redirects a name in another spelling (Jinja2 -> jinja2) to its canonical page
    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": "pin-requirements-check"}) as client:
        for name in names:
            response = client.get(f"https://pypi.org/pypi/{name}/json")
            out[name] = response.json().get("releases", {}) if response.status_code == 200 else {}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="don't write: exit 1 if requirements.txt differs from the venv")
    parser.add_argument("--releases", action="store_true", help="don't write: list pins that are yanked or under the wait (asks PyPI)")
    args = parser.parse_args(argv)
    if args.releases:
        from datetime import date
        pins = pins_of()
        problems = release_problems(pins, fetch_releases(pins), date.today())
        for problem in problems:
            print(f"[pin] {problem}")
        print(f"[pin] {len(pins)} pins asked at PyPI: " + (f"{len(problems)} to look at." if problems else
                                                          f"none yanked, none under {COOLDOWN_DAYS} days old."))
        return 1 if problems else 0
    text, problems = build()
    for problem in problems:
        print(f"[pin] {problem}")
    if problems:
        return 1
    current = REQUIREMENTS.read_text(encoding="utf-8")
    count = sum(1 for line in text.splitlines() if "==" in line and not line.startswith("#"))
    if args.check:
        if text != current:
            print(f"[pin] requirements.txt doesn't match the installed packages ({sys.prefix}). "
                  f"Run scripts/pin_requirements.py to rewrite it.")
            return 1
        print(f"[pin] requirements.txt matches the installed packages: {count} pins.")
        return 0
    REQUIREMENTS.write_text(text, encoding="utf-8", newline="\n")
    print(f"[pin] wrote requirements.txt: {count} pins for Python {render_python()} on Linux.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
