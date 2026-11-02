"""
Builds static/tailwind.css, the page's only stylesheet (feature 3 #110), with Tailwind's standalone CLI:
one pinned executable, no Node or npm. Run it after adding or changing Tailwind classes in
templates/index.html or static/app.js, and commit static/tailwind.css (Render serves the file as built).

  python scripts/build_css.py           build static/tailwind.css
  python scripts/build_css.py --check   rebuild into a temporary file and fail if static/tailwind.css differs

The CLI is downloaded once into .local/tools/ (git-ignored) from Tailwind's GitHub release and checked
against the release's SHA-256 before it is run.
"""
import argparse
import hashlib
import platform
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "3.4.17"
# From the release's sha256sums.txt (github.com/tailwindlabs/tailwindcss/releases/tag/v3.4.17)
SHA256 = {
    "windows-x64.exe": "67f1c5e3f5a03406a7bf5badf5ada09b79f3ae78ec43450c15f7e983068da346",
    "windows-arm64.exe": "76f516476784c00f1562160b5758e3d8f0e6c48957efb26b5b50fbdfd76aa382",
    "linux-x64": "7d24f7fa191d2193b78cd5f5a42a6093e14409521908529f42d80b11fde1f1d4",
    "linux-arm64": "69b1378b8133192d7d2feb12a116fa12d035594f58db3eff215879e4ad8cf39b",
    "macos-x64": "6cbdad74be776c087ffa5e9a057512c54898f9fe8828d3362212dfe32fc933a3",
    "macos-arm64": "a1d0c7985759accca0bf12e51ac1dcbf0f6cf2fffb62e6e0f62d091c477a10a3",
}
CONFIG = ROOT / "styles" / "tailwind.config.js"
INPUT = ROOT / "styles" / "input.css"
OUTPUT = ROOT / "static" / "tailwind.css"
TOOLS = ROOT / ".local" / "tools"


def asset() -> str:
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system())
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64" if machine in ("x86_64", "amd64") else None
    name = f"{system}-{arch}" + (".exe" if system == "windows" else "")
    if system is None or arch is None or name not in SHA256:
        sys.exit(f"No pinned Tailwind CLI for {platform.system()} {platform.machine()}.")
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cli() -> Path:
    """The pinned CLI, downloaded and verified the first time."""
    name = asset()
    path = TOOLS / f"tailwindcss-v{VERSION}{'.exe' if name.endswith('.exe') else ''}"
    if not path.exists():
        TOOLS.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/tailwindlabs/tailwindcss/releases/download/v{VERSION}/tailwindcss-{name}"
        print(f"Downloading {url}")
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    if sha256(path) != SHA256[name]:
        path.unlink()
        sys.exit(f"{path.name} doesn't match the release's SHA-256: deleted. Run again to download it again.")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def build(out: Path):
    subprocess.run([str(cli()), "-c", str(CONFIG), "-i", str(INPUT), "-o", str(out), "--minify"],
                   cwd=ROOT, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if static/tailwind.css isn't up to date")
    args = parser.parse_args()
    if not args.check:
        build(OUTPUT)
        print(f"Built {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes) with Tailwind {VERSION}.")
        return
    with tempfile.TemporaryDirectory(dir=TOOLS if TOOLS.exists() else None) as tmp:
        fresh = Path(tmp) / "tailwind.css"
        build(fresh)
        if not OUTPUT.exists() or fresh.read_bytes() != OUTPUT.read_bytes():
            sys.exit("static/tailwind.css is out of date: run python scripts/build_css.py and commit it.")
    print("static/tailwind.css is up to date.")


if __name__ == "__main__":
    main()
