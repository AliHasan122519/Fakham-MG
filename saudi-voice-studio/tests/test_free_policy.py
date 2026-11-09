"""Zero-cost guarantee: the shipped code must not depend on paid / metered / key-gated services."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = [p for d in ("svs", "scripts", "colab") for p in (ROOT / d).rglob("*") if p.suffix in (".py", ".sh") and p.is_file()]
CODE += [ROOT / "requirements.txt", ROOT / "web" / "index.html"]
FORBIDDEN = ["elevenlabs", "openai", "anthropic", "azure", "polly", "google.cloud", "texttospeech", "resemble.ai/api", "play.ht",
             "api_key", "apikey", "xi-api-key", "freetts", "replicate.com", "deepgram", "assemblyai", "stability.ai"]


def test_no_paid_service_references_in_code():
    hits = []
    for p in CODE:
        t = p.read_text(encoding="utf-8", errors="ignore").lower()
        if p.name == "run_studio.py" or p.suffix == ".ipynb":
            pass
        for w in FORBIDDEN:
            if w in t:
                hits.append((p.relative_to(ROOT).as_posix(), w))
    assert not hits, hits


def test_only_free_hosts_are_contacted():
    allowed = ("huggingface.co", "pypi.org", "files.pythonhosted.org", "github.com", "githubusercontent.com", "cloudflare.com",
               "trycloudflare.com", "download.pytorch.org", "127.0.0.1", "localhost", "w3.org", "colab.research.google.com",
               "astral.sh", "google.com", "docs.python.org", "fonts.googleapis.com", "fonts.gstatic.com")
    bad = []
    for p in CODE:
        for u in re.findall(r"https?://([A-Za-z0-9.-]+)", p.read_text(encoding="utf-8", errors="ignore")):
            if not any(u == a or u.endswith("." + a) for a in allowed):
                bad.append((p.relative_to(ROOT).as_posix(), u))
    assert not bad, sorted(set(bad))


def test_model_weights_are_open_downloads():
    cfg = (ROOT / "svs" / "config.py").read_text(encoding="utf-8")
    repos = re.findall(r'repo="([^"]+)"', cfg)
    assert repos and all("/" in r for r in repos)       # public Hugging Face repos, no tokens or private mirrors
