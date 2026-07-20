"""
Reads corpus.yaml and returns the source list for a named corpus.

corpus.yaml shape:

    corpora:
      <name>:
        - {type: ..., ...source-specific fields...}
        - ...

See corpus.example.yaml for the committed template.
"""
from pathlib import Path

import yaml

CONFIG_FILE = "corpus.yaml"


def load_corpus(name: str, config_path: str = CONFIG_FILE) -> list[dict]:
    """Return the list of source dicts for corpus `name`."""
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(
            f"No {config_path} found. Copy corpus.example.yaml to {config_path} "
            f"and edit the paths."
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    corpora = data.get("corpora", {})

    if name not in corpora:
        raise SystemExit(
            f"Corpus '{name}' not defined in {config_path}. "
            f"Known corpora: {sorted(corpora)}"
        )

    sources = corpora[name]
    if not sources:
        raise SystemExit(f"Corpus '{name}' in {config_path} has no sources.")
    return sources
