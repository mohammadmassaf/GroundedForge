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
from string import Template

import yaml

CONFIG_FILE = "corpus.yaml"


def _expand(value, roots: dict):
    """Replace ${name} placeholders anywhere inside a parsed YAML value.

    Returns a NEW value of the same shape -- the parsed config is never
    mutated. Recurses on containers, so nothing here knows what a "source"
    is: a field added to an adapter later is expanded without touching this
    function.

    Non-string leaves (ints, bools, None) pass through untouched. A value
    with no placeholder to substitute isn't an error -- the loud failure is
    reserved for a placeholder that was written and can't be resolved.
    """
    if isinstance(value, str):
        try:
            return Template(value).substitute(roots)
        except KeyError as err:
            raise SystemExit(
                f"Unknown root '{err.args[0]}' in {value!r}. "
                f"Defined roots: {sorted(roots)}"
            ) from err
        except ValueError as err:
            # Template couldn't even parse the placeholder ('${vault' , a bare
            # '$'), so there is no name to report -- only the string and where.
            raise SystemExit(
                f"Malformed placeholder in {value!r} ({err})"
            ) from err
    elif isinstance(value, list):
        result = []
        for item in value:
            result.append(_expand(item, roots))
        return result
    elif isinstance(value, dict):
        result = {}
        for key, val in value.items():
            result[key] = _expand(val, roots)
        return result
    return value


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
    roots = data.get("roots", {})

    if name not in corpora:
        raise SystemExit(
            f"Corpus '{name}' not defined in {config_path}. "
            f"Known corpora: {sorted(corpora)}"
        )

    sources = corpora[name]
    if not sources:
        raise SystemExit(f"Corpus '{name}' in {config_path} has no sources.")
    return _expand(sources, roots)
