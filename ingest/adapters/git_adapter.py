"""
git adapter: turn a repo's commit history into chunks. One commit = one chunk.

Source config (from corpus.yaml):
    {type: git, path: <repo path>, repo: <short name for citations>}

YOU implement load_git. It should:

  1. Run `git log` on `path` and get, per commit:
       - short sha, ISO date, subject + body (the full message)
       - diff stats: files changed + insertions/deletions (see --numstat)
     Hint: one robust way is `git log --no-merges --numstat --date=short
     --pretty=<format>` with a rare separator token in the pretty format, so
     you can split the output into per-commit records without the message body
     confusing the parser.

  2. Skip commits that carry no signal:
       - merge commits (--no-merges already drops these)
       - trivial/lockfile-only commits (e.g. only requirements.txt or a lockfile
         changed) — your call how strict

  3. Build ONE chunk per surviving commit with make_chunk(...):
       - text        = message + a short summary of what changed (files, +/-),
                        i.e. what a retriever should match against
       - source_file = f"{repo}@{short_sha}"   (this is the citation identity)
       - source_type = "git"
       - chunk_id    = something unique, e.g. f"{repo}@{short_sha}"
       - metadata    = repo=..., sha=..., date=...   (extra kwargs to make_chunk)
     page/char_start/char_end aren't meaningful for a commit — let make_chunk
     default them.

Edge cases to get right (these are where git-log parsing bites):
  - multi-line commit messages (bodies with blank lines)
  - binary files in --numstat show "-\t-" instead of numbers
  - the very first commit / empty repo
  - a repo with zero non-merge commits

Return: list[chunk dict]. Empty list is valid (e.g. empty repo) — don't crash.
"""
import subprocess

from ingest.adapters.base import register, make_chunk


def _run_git_log(repo_path: str, pretty: str) -> str:
    """Run git log in repo_path and return raw stdout. (Helper you can use.)"""
    result = subprocess.run(
        ["git", "log", "--no-merges", "--numstat", "--date=short",
         f"--pretty={pretty}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git log failed in {repo_path!r}: {result.stderr.strip()}"
        )
    return result.stdout


@register("git")
def load_git(source: dict) -> list[dict]:
    repo_path = source["path"]
    repo = source["repo"]
    log = _run_git_log(repo_path , "format:%x00%h%n%ad%n%s%n%b")    
    chunks = [  ]
    for block in log.split("\x00"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        sha = lines[0]
        date = lines[1]
        subject = lines[2]
        
        for i , line in enumerate(lines[3:], start=3):
            if line == "":
                x = i

        body = lines[3:x]
        numstat = lines[x:]
        added = 0
        deleted = 0
        changed = []
        for line in numstat:
            if not line:
                continue
            a , b , path = line.split("\t")
            changed.append(path)
            added += 0 if a == "-" else int(a)
            deleted  += 0 if b == "-" else int(b)

        change_summary = f"Changed: {', '.join(changed)} (+{added}/-{deleted})"
        text = subject + "\n" + "\n".join(body) + "\n" + change_summary
        chunk = make_chunk(
            text = text,
            chunk_id = f"{repo}@{sha}",
            source_file = f"{repo}@{sha}",
            source_type ="git",
            repo=repo, sha=sha, date=date,
        )
        chunks.append(chunk)


    return chunks
