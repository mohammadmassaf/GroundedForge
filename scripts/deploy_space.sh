#!/bin/sh
# Deploy the current tree to the Hugging Face Space.
#
# WHY NOT JUST `git push space main`
# ----------------------------------
# HF's pre-receive hook rejects binary files that are not in LFS/Xet, and it
# scans the WHOLE history, not the current tree. Two things fall foul of it:
#
#   index/demo_vectors.npz     0.75 MB, present at the tip and required at boot
#   chroma_demo/chroma.sqlite3 4.70 MB, deleted in 49610ab but still in history
#
# Fixing the second on `main` would mean rewriting published history and
# force-pushing over GitHub. Not worth it: the Space is a deployment target, not
# an archive, and it has no use for the project's history.
#
# WHY THE .npz IS LFS HERE AND NOT ON GITHUB
# ------------------------------------------
# On GitHub it must stay an ordinary blob. ensure_store() rebuilds the demo
# index from it at boot, so a plain `git clone` has to contain the real bytes --
# under LFS, a clone on a machine without git-lfs checks out a pointer file and
# the app starts with no vectors. HF's builder resolves LFS properly, so the
# Space is the one place LFS is safe.
#
# That divergence is the reason this is a script and not a documented sequence
# of commands: the deploy branch is GENERATED, never edited by hand, so the two
# .gitattributes cannot drift.
#
# Usage:  sh scripts/deploy_space.sh
# Needs:  git-lfs installed, and `hf auth login` done once with a Write token.
set -e

BRANCH=space-deploy
REMOTE=space

git diff --quiet || { echo "working tree is dirty - commit or stash first"; exit 1; }
SOURCE=$(git rev-parse --abbrev-ref HEAD)
echo "deploying $SOURCE -> $REMOTE/main"

# Always come back to the source branch, even if the push is rejected. Without
# this, `set -e` exits mid-script and leaves the repo checked out on the orphan
# deploy branch, which is a confusing place to discover you are standing.
cleanup() {
    git checkout -q "$SOURCE" 2>/dev/null || true
    git branch -D "$BRANCH" -q 2>/dev/null || true
}
trap cleanup EXIT

# Orphan branch: no parent, so none of the history HF objects to comes with it.
git checkout --orphan "$BRANCH"
git reset

# Everything tracked at the tip of the source branch, and nothing else.
git checkout "$SOURCE" -- .

# The rules that differ from the source branch. HF's pre-receive hook rejects any
# binary that is not in LFS, and it scans the pushed tree, so this has to cover
# everything binary the repo carries: the vector pack, and the README
# screenshots under docs/images/. The Space does not need the screenshots, but
# HF renders README.md on the Space page and they would be broken images there.
git lfs track "*.npz" "*.png" >/dev/null
git add .gitattributes
git add -A

git commit -q -m "Deploy $SOURCE ($(git rev-parse --short "$SOURCE"))"

# --force because the Space's history is unrelated by construction. Nothing is
# lost: the Space's own initial commit was merged into main in 63256a4, so its
# .gitattributes and frontmatter live on in the real repo.
git push --force "$REMOTE" "$BRANCH:main"

# cleanup() runs on EXIT and returns us to $SOURCE.
echo "deployed. build log: https://huggingface.co/spaces/mohammad778/grounded-forge"
