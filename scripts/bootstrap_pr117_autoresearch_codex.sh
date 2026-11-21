#!/usr/bin/env bash
# PR117: install the verified autoresearch skills in an isolated Ceph-backed Codex worktree.
# This script does NOT start Codex, training, evaluation, or an GPU job.
set -euo pipefail

ENV_FILE=/aaa/fionafyang/buddy1/whaleywang/MechET/configs/runtime/shared_host_storage.env
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing required shared Ceph environment: $ENV_FILE" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

WORKTREE="$(git rev-parse --show-toplevel)"
case "$WORKTREE" in
  /aaa/fionafyang/buddy1/*) ;;
  *) echo "Refusing installation outside shared Ceph: $WORKTREE" >&2; exit 2 ;;
esac

BASE=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/research_tools/pr117_autoresearch
OUT=/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/diagnostics/retroe_pr117_autoresearch_20261121
mkdir -p "$BASE" "$OUT" "$WORKTREE/.agents/skills"

ORCH="$BASE/orchestra"
MEASURED="$BASE/measured"

if [[ ! -d "$ORCH/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Orchestra-Research/AI-Research-SKILLs.git "$ORCH"
  git -C "$ORCH" sparse-checkout set 0-autoresearch-skill
fi
if [[ ! -d "$MEASURED/.git" ]]; then
  git clone --depth 1 \
    https://github.com/drivelineresearch/autoresearch-claude-code.git "$MEASURED"
fi

OUTER="$ORCH/0-autoresearch-skill/SKILL.md"
INNER="$MEASURED/skills/autoresearch/SKILL.md"
[[ -s "$OUTER" && -s "$INNER" ]] || {
  echo "Skill checkout incomplete; inspect repository files, do not claim skill loaded." >&2
  exit 3
}

LINK="$WORKTREE/.agents/skills/autoresearch"
if [[ -e "$LINK" || -L "$LINK" ]]; then
  if [[ "$(readlink -f "$LINK")" != "$(readlink -f "$MEASURED/skills/autoresearch")" ]]; then
    echo "Existing foreign skill path: $LINK. Refusing to overwrite." >&2
    exit 4
  fi
else
  ln -s "$MEASURED/skills/autoresearch" "$LINK"
fi

python - "$OUT" "$ORCH" "$MEASURED" "$OUTER" "$INNER" "$LINK" <<'PY'
import hashlib, json, pathlib, subprocess, sys, time
out, orch, measured, outer, inner, link = map(pathlib.Path, sys.argv[1:])
def git_rev(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "created_unix": time.time(),
    "install_status": "FILES_VERIFIED",
    "codex_skill_loaded": False,
    "codex_skill_loaded_note": "Codex must explicitly read BOTH complete SKILL.md files and verify active discovery before setting true.",
    "outer": {"repo": "Orchestra-Research/AI-Research-SKILLs", "git_sha": git_rev(orch), "file": str(outer), "sha256": sha(outer)},
    "inner": {"repo": "drivelineresearch/autoresearch-claude-code", "git_sha": git_rev(measured), "file": str(inner), "sha256": sha(inner)},
    "project_local_skill": str(link),
    "critical_limits": ["NO occupied 8-H20 reuse", "NO unbounded autoresearch", "NO extra LLM", "NO more than 0.8B actor tonight"],
}
dest = out / "autoresearch_skill_receipt.json"
dest.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Skill files verified. Receipt: ", dest)
print("Outer method: ", outer)
print("Inner Codex skill: ", inner)
print("NEXT: open and read both complete SKILL.md files inside Codex; verify discovery, then mark loaded in receipt.")
PY
