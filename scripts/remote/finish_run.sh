#!/usr/bin/env bash
# Workstation-side post-run step: validate, commit ONLY this run's small result files, push.
#
# Usage (inside the repo checkout on the workstation, after the run's tmux session ended):
#   scripts/remote/finish_run.sh <module> <run_id> [--archive-cache] [--no-push] [--dry-run]
#
#   <module>  baseline | personalized | adaptive | pfedrec
#   <run_id>  e.g. 20260726-083121-cf72e0  (results/federated/<module>/<run_id>/)
#
# Only commit runs that will be cited (thesis label / numbers in a report);
# smoke/debug runs: just read W&B and skip this script.
#
# --archive-cache  move the run's per-user cache to the HDD archive
#                  (ARCHIVE_ROOT, default /mnt/nfs-data/vinh_dataset/movie-rec), verify
#                  every file by SHA-256, then replace the original with a symlink.
#                  Only for FINISHED runs — never archive a cache a federation is using.
set -euo pipefail

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/nfs-data/vinh_dataset/movie-rec}"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

DRY_RUN=0; ARCHIVE=0; PUSH=1
ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --archive-cache) ARCHIVE=1 ;;
    --no-push) PUSH=0 ;;
    -h|--help) usage ;;
    *) ARGS+=("$a") ;;
  esac
done
[ "${#ARGS[@]}" -eq 2 ] || usage
MODULE="${ARGS[0]}"; RUN_ID="${ARGS[1]}"

case "$MODULE" in
  baseline)     MODDIR=federated-baseline-cf ;;
  personalized) MODDIR=federated-personalized-cf ;;
  adaptive)     MODDIR=federated-adaptive-personalized-cf ;;
  pfedrec)      MODDIR=federated-pfedrec ;;
  *) echo "unknown module '$MODULE'"; usage ;;
esac

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
RUN_DIR="results/federated/$MODULE/$RUN_ID"

fail() { printf "[finish_run] ABORT: %b\n" "$*" >&2; exit 1; }
ok()   { echo "[finish_run] ok: $*"; }
run()  { if [ "$DRY_RUN" = 1 ]; then echo "[finish_run] DRY-RUN: $*"; else eval "$@"; fi; }

# 1. The run must be finished and its federation gone.
[ -f "$RUN_DIR/results.json" ] || fail "$RUN_DIR/results.json not found — run not finished (or wrong module/run_id)"
[ -f "$RUN_DIR/manifest.json" ] || fail "$RUN_DIR/manifest.json not found"
LIVE="$(pgrep -af 'flwr run|ServerApp|ray::ClientAppActor' | grep -v pgrep || true)"
[ -z "$LIVE" ] || fail "a federation is still running — wait for it to exit:\n$LIVE"
ok "results present: $RUN_DIR"

# 2. Headline + trust signals straight from results.json.
python3 - "$RUN_DIR/results.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
b = (r.get("final_metrics") or {}).get("best") or {}
f = lambda k: ("%.4f" % b[k]) if isinstance(b.get(k), (int, float)) else "n/a"
print(f"[finish_run] NDCG@10 {f('sampled_ndcg@10')} (sparse {f('sampled_ndcg@10/sparse')}) "
      f"HR@10 {f('sampled_hr@10')} | evaluated_users {b.get('evaluated_users')} "
      f"| d06_cache_misses {b.get('d06_cache_misses', 'not stamped')}")
PY

# 3. Validity sidecar (the script reclassifies every run; we keep only this run's files).
run "python3 scripts/thesis/eval_validity.py >/dev/null"
if [ "$DRY_RUN" = 0 ]; then
  STATUS="$(python3 -c "import json;print(json.load(open('$RUN_DIR/EVAL_VALIDITY.json'))['status'])")"
  echo "[finish_run] EVAL_VALIDITY status: $STATUS"
  [ "$STATUS" = "valid" ] || echo "[finish_run] WARNING: not 'valid' — do NOT cite this run's best block until investigated."
fi

# 4. Commit exactly this run's small files; revert the timestamp churn on other sidecars.
FILES="$RUN_DIR/results.json $RUN_DIR/manifest.json $RUN_DIR/EVAL_VALIDITY.json"
[ -f "$RUN_DIR/INVALID_COLD_EVAL.md" ] && FILES="$FILES $RUN_DIR/INVALID_COLD_EVAL.md"
for extra in "$RUN_DIR"/per_user_metrics.json "$RUN_DIR"/reeval_*.json; do
  [ -f "$extra" ] && FILES="$FILES $extra"
done
run "git add $FILES"
run "git checkout -- results/ 2>/dev/null || true"   # unstaged churn on OTHER runs' sidecars
if [ "$DRY_RUN" = 0 ] && git diff --cached --quiet; then
  ok "nothing new to commit (already recorded)"
else
  run "git commit -q -m 'chore(results): record $MODULE run $RUN_ID'"
  ok "committed: $FILES"
fi
if [ "$PUSH" = 1 ]; then
  run "git push -q"
  ok "pushed — on the laptop: git pull"
fi

# 5. Optional: archive the per-user cache to the HDD (verify-before-delete, then symlink).
if [ "$ARCHIVE" = 1 ]; then
  SRC="$MODDIR/.embedding_cache/$RUN_ID"
  DST="$ARCHIVE_ROOT/embedding_cache/$MODULE/$RUN_ID"
  if [ -L "$SRC" ]; then ok "cache already archived (symlink): $SRC"
  elif [ ! -d "$SRC" ]; then ok "no per-user cache for this run ($SRC) — nothing to archive"
  else
    [ -d "$ARCHIVE_ROOT" ] || fail "archive root not found: $ARCHIVE_ROOT"
    run "mkdir -p '$(dirname "$DST")' && rsync -a '$SRC/' '$DST/'"
    if [ "$DRY_RUN" = 0 ]; then
      A="$(cd "$SRC" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)"
      B="$(cd "$DST" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)"
      [ "$A" = "$B" ] || fail "checksum mismatch between $SRC and $DST — original KEPT, investigate"
      ok "per-file SHA-256 identical ($(find "$DST" -type f | wc -l) files)"
    fi
    run "rm -rf '$SRC' && ln -s '$DST' '$SRC'"
    ok "cache archived: $SRC -> $DST"
  fi
fi
echo "[finish_run] done."
