#!/usr/bin/env bash
# Workstation-side launcher (the training machine runs NO Claude — this script is the whole interface).
#
# Usage (from anywhere inside the repo checkout on the workstation):
#   scripts/remote/start_run.sh <wrapper.sh> [tmux-session-name] [--dry-run]
#
# <wrapper.sh> is a launch wrapper COMMITTED from the laptop (e.g.
# .bug-investigation/run_pfedrec_calibration.sh) — never a hand-typed flwr command.
#
# Steps: git pull --ff-only -> guards (wrapper tracked, clean code tree, disk, no live
# federation, tmux name free) -> tmux new -d <name> "bash <wrapper> | tee <log>".
set -euo pipefail

MIN_FREE_GB="${MIN_FREE_GB:-40}"   # adaptive runs write ~14GB of cache; keep headroom

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

DRY_RUN=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage ;;
    *) ARGS+=("$a") ;;
  esac
done
[ "${#ARGS[@]}" -ge 1 ] || usage
WRAPPER="${ARGS[0]}"
SESSION="${ARGS[1]:-$(basename "$WRAPPER" .sh)}"

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
LOG=".bug-investigation/${SESSION}.log"

fail() { printf "[start_run] ABORT: %b\n" "$*" >&2; exit 1; }
ok()   { echo "[start_run] ok: $*"; }

# 1. Sync code (fast-forward only: the workstation never edits code).
BRANCH="$(git branch --show-current)"
if [ "$DRY_RUN" = 1 ]; then
  ok "dry-run: would 'git pull --ff-only' on branch $BRANCH"
else
  git pull --ff-only || fail "git pull --ff-only failed on $BRANCH — someone edited code on the workstation, or the branch diverged. Fix on the laptop, never here."
  ok "pulled $BRANCH @ $(git rev-parse --short HEAD)"
fi

# 2. Wrapper must exist and be committed (reviewed on the laptop).
[ -f "$WRAPPER" ] || fail "wrapper not found: $WRAPPER"
git ls-files --error-unmatch "$WRAPPER" >/dev/null 2>&1 || fail "wrapper is not committed: $WRAPPER (commit + push it from the laptop)"
ok "wrapper tracked: $WRAPPER"

# 3. No uncommitted CODE changes (results/ noise and the upstream submodule are tolerated).
DIRTY="$(git status --porcelain --untracked-files=no -- . ':!results' ':!IJCAI-23-PFedRec' || true)"
[ -z "$DIRTY" ] || fail "uncommitted code changes on the workstation:\n$DIRTY"
ok "code tree clean"

# 4. Disk headroom on the repo filesystem (NVMe, shared, often near-full).
FREE_GB="$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || fail "only ${FREE_GB}GB free (< ${MIN_FREE_GB}GB). Archive finished caches first: scripts/remote/finish_run.sh ... --archive-cache"
ok "disk: ${FREE_GB}GB free"

# 5. One federation at a time on the single GPU.
LIVE="$(pgrep -af 'flwr run|ServerApp|ray::ClientAppActor' | grep -v pgrep || true)"
[ -z "$LIVE" ] || fail "a federation is already running:\n$LIVE"
ok "no live federation"

# 6. tmux session name free.
command -v tmux >/dev/null || fail "tmux not installed"
if tmux has-session -t "$SESSION" 2>/dev/null; then fail "tmux session '$SESSION' already exists"; fi
ok "tmux session name free: $SESSION"

CMD="bash $WRAPPER 2>&1 | tee $LOG"
if [ "$DRY_RUN" = 1 ]; then
  echo "[start_run] DRY-RUN — would run: tmux new -d -s $SESSION \"$CMD\""
  exit 0
fi
tmux new -d -s "$SESSION" "cd $REPO && $CMD"
echo "[start_run] LAUNCHED in tmux '$SESSION' (log: $LOG)"
echo "  watch:  tmux attach -t $SESSION      (detach: Ctrl-b d)"
echo "  tail:   tail -f $REPO/$LOG"
echo "  metrics live on W&B; an email/Slack alert fires when the run finishes."
