#!/usr/bin/env bash
# PFedRec post-tuning re-score, option (a) of the advisor's 2026-11-04 feedback.
# Pre-registered in results/federated/pfedrec/20260726-083121-cf72e0/posttune/README.md (v1, §9).
# Offline, CPU only, NO federation: per-user fine-tune of the item table rows from the
# restored best global (head frozen), 4 cells = {clean, leaky} negatives x E in {1, 3}
# (= reeval variant `qu_diag`), scored on the identical D-06 candidates, then paired
# Wilcoxon vs the 0.3447118 baseline + leaky-vs-clean contrasts.
# Inputs that exist only on this machine: global_state_best.pt (sha256 pinned below) and
# federated-pfedrec/.embedding_cache/20260726-083121-cf72e0/ (6040 heads, may be an HDD symlink).
# Pre-flight on the laptop 2026-11-04: code on master (PR #13), 21 qu/gate tests + reeval
# --self-test PASS, synthetic full-size run ~3 min/pool. Expected wall-clock here: < 30 min.
# Any trust-check failure exits 3 and writes NO cell; this wrapper then stops before step 3.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=/home/bes/miniconda3/bin/python3; [ -x "$PY" ] || PY=python3

RUN=results/federated/pfedrec/20260726-083121-cf72e0
OUT=$RUN/posttune
CACHE=federated-pfedrec/.embedding_cache/20260726-083121-cf72e0
EXPECTED_SHA=279459ba72987a98dfd1065c59ab0bb32f0e6f33323c08ccbc25211103b41c57

fail() { echo "[posttune] ABORT: $*" >&2; exit 1; }
step() { echo; echo "[posttune] == $* ($(date '+%Y-%m-%d %H:%M:%S'))"; }

step "0. pre-flight (README §2, §9)"
[ -f "$RUN/global_state_best.pt" ] || fail "$RUN/global_state_best.pt missing"
SHA="$(sha256sum "$RUN/global_state_best.pt" | cut -d' ' -f1)"
[ "$SHA" = "$EXPECTED_SHA" ] || fail "sha256 of global_state_best.pt = $SHA, expected $EXPECTED_SHA"
[ -f "$CACHE/manifest.json" ] || fail "head cache manifest missing: $CACHE/manifest.json"
N_HEADS="$(ls "$CACHE" | grep -c '^partition_[0-9]*\.pt$')"
[ "$N_HEADS" -eq 6040 ] || fail "head cache has $N_HEADS partition files, expected 6040"
[ -f "$RUN/reeval_cached_best_per_user.csv" ] || fail "baseline CSV missing: $RUN/reeval_cached_best_per_user.csv"
"$PY" -c "import scipy, torch" 2>/dev/null || fail "scipy/torch not importable with $PY (pip install scipy)"
if ls "$OUT"/reeval_qu_diag_best*.json "$OUT"/*.csv "$OUT"/stats_wilcoxon.json >/dev/null 2>&1; then
  fail "outputs already exist in $OUT — a registered run is never overwritten; move them aside first"
fi
echo "[posttune] ok: sha256 pinned, $N_HEADS heads, baseline CSV, python=$PY, code=$(git rev-parse --short HEAD)"

step "1. probe: 200 users, trust gate (tolerant mode), timing only — writes nothing"
"$PY" scripts/thesis/reeval.py --run-dir "$RUN" --module pfedrec --globals best \
  --local-variant qu_diag --device cpu --limit-users 200 --no-json --no-per-user \
  || fail "200-user probe failed (exit $?)"

step "2. full run: 4 cells x 6040 users -> $OUT"
"$PY" scripts/thesis/reeval.py --run-dir "$RUN" --module pfedrec --globals best \
  --local-variant qu_diag --device cpu --out-dir "$OUT" \
  || fail "full run failed (exit $?; 3 = trust gate / trust check FAIL, see $OUT/reeval_qu_diag_best.json)"

step "3. statistics (README §6): per cell vs baseline + leaky-vs-clean contrasts"
"$PY" scripts/thesis/gate_stats.py \
  --grid "$OUT/reeval_qu_diag_best_per_user.csv" \
  --baseline "$RUN/reeval_cached_best_per_user.csv" \
  --grid-json "$OUT/reeval_qu_diag_best.json" \
  --contrast a0.00_E1_leaky a0.00_E1_clean \
  --contrast a0.00_E3_leaky a0.00_E3_clean \
  --out "$OUT/stats_wilcoxon.json" \
  || fail "gate_stats failed (exit $?)"

step "4. record the three small output files (no .pt, no cache) and push"
git add "$OUT/reeval_qu_diag_best.json" "$OUT/reeval_qu_diag_best_per_user.csv" "$OUT/stats_wilcoxon.json" \
  || fail "git add failed"
git commit -q -m "chore(results): PFedRec post-tuning re-score (option a) on cf72e0 — 4 qu_diag cells + stats" \
  || fail "git commit failed (nothing staged?)"
git push -q || fail "git push failed — push manually from this checkout"
echo "[posttune] pushed. On the laptop: git pull, then fill $OUT/RESULT.md from the JSONs."
echo "[posttune] DONE $(date '+%Y-%m-%d %H:%M:%S')"
