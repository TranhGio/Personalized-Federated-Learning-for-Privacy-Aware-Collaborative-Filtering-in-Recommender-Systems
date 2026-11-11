---
name: experiment-preregistration
description: "How to write a binding pre-registration README and its fill-only RESULT.md / DECISION.md template for a thesis experiment (offline re-score, ablation, gate, privacy attack) before any run: fixed inputs with hashes, protocol, grid, statistics, decision or reporting rule, trust checks, numeric predictions, cost, exact commands, changelog; and how to amend or withdraw one afterwards. Use when the user says đăng ký trước, viết README cho thí nghiệm, pre-register, 'before we run', or asks to change a registered README. Not for filling in results (analyze-results) or for reports to the advisor (advisor-report)."
---

# Pre-registering a thesis experiment

Every number the thesis cites comes from an experiment whose rules were written down before it ran. The README is the contract; the template is the only place results go. This is what let the advisor accept a refuted prediction (A5 of the post-tuning re-score) without the experiment being re-interpreted after the fact.

## Exemplars (read one before writing)
- Full form (gate with GO/NO-GO, 14 sections, two review rounds): `results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/README.md` + `DECISION.md`.
- Short form (re-score with a reporting rule, 10 sections): `results/federated/pfedrec/20260726-083121-cf72e0/posttune/README.md` + `RESULT.md` (filled — shows what a template must contain).

## Where it lives
`results/federated/<module>/<run_id>/<experiment>/README.md`, next to the run it re-uses. A new training run is registered as a launch wrapper plus a README in `results/federated/_thesis/` or the module folder; the `train-remote` skill covers the wrapper.

## Mandatory sections (adapt names, keep content)

| # | Section | Must contain |
|---|---|---|
| 0 | Status line | `ĐĂNG KÝ TRƯỚC, CHƯA CHẠY` / `ĐÃ CHẠY` / `ĐÃ RÚT`; version; date; source of the request (advisor text quoted verbatim where it binds) |
| 1 | Question | one question; what the answer is used for (thesis table, limitation, decision) and what it is **not** used for |
| 2 | Fixed inputs | run id, protocol of the run, hyper-parameters, globals file + sha256, per-user cache location + expected file count, candidate set (seed / round_num), baseline numbers with their file, user-group definition with counts, data hashes, code commits, where the heavy inputs physically are |
| 3 | Protocol | step by step per user; seeds (`np_rng/torch_gen(run_seed, pid, round)`); what is frozen; negative pool definition |
| 4 | Grid / cells | fixed list; "no cell added or removed after running"; which cell is the reproduction check |
| 5 | Scoring | same evaluation function as the run; aggregation (sufficient stats, groups) |
| 6 | Statistics | test, pairing unit, sidedness and why, zero handling, multiplicity correction, CI method with resamples and seed, effect size; what is binding vs reported |
| 7 | Decision or reporting rule | thresholds (verbatim from the advisor when they set them) or the fixed table rows that will appear in the thesis; the limitation sentence chosen per outcome |
| 8 | Trust checks | each check, its pass condition, and "fail → stop, write no cell" |
| 9 | Predictions | numbered, with direction, magnitude, margin (0.001 NDCG ≈ 1 SE at 6040 users), the table that decides, and the structural argument behind them |
| 10 | Diagnostics outside the grid | only if needed; "requires advisor approval; never a decision quantity" |
| 11 | Outputs | file names and schemas (column list for CSVs) |
| 12 | Cost | measured where possible (synthetic run at full size), GPU = 0 if CPU |
| 13 | Implementer decisions | table: advisor/official text → our choice → consequence (deviations are declared, not hidden) |
| 14 | Run plan | pre-flight checks, probe run, full run, statistics, fill template — exact commands with paths |
| 15 | Changelog | date, version, what changed, what did not change |

Short experiments can merge 4–5 and 10–12, but no section may be dropped silently; write "không áp dụng" with the reason.

## Rules that make it binding
- Numbers in sections 2 and 9 are computed from files (Python snippet) and cite the file; no chat values.
- Predictions must be falsifiable. Give two scenarios when the mechanism allows it (e.g. "strong push" vs "weak push") and state the observable that distinguishes them. A prediction can be "Không phân biệt" after the run; it cannot be re-worded.
- After the first real run, sections 1–14 are frozen. Any change is a new version row in the changelog with the reason; status line changes to `ĐÃ CHẠY` or `ĐÃ RÚT` ("đã rút theo quyết định của GVHD ngày …" when the advisor drops it).
- Tooling changes required by the README are listed in section 14 as prerequisites with the test that proves them; the README is runnable as written once they land.
- The template (`RESULT.md` / `DECISION.md`) is written in the same commit: header table (dates, commits, sha256, trust gate, trust checks, timing, aborted runs), every result table with "—" cells, the prediction check table, the limitation-sentence checkboxes. Cells not in the template are not reported.
- Aborted or repeated runs are recorded in the template with the reason, even when no cell was produced.

## Advisor language
Vietnamese with full diacritics, "GVHD" / "cô", "tập mẫu âm" (negative pool), "biến thể" (variant), "ô" (cell), "mốc" (baseline), "bảng phim chung / riêng" (shared / private item table), "hiệu chuẩn" (calibration), "chấm nguội" (cold eval), "rò rỉ" (leak), "vòng tốt nhất" (best round). Code identifiers, flags and file names unchanged.

## Review before running
Hand the README to the orchestrator's review track (methodology-critic across angles + numbers-auditor). Fix findings in place and add one changelog row listing them. Only then is the run handed off (`train-remote`).
