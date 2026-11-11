---
name: fl-thesis-orchestrator
description: "Coordinates the thesis research team (experiment-registrar, methodology-critic, numbers-auditor, tooling-engineer, results-analyst, thesis-writer, literature-scout) for the federated-recommendation thesis. Use for multi-step research work: pre-registering and reviewing an experiment (đăng ký trước, phản biện README), implementing analysis tooling for a registered experiment, analysing results that came back from a run (phân tích kết quả, điền RESULT, so sánh các module / các run có kiểm định theo cặp), writing or revising advisor-facing documents (báo cáo thứ Sáu, tóm tắt cho cô, phân tích và trả lời góp ý / feedback của cô, mục luận văn, mục hạn chế, chương kết quả), related-work mapping, and follow-ups such as re-run the review, update the report, redo only the statistics, improve the previous result, chạy lại phản biện, cập nhật báo cáo. Not for launching a training run (train-remote), W&B sweeps (sweep), or one-line questions that a file read answers."
---

# FL thesis orchestrator

## Execution mode: mixed

| Track | Trigger (examples) | Mode | Why |
|---|---|---|---|
| A. Register + review | "đăng ký trước", "viết README cho thí nghiệm", "review README này" | named agent `registrar` + **Workflow** review | the README needs rounds of feedback; the review has a fixed list of angles and a per-finding refutation step |
| B. Tooling | "cài biến thể … trong reeval.py", "thêm adapter", "viết test", "wrapper cho …" | subagent `tooling-engineer` → tests → `/code-review` (main) → critic if protocol-affecting | one implementer, mechanical verification |
| C. Run handoff | "chạy trên máy train", "handoff", "bàn giao" | main session uses the `train-remote` skill | interactive with the user; no agent |
| D. Results | "kết quả về rồi", "điền RESULT", "phân tích kết quả", "so sánh các run" | subagent `results-analyst` ∥ then `numbers-auditor` | registered rules; one pass each |
| E. Advisor documents | "báo cáo thứ Sáu", "tóm tắt cho cô", "trả lời góp ý", "viết mục …" | named agent `writer` + parallel reviewers (`numbers-auditor`, `methodology-critic`) → revise via SendMessage (≤ 2 rounds) | drafts need targeted revisions that keep context |
| F. Literature | "digest", "tìm bài", "related work cho đóng góp N" | subagent `literature-scout`; Workflow `pipeline` over papers when > 3 | independent items |

Invoking this skill counts as consent to use the Workflow tool for track A's review. Default scale: 4 angles, 1 refuter per finding; "kỹ" / "toàn bộ" / "thorough" → 6 angles, 3 refuters (majority rule).

## Agents

| `subagent_type` / `agentType` | Model | Skill it loads | Writes |
|---|---|---|---|
| `experiment-registrar` | opus | experiment-preregistration | README + template in the results folder |
| `methodology-critic` | opus | methodology-review | nothing (returns findings) |
| `numbers-auditor` | sonnet | analyze-results | nothing (returns mismatches) |
| `tooling-engineer` | opus | thesis-tooling | code + tests |
| `results-analyst` | sonnet | analyze-results | filled template, `_workspace` tables |
| `thesis-writer` | opus | advisor-report | Markdown draft in `_workspace` |
| `literature-scout` | sonnet | literature-mapping | digests, index, `_workspace` tables |

## Step 0 — existing work
1. `_workspace/<track>-<slug>/` exists?
   - No → fresh run.
   - Yes and the request edits part of the output ("chỉ sửa mục 3", "chạy lại phản biện") → rerun that stage only; pass previous artefact paths to the agent.
   - Yes and the input changed (new results, new advisor text) → move it to `_workspace/<track>-<slug>_<YYYYMMDD-HHMM>/` and start fresh.
2. Track A review: `_workspace/<...>/run_meta.json` holds the last Workflow `runId`; after editing the README, resume with `resumeFromRunId` so unchanged angles come from cache.
3. Read the memory notes on the advisor plan before any track (the four contributions and the current decisions are there), and `results/federated/<run>/<experiment>/README.md` when the request names an experiment.

## Track A — register, then review
1. `Agent(name: "registrar", subagent_type: "experiment-registrar", prompt: request + run dir + exemplar paths + output path)`; wait for the README and template.
2. Review with the Workflow in `references/review-workflow.md`: `args = {doc, dimensions, verifiers, now}`; dimensions from the `methodology-review` skill's angle list (always include `leakage-and-protocol`, `statistics`, `provenance-and-reproducibility`, `advisor-alignment`). Record `runId` in `run_meta.json`.
3. Save the returned `{confirmed, dropped, numbers}` to `_workspace/<...>/02_review_findings.json`.
4. `SendMessage({to: "registrar"}, "Fix these confirmed findings and number mismatches in place; add one changelog row; return what changed")`. One more review round only if a `blocking` finding was fixed.
5. Report to the user: README path, version, counts (findings / confirmed / fixed), open questions for the advisor. The run itself is track C.

## Track B — tooling
1. `Agent(subagent_type: "tooling-engineer", prompt: README section with commands + files to change + "run the tests and report output")`.
2. Main session: `/code-review` on the diff; if the change touches sampling, scoring, trust checks or statistics, `Agent(subagent_type: "methodology-critic", prompt: diff path + angle "statistics" or "leakage-and-protocol")`.
3. Findings → same `tooling-engineer` call again with the list (new subagent is fine; give it the diff).
4. Commit only when the user asks; branch per CLAUDE.md prefixes.

## Track D — results
1. `Agent(subagent_type: "results-analyst", prompt: run/output dir + README path + template path)`.
2. `Agent(subagent_type: "numbers-auditor", prompt: filled template path)` — run after the analyst returns (it audits the filled file).
3. Mismatches → analyst again (new call, pass the list). Deviations from predictions are reported as-is; never re-interpret.
4. Deliver: the filled template path, verdict per prediction, anything with no cell (the user decides whether the registrar adds it).

## Track E — advisor documents
1. `Agent(name: "writer", subagent_type: "thesis-writer", prompt: document type + sources (filled templates, tables, advisor text) + output path)`.
2. In one message: `Agent(subagent_type: "numbers-auditor", prompt: draft path)` and `Agent(subagent_type: "methodology-critic", prompt: draft path, angles advisor-alignment + writing)`.
3. `SendMessage({to: "writer"}, findings)`; at most two rounds; then freeze: `shasum _workspace/<...>/0N_* > _workspace/<...>/freeze.sha`.
4. Main session publishes: Claude Doc (edit the existing thesis-status doc for summaries) or hand over the Markdown; always Vietnamese; never add infrastructure detail back in.

## Track F — literature
- ≤ 3 papers: one `literature-scout` call. More: `Workflow` with `pipeline(args.papers, p => agent(..., {agentType: 'literature-scout', schema: DIGEST_RESULT}))` where `DIGEST_RESULT = {digest_path, protocol_row, contributions}`; `.filter(Boolean)`; `log()` the count not obtained.

## Data passing
- Files under `_workspace/<track>-<slug>/`, named `NN_<agent>_<artifact>.<ext>` (e.g. `01_registrar_open_questions.md`, `02_review_findings.json`, `03_results-analyst_tables.md`, `04_thesis-writer_friday.md`).
- Final outputs go where the repo expects them (results folder, `Papers/`, code); `_workspace/` stays for audit and is git-ignored.
- Workflow returns are structured (schemas in `references/review-workflow.md`); persistent agents exchange file paths, never pasted tables.

## Error handling

| Situation | Action |
|---|---|
| An `agent()` in the review workflow returns `null` | `.filter(Boolean)` drops it; `log()` the angle lost; the final report lists "angle X not reviewed" |
| Workflow fails mid-way | read the journal for actual returns, fix the stage, `resumeFromRunId` |
| Usage limit, auth expiry, permission denied | do not retry; open the partial artefacts, write what is missing to `_workspace/<...>/MISSING.md`, report with the time the limit lifts; fill gaps only with facts you verified yourself, never with guessed agent judgement |
| Named agent silent | `SendMessage` status check; then a new name (`writer2`) with the artefact paths |
| Heavy inputs absent (laptop) | tooling validates on synthetic fixtures; results track waits for `git pull`; critic marks `uncertain`, not `confirmed` |
| Conflicting numbers between files | keep both with sources in the artefact; raise to the user |
| > half of parallel agents fail | stop and ask the user |

## Test scenarios

### Normal — track A
1. User: "Viết README đăng ký trước cho thí nghiệm đo chi phí head bị cũ trên cf72e0". Step 0: no workspace → `_workspace/A-stale-head/`.
2. `registrar` writes `results/federated/pfedrec/20260726-083121-cf72e0/stale_head/README.md` + `RESULT.md`.
3. Review workflow with 4 angles → e.g. 9 findings, 6 confirmed, 0 number mismatches; saved to `02_review_findings.json`; `run_meta.json` has the runId.
4. `registrar` fixes, changelog v1.1; report lists 2 open questions for the advisor. Expected: README v1.1 and template exist; `_workspace/A-stale-head/` has 01–02 and run_meta.

### Normal — track D + E
1. User: "Kết quả posttune về rồi, điền RESULT và viết báo cáo thứ Sáu". `results-analyst` fills `posttune/RESULT.md`; `numbers-auditor` finds 1 rounding note, 0 mismatches.
2. `writer` drafts `04_thesis-writer_friday.md`; auditor + critic return 2 findings; one SendMessage round; freeze; main publishes. Expected: report ≤ 1 page, all numbers sourced.

### Error — review workflow
1. Angle `statistics` returns `null` (agent died). `.filter(Boolean)` drops it; `log("1 angle lost: statistics")`.
2. Final report says "góc thống kê chưa được phản biện; chạy lại với resumeFromRunId sau". The README is still fixed for the other angles.

### Error — usage limit during track E
1. `writer` stops mid-draft on a usage limit. Do not restart it; read `04_thesis-writer_friday.md` as it is; write `MISSING.md` with the sections absent; tell the user the reset time. Resume with `SendMessage` after the limit lifts.
