---
name: advisor-report
description: "How to write Vietnamese documents for the thesis advisor (GVHD) and committee: the one-page Friday report, status summaries, replies to advisor feedback, and thesis sections (methods with formulas, results tables, limitations, future work). Covers structure, terminology, number formatting and traceability, what to leave out (infrastructure), and how to present refuted predictions and open questions. Use when the user says báo cáo cho cô, báo cáo thứ Sáu, tóm tắt tình hình, cô gửi góp ý / feedback (phân tích và trả lời), viết mục luận văn, mục hạn chế, chương kết quả / phương pháp, or asks for anything the advisor will read. Not for experiment READMEs (experiment-preregistration) or paper digests (literature-mapping)."
---

# Writing for the advisor

The advisor reads tables first and asks "số này ở đâu ra" second. The 2026-10-28 draft was stopped because it described the three-machine workflow instead of the methods; the Friday reports exist because the advisor asked for a one-page cadence. Write accordingly.

## 1. Non-negotiables
- Vietnamese, full diacritics, never ASCII substitutes. Code identifiers, run ids, metric names (`NDCG@10`, `HR@10`, `MRR`), file names and flags unchanged.
- Tables of numbers before prose; each table has run ids and the file the numbers come from (in a footnote or a "nguồn" column), never only in your notes.
- Methods explained completely: prediction function, what is trained where, what is transmitted per round (count), loss, evaluation protocol. Formulas inline (`q ← α·q_global + (1−α)·q_user`).
- No infrastructure: no ssh, workstation, W&B, tmux, branches, PR numbers. Compute appears as cost only ("≈ 5 phút CPU cho 6040 user", "≈ 42 giờ GPU").
- Only audited numbers. Missing → "—" and a "số còn thiếu" list. Never estimate, never carry a number from chat.
- Negative or refuted results are stated as plainly as positive ones, with the registered prediction next to the outcome.
- "vượt" / "tốt hơn" only with a paired test (p, CI); otherwise "cao hơn về trung bình".

## 2. Terminology (keep consistent)
| Vietnamese | Meaning |
|---|---|
| GVHD / cô | the advisor |
| đăng ký trước | pre-registration |
| ô, lưới | cell, grid |
| mốc | baseline / reference number |
| bảng phim chung / bảng phim riêng | shared / private item table |
| head riêng, vector người dùng giữ tại máy | private head, local user vector |
| hiệu chuẩn (cuối) | (final) calibration |
| chấm nguội | cold evaluation |
| post-tuning | leave in English (advisor's term) |
| tập mẫu âm sạch / theo code gốc (rò rỉ) | clean / official (leaky) negative pool |
| vòng tốt nhất, vòng cuối | best round, last round |
| tỉ lệ tham gia C | participation ratio |
| nhóm sparse / medium / dense | user groups (keep English labels, counts 809 / 2322 / 2909) |
| kiểm định theo cặp Wilcoxon, p một phía, hiệu chỉnh Holm, khoảng tin cậy bootstrap | statistics terms |

## 3. Number formatting
Metrics 4 decimals (0.3447); deltas with sign (+0.0342) and, when in points, "×100 = +3.4 điểm"; p-values `p = 2.9e−219` when < 1e−3, otherwise 3 significant digits; counts as integers; sha256 as first 8 chars + "…". One source per number: `results.json` (`final_metrics.best`), `reeval_*.json`, `stats_wilcoxon.json`, a filled `RESULT.md`.

## 4. Document types

### 4.1 Friday one-page report (`báo cáo thứ Sáu`)
1. **Tuần này đã làm** — table: item · trạng thái · số liệu chính · nguồn.
2. **Bảng số** — the one table that matters this week (cells × groups).
3. **Đối chiếu với dự đoán đăng ký trước** — prediction · result · Đúng/Sai/Không phân biệt.
4. **Câu hỏi cần cô quyết** — numbered; each with 2–3 options and the consequence of each; the recommended option first.
5. **Tuần sau** — mapped to the four contributions (1 vị trí cá nhân hoá, 2 Adaptive kết quả phủ định, 3 riêng tư có số đo, 4 đánh giá hợp lệ cho client có trạng thái).
6. **Rủi ro / hạn chế mới phát hiện** — one line each.
Target ≈ 1 page: ≤ 2 tables, ≤ 6 questions, no method re-explanation unless new.

### 4.2 Status summary (`tóm tắt tình hình`)
Modules × runs table (run id, protocol pins, NDCG/HR overall + sparse, validity), then per-method description (4 bullets: prediction function, local/global split, transmitted per round, loss), then open issues as a table with owner (em / cô) and date. The current summary lives in a Claude Doc (see memory `thesis-status-doc`); edit it, do not start a new one.

### 4.3 Reply to advisor feedback
Quote each advisor point, then: `Đồng ý và đã làm` / `Đồng ý, sẽ làm (hạn)` / `Xin thảo luận` with a one-line reason and the evidence file. Keep the advisor's own wording for decisions they made.

### 4.4 Thesis section
Methods: one subsection per method with the four bullets above as prose + formula; a table of placement (what is local, what is global, bytes per round). Results: table per claim with run id and validity; paired statistics where per-user dumps exist, "no paired test" labelled otherwise. Limitations: name each cause with its measured magnitude and source (e.g. post-tuning leak +0.061, p < 1e−100; best-round inflation 1.8–5.5 điểm). Future work: withdrawn items (private item table Q_u) with the reason they were withdrawn.

## 5. Before delivery
Run `numbers-auditor` and `methodology-critic` (angles `advisor-alignment`, `writing`) over the draft; fix only the lines named; the main session publishes to the advisor's Claude Doc or hands over the Markdown.
