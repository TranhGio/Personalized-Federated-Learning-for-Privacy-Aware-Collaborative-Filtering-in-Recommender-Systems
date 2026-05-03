# Tình Hình Hiện Tại — Phase 7: Thesis Evaluation Run

> Tài liệu tiếng Việt mô tả trạng thái hiện tại của dự án luận văn thạc sĩ về **Personalized Federated Learning for Privacy-Aware Collaborative Filtering**.
>
> **Cập nhật:** 2026-04-29
> **Branch:** `feat/try_to_run_the_baseline`
> **Phase:** 7 / 7 (phase cuối cùng của milestone v1.0)

---

## 1. Tổng Quan Dự Án

Đây là dự án luận văn thạc sĩ của **Đặng Vinh** (`vinh.nguyen@zozo.com`) nghiên cứu về **học liên kết cá nhân hóa (personalized federated learning)** cho hệ thống gợi ý phim, sử dụng dataset MovieLens 1M (6,040 người dùng / 3,706 phim / ~1 triệu rating).

### Bốn module được so sánh trong luận văn

| Module | Hướng tiếp cận | Vai trò |
|--------|----------------|---------|
| `federated-baseline-cf/` | Tất cả tham số đều global (FedAvg/FedProx) | **Baseline cận dưới** |
| `federated-pfedrec/` | PFedRec (IJCAI-23): score function local + item embeddings global | **Baseline tham chiếu** (calibration) |
| `federated-personalized-cf/` | Split learning (user embeddings local) | Privacy + cá nhân hóa |
| `federated-adaptive-personalized-cf/` | Hierarchical conditional alpha + dual-level | **Đóng góp luận văn** |

### Tuyên bố cốt lõi (thesis claim)

> Dưới giao thức cross-device đúng (1 user = 1 client, N=6040), phương pháp adaptive/hierarchical-conditional **phải thắng** cả ba baseline trên NDCG@10 — bao gồm cả trên nhóm người dùng thưa (sparse) — đồng thời PFedRec phải tái hiện được kết quả tham chiếu IJCAI-23 trong khoảng ±2 điểm.

Nếu phương pháp adaptive **không** thắng dưới giao thức đúng, thì đóng góp luận văn phải được thiết kế lại. **Tính đúng đắn về phương pháp luận là không thể thương lượng.**

---

## 2. Lộ Trình 7 Phase (Roadmap)

| Phase | Tên | Trạng thái |
|-------|-----|------------|
| 1 | Foundation Contract | ✓ Hoàn thành (2026-04-19) |
| 2 | Baseline Migration | ✓ Hoàn thành (2026-04-19) |
| 3 | Personalized Migration | ✓ Hoàn thành (2026-04-20) |
| 4 | Adaptive Migration & Bug Fixes | ✓ Hoàn thành (2026-04-28) |
| 5 | PFedRec Migration & Reproduction | ✓ Hoàn thành (2026-04-28) |
| 6 | Evaluation & Reporting Harness | ✓ Hoàn thành (2026-04-29) |
| **7** | **Thesis Evaluation Run** | **🟡 Đang thực thi** |

Phase 1-6 đã xây dựng toàn bộ hạ tầng phương pháp luận (cross-device migration, evaluation harness, manifest schema). **Phase 7 là phase chạy thí nghiệm thực tế** để tạo ra các bảng số liệu của luận văn.

---

## 3. Phase 7 — Tình Hình Chi Tiết

### Mục tiêu của Phase 7

Chạy bộ so sánh thesis chuẩn hóa cross-device + các ablation của phương pháp adaptive trên cả 4 module, sau đó xuất các bảng thesis (markdown + CSV) ra `results/federated/_thesis/`.

**Yêu cầu (Requirements):** THS-01, THS-02, THS-03, THS-04, THS-05, THS-06, THS-07.

### Cấu trúc 5 Plan trong Phase 7

| Wave | Plan | Mô tả | Trạng thái |
|------|------|-------|------------|
| 1 | **07-01** | Foundation extensions: `_THESIS_CROSSDEVICE_MAIN` mode profile + manifest schema v3 + `atomic_write_text` + entry trong `scripts/run.py` MODE_NUM_SUPERNODES | ✓ Hoàn thành |
| 2 | **07-02** | Wire 4 server_app.py: thêm `"thesis_crossdevice_main"` vào cả 2 mode-tuple gates + manifest mutation patch + 4 pyproject keys mới | ✓ Hoàn thành |
| 3 | **07-03** | Orchestrator (`scripts/thesis/run_thesis_sweep.py`) — matrix-driven `flwr run` launcher với `THESIS_BASE_OVERRIDES` enforcing D-02 + D-03 | ✓ Hoàn thành |
| 3 | **07-04** | Aggregator (`scripts/thesis/aggregate_results.py`) — đọc per-run results.json, tính mean ± std, áp dụng D-11 win criterion, hard-fail D-20 trên cells thiếu, xuất 6 file output | ✓ Hoàn thành |
| 4 | **07-05** | Task 1: RUNBOOK + UAT docs ✓ | 🟡 Một phần (1/5 task) |
| 4 | **07-05** | Tasks 2-5: 4 cổng kiểm tra (checkpoint:human-verify) cho ~50h chạy GPU thực tế | ⏸ Chờ operator |

### Code-side đã hoàn thành ✓

- **Foundation test suite:** 107 → **139 tests GREEN** (+32 tests mới, không có regression nào)
- **Toàn bộ logic D-02/D-03 enforcement** trong orchestrator: `THESIS_BASE_OVERRIDES` dict đảm bảo mọi cell thesis đều dùng `strategy=fedavg`; cells adaptive thêm `model-type=dual` + `alpha-method=hierarchical_conditional`
- **D-11 win criterion** đã code chính xác: `adaptive.mean - adaptive.std > baseline.mean + baseline.std` (chặt > không phải ≥)
- **D-20 hard-fail** kiểm tra cell thiếu trước khi viết bất kỳ file output nào
- **Schema-v3 backward compat** an toàn cho manifests v1/v2 cũ (default values qua `dict.get`)

### Pending: 4 cổng kiểm tra GPU (~50 giờ wallclock)

Đây là phần **chỉ con người mới có thể chạy** — các cell GPU thực tế trên RTX 5090, được mô tả chi tiết trong `07-thesis-evaluation-run-05-RUNBOOK.md`:

| Gate | Mô tả | Thời gian ước tính |
|------|-------|---------------------|
| **A** | Smoke test: 1 cell + idempotency + D-20 hard-fail demo | ~1.5 giờ GPU |
| **B** | Main matrix: 12 cells (4 modules × 3 seeds = `{42, 1337, 2026}`) | ~19.5 giờ GPU |
| **C** | Ablation matrix: 21 cells (7 ablation knobs × 3 seeds, chỉ adaptive) | ~31.5 giờ GPU |
| **D** | Pre-aggregation gate: kiểm tra 33 manifests đã có trên disk | < 1 phút |
| **E** | Aggregation + verification thesis-claim: chạy aggregator để tạo `results/federated/_thesis/*` | ~5 phút |

**Tại sao không thể auto-approve các cổng này?**

> Vì cổng đại diện cho ~50 giờ tính toán GPU thật. Nếu auto-approve, hệ thống sẽ ghi PASS cho công việc chưa thực thi → làm sai lệch chuỗi attribution của THS-03/THS-04 → vô hiệu hóa tuyên bố luận văn.

---

## 4. Output Cuối Cùng Mong Đợi

Sau khi Gate E hoàn tất, thư mục `results/federated/_thesis/` sẽ chứa **6 file**:

```
results/federated/_thesis/
├── main_comparison.md       # Bảng so sánh chính (4 hàng: baseline, personalized, adaptive, pfedrec†)
├── main_comparison.csv      # Phiên bản CSV
├── ablations.md             # Bảng ablation (7 hàng + main row tham chiếu)
├── ablations.csv            # Phiên bản CSV
├── sparse_slice.md          # View THS-04: chỉ user thưa (sparse), tuyên bố luận văn chính
└── sparse_slice.csv         # Phiên bản CSV
```

**Định dạng cell:** `0.4123 ± 0.0089` (mean ± std, 4 chữ số thập phân — D-24).

**Quy tắc bold winner:** Cell adaptive được in đậm khi thoả D-11 (mean - std > tất cả các baseline khác mean + std).

**Hàng PFedRec:** Có chú thích cuối trang `† dim=32, SGD lr=0.1, BCE, fraction-train=1.0; matches IJCAI-23 reference within ±2 points.` — PFedRec là **calibration reference**, KHÔNG tính vào tuyên bố "adaptive thắng baselines" (D-05).

---

## 5. Các Lựa Chọn Tiếp Theo

| # | Hành động | Khi nào dùng |
|---|-----------|--------------|
| 1 | **Chạy Gate A ngay** (`python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main`) → khoảng 1.5h, sau đó nhập `"Gate A passed run_id=<id>"` để continuation agent tiếp tục Gate B | Đã sẵn sàng dành GPU cho thesis |
| 2 | **Hoãn các cổng GPU**, đi tiếp đến verification trên code-side. Plan 05 sẽ vẫn ở trạng thái "partial"; `07-thesis-evaluation-run-05-UAT.md` sẽ giữ Gate A-E để `/gsd:audit-uat` và `/gsd:verify-work` xử lý sau | Cần làm việc khác trước, sẽ quay lại sau |
| 3 | **Chạy toàn bộ ~50h** (Gate A → B → C → D → E tuần tự) rồi báo cáo khi `_thesis/main_comparison.md` đã tồn tại | Có khoảng thời gian dài không cần máy |
| 4 | **Đọc kỹ runbook trước** (`cat .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md`) | Muốn review chi tiết các bước trước khi chạy |

---

## 6. Lệnh Hữu Ích

```bash
# Xem trạng thái hiện tại của roadmap
/gsd:progress

# Xem chi tiết runbook Gate A-E
cat .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md

# Xem checklist UAT để track tiến độ
cat .planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md

# Chạy full test suite (sau Phase 7 code-side hoàn thành)
cd scripts/foundation && pytest -ra

# Smoke test 1 cell adaptive (~1.5h GPU)
python scripts/thesis/run_thesis_sweep.py --module=adaptive --seed=42 --phase=main

# Demo D-20 hard-fail (chứng minh cơ chế kiểm tra cells thiếu hoạt động)
python scripts/thesis/aggregate_results.py --check-only
# Mong đợi: "[D-20 HARD-FAIL] Missing 32 cells: [list]" + exit code 1

# Chạy main matrix (12 cells, ~19.5h GPU)
python scripts/thesis/run_thesis_sweep.py --phase=main

# Chạy ablation matrix (21 cells, ~31.5h GPU)
python scripts/thesis/run_thesis_sweep.py --phase=ablation

# Aggregate kết quả (chỉ chạy sau khi 33 manifests đã có trên disk)
python scripts/thesis/aggregate_results.py
```

---

## 7. Cấu Trúc Quyết Định Then Chốt (Locked Decisions)

Phase 7 dựa trên **24 quyết định locked (D-01..D-24)** trong `.planning/phases/07-thesis-evaluation-run/07-CONTEXT.md`. Một số điểm quan trọng:

- **D-02:** Adaptive main config = `model-type=dual + alpha-method=hierarchical_conditional` ONLY. Các knob "next-gen" (per-user alpha, item perturbation, contrastive λ) đều OFF trong bảng main; chúng chỉ là ablation cells.
- **D-03:** Chỉ FedAvg cho main comparison (KHÔNG có FedProx). Tuyên bố luận văn về cơ chế cá nhân hóa, không phải về aggregation strategy.
- **D-05/D-06:** PFedRec là **calibration reference**, KHÔNG tính vào "adaptive thắng baselines". PFedRec chỉ chạy với `paper_compat_pfedrec` (trung thành với paper IJCAI-23).
- **D-09:** **3 seeds** = `{42, 1337, 2026}` — XKCD / leet / năm hiện tại.
- **D-11:** Tiêu chí thắng nghiêm ngặt: `adaptive.mean - adaptive.std > baseline.mean + baseline.std` (≥1σ không chồng lấn).
- **D-12:** Nếu adaptive KHÔNG thắng main → ablations trở thành **recovery runs** để tìm variant nào thắng. Nếu vẫn không thắng → **escalate lên cấp luận văn**.
- **D-20:** Aggregator hard-fail nếu thiếu bất kỳ cell nào — KHÔNG có bảng từng phần.

---

## 8. Tiến Độ Tổng Thể

```
Phase 1: ████████████ 100% (6/6 plans)
Phase 2: ████████████ 100% (5/5 plans)
Phase 3: ████████████ 100% (5/5 plans)
Phase 4: ████████████ 100% (6/6 plans)
Phase 5: ████████████ 100% (5/5 plans)
Phase 6: ████████████ 100% (7/7 plans)
Phase 7: ██████████░░  85% (4.2/5 plans — code-side hoàn thành, GPU gates pending)

Tổng dự án: ████████████░ ~97% (38.2 / 39 plans)
```

**Bottleneck duy nhất còn lại:** ~50 giờ GPU wallclock cho Tasks 2-5 của Plan 05.

---

## 9. Tài Liệu Liên Quan

- `CLAUDE.md` — Conventions của dự án (Python 3.9+ typing, snake_case, NumPy-style docstrings, dataclasses, atomic writes)
- `.planning/PROJECT.md` — Core value, requirements, constraints, decision log
- `.planning/STATE.md` — Trạng thái phase hiện tại (luôn được sync sau mỗi commit)
- `.planning/ROADMAP.md` — Lộ trình 7 phase với dependency graph
- `.planning/REQUIREMENTS.md` — Bảng 52 requirements (THS-01..THS-07 cho Phase 7)
- `.planning/phases/07-thesis-evaluation-run/07-CONTEXT.md` — 24 locked decisions (D-01..D-24)
- `.planning/phases/07-thesis-evaluation-run/07-RESEARCH.md` — Research kỹ thuật (10 pitfalls, 4 patterns, validation architecture)
- `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-RUNBOOK.md` — Runbook vận hành cho Gate A-E
- `.planning/phases/07-thesis-evaluation-run/07-thesis-evaluation-run-05-UAT.md` — Checklist pass/fail cho operator

---

*File này được tạo tự động vào 2026-04-29 để mô tả tình hình Phase 7. Cập nhật sau mỗi gate complete bằng cách edit phần "Tiến Độ Tổng Thể" và bảng "Phase 7 — Tình Hình Chi Tiết".*
