# Báo cáo Tiến độ Luận văn Thạc sĩ

**Đề tài:** Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
**Tác giả:** Đăng Vĩnh
**Dataset:** MovieLens-1M | **Framework:** Flower (flwr) + PyTorch
**Ngày cập nhật:** 2026-07-27 *(bản trước: 2026-06-15)*

---

## 1. Tóm tắt cho người đọc nhanh (Executive Summary)

Bản cập nhật này bổ sung **ba thí nghiệm mới (07/2026)** lên trên kết quả phủ định quyết định đã báo cáo hồi tháng 6. Trạng thái hiện tại:

1. **Claim gốc của luận văn vẫn bị falsified** (adaptive không thắng, xếp bét trên sparse users) — không đổi.
2. **Claim dương trung tâm được NÂNG CẤP:** khi so knob-for-knob (cùng loss BPR, cùng embedding-dim), split-learning thắng all-global baseline **+20.9%** NDCG@10 (trước đây +13% nhưng dính confound MSE-vs-BPR — confound này nay đã bị loại bằng thực nghiệm).
3. **PFedRec + final-calibration đạt 0.3447** — số full-population cao nhất toàn ma trận, **vượt cả centralized BPR-MF (0.3332) cùng protocol** → "federated đạt/vượt centralized" là một câu bảo vệ được.
4. **Capstone đề xuất (Adaptive-PFedRec, α ở mức head) đã bị BÁC BỎ ngay ở gate offline** với chi phí 0 GPU: ngay cả oracle-α theo bucket (biết trước đáp án) cũng không thắng nổi pure-local calibrated heads (Δ = −0.0002). Nhưng chính sweep đó cho **counterfactual dương mạnh nhất từ trước đến nay**: thay head cá nhân bằng head dân số làm mất **−9.4 điểm** → per-user personalization ở head là load-bearing.
5. Hạ tầng đánh giá được củng cố thêm: công cụ **re-evaluation offline tái lập bit-exact** lượt eval D-06 của run gốc (trust gate max|Δ| = 0.0), và một bug phân loại validity (hậu quả của đợt purge git) được phát hiện + sửa.

Khung luận văn khuyến nghị giữ nguyên hướng đã đề xuất: **kết quả phủ định trung thực + đóng góp phương pháp + cơ chế**, nay được làm giàu bằng chuỗi counterfactual về "personalization nên đặt ở đâu". Phần thực nghiệm còn thiếu duy nhất mang tính bắt buộc: **ma trận seed + kiểm định ý nghĩa thống kê theo cặp** (dữ liệu per-user cho việc này đã bắt đầu được sinh tự động).

---

## 2. Những gì mới so với bản 2026-06-15

| # | Hạng mục | Kết quả | Chi phí |
|---|---|---|---|
| Exp 1 | **BPR all-global baseline** (run `e81518`, 2026-07-26) | Giết confound MSE-vs-BPR; split-learning +20.9% knob-for-knob | ~15h GPU |
| Exp 2 | **PFedRec + final-calibration** (run `cf72e0`, 2026-07-27) | 0.3447 — comparator bar mới; calibration đóng 45% staleness gap | ~8h GPU |
| Exp 3 | **Capstone α-PFedRec — Stage A offline gate** (2026-07-27) | **NO-GO** theo bands đăng ký trước; + counterfactual α=0 mạnh | **0 GPU, ~5 phút** |
| Tooling | reeval.py PfedrecAdapter + trust gate bit-exact; per-user NDCG dump; sửa bug purge-alias trong eval-validity | Nền cho kiểm định theo cặp (queue #4) | — |

Chiến lược capstone được xây dựng qua một phiên đề xuất (2026-07-04) và một vòng **phản biện đối kháng nội bộ** (2026-07-06, 3 reviewer độc lập + 1 công cụ ngoài) — vòng phản biện này yêu cầu: comparator phải là PFedRec+calibration (→ Exp 2), kỳ vọng trung thực −1..+1 điểm, và **gate oracle-α offline bắt buộc trước khi viết code federation** (→ Exp 3). Gate đã làm đúng nhiệm vụ của nó: chặn ~4 ngày code + ~8h GPU cho một hướng không có tín hiệu.

---

## 3. Evaluation Protocol (không đổi, đã khóa)

Như bản trước: natural partition (cross-device, 1 user = 1 client, N=6040), leave-one-out + 99 negatives, `sampled_ndcg@10` chính, nhóm sparse/medium/dense, C=0.1, 100 rounds, `best_round_restore`, full-population D-06 (6040 users), seed 42. Bổ sung duy nhất: **final-calibration** (1 epoch căn lại head cá nhân cho toàn bộ user sau khi khôi phục vòng tốt nhất) nay được xác lập là **một phần của comparator bar** cho mọi so sánh liên quan đến per-user head (lý do: Exp 2 cho thấy nó cải thiện thật và đều).

---

## 4. Ma trận kết quả cập nhật (Main Results)

Tất cả: N=6040, warm full-population D-06, best-round restore, seed 42. Copy nguyên văn từ `final_metrics.best`.

| Hệ thống | Run ID | Best round | NDCG@10 | NDCG@10 sparse | NDCG@10 medium | NDCG@10 dense | HR@10 | HR@10 sparse |
|---|---|---|---|---|---|---|---|---|
| **PFedRec + calibration** *(MỚI)* | `…083121-cf72e0` | 83 | **0.34471** | **0.42697** | **0.38016** | **0.29354** | **0.61109** | **0.70581** |
| PFedRec (không calibration) | `…071106-ef41ab` | 100 | 0.33518 | 0.40567 | 0.37104 | 0.28697 | 0.59238 | 0.67120 |
| Personalized (split-learning) | `…064423-f18e64` | 33 | 0.22748 | 0.27263 | 0.24509 | 0.20087 | 0.41970 | 0.49197 |
| Adaptive (đóng góp gốc) | `…152226-869730` | 83 | 0.21436 | 0.23546 | 0.22641 | 0.19887 | 0.38675 | 0.40915 |
| Baseline all-global, MSE (dim 128) | `…203751-1bf513` | 25 | 0.20131 | 0.24458 | 0.22171 | 0.17300 | 0.36457 | 0.42645 |
| **Baseline all-global, BPR (dim 64)** *(MỚI)* | `…093704-e81518` | 83 | 0.18817 | 0.24687 | 0.19935 | 0.16292 | 0.32583 | 0.40049 |

Điểm tham chiếu ngoài-federated: **centralized BPR-MF cùng LOO-99 protocol = NDCG@10 0.3332** → PFedRec federated (cả hai biến thể) **đạt/vượt centralized**. (SVD/NCF centralized là RMSE-only, không so được trên protocol này.)

**Cặp so sánh knob-for-knob duy nhất trong ma trận** — hai run chỉ khác đúng một trục (user embedding bị FedAvg vs giữ local): Baseline-BPR (0.18817) vs Personalized (0.22748) = **+20.9% overall, +10.4% sparse, +28.8% HR@10**. Đây nay là ô nhân-quả sạch nhất của luận văn.

---

## 5. Ba thí nghiệm tháng 7 — chi tiết

### 5.1. Exp 1 — BPR all-global baseline: loại confound MSE-vs-BPR

**Vấn đề:** claim "+13% split-learning vs baseline" cũ so personalized (BPR, dim 64) với baseline (MSE/BasicMF, dim 128) — khác cả loss lẫn capacity, không quy được nhân quả.
**Thí nghiệm:** chạy baseline all-global với đúng cấu hình của personalized (BPR, dim 64, lr 0.005, FedAvg, 100 rounds, C=0.1, seed 42), chỉ khác duy nhất chỗ user embedding bị trung bình hóa. Run `20260725-093704-e81518`, sidecar valid, đủ 6040 users.
**Kết quả:** 0.18817 — split-learning thắng knob-for-knob **+20.9%**. Confound chết theo hướng CÓ LỢI: claim cũ hóa ra còn khiêm tốn.
**Ghi chú trung thực:** all-global BPR (0.18817) < all-global MSE (0.20131) *gợi ý* FedAvg phá user embedding của BPR nặng hơn, nhưng cặp này còn lệch dim (64 vs 128) — không trích dẫn như hiệu ứng thuần của loss. Verdict về adaptive không đổi: sparse của adaptive (0.23546) vẫn thua cả baseline yếu hơn này (0.24687).

### 5.2. Exp 2 — PFedRec + final-calibration: xác lập comparator bar 0.3447

**Vấn đề:** trước capstone, phải đo xem cơ chế "reconstruction" đơn giản (calibration 1 epoch toàn bộ user sau best-round restore, item embeddings đóng băng) đóng được bao nhiêu phần staleness gap (full-pop 0.3352 vs trần in-loop 0.3563, gap 0.0211).
**Kết quả (run `20260726-083121-cf72e0`, ~8h):** mọi gate đạt (`d06_cache_misses=0` — bằng chứng warm trực tiếp; 6040 users; quỹ đạo in-loop tái lập run cũ với lệch 0.0001). Full-pop best = **0.34471** → calibration đóng **45.2% gap** (band PARTIAL đăng ký trước). Sparse hưởng lợi nhiều nhất (+2.1 điểm → 0.42697).
**Hệ quả:** (a) comparator trung thực cho mọi cải tiến head-level từ nay là **0.3447**, không phải 0.3352; (b) headroom còn lại đến trần in-loop chỉ ~0.0115; (c) federated vượt centralized BPR-MF (0.3332).

### 5.3. Exp 3 — Capstone α-PFedRec: gate offline trả NO-GO

**Đề xuất capstone** (07-04, sau phản biện 07-06): client upload head cá nhân `affine_output.weight` (32 số) → server FedAvg thành `global_head` → khi chấm điểm blend `w = α·w_local + (1−α)·w_global`, α per-user. Kế hoạch 3 giai đoạn với **gate bắt buộc**: Stage A đo bằng oracle offline xem hướng này CÓ THỂ thắng hay không, trước khi viết plumbing federation (Stage B) và chạy GPU (Stage C).

**Stage A đã thực thi** (commit `3ce1367`): viết adapter tái lập offline lượt eval D-06 của run cf72e0. **Trust gate: bản replay khớp BIT-EXACT** khối `final_metrics.best` của run (max|Δ| = 0.0, đủ 6040/809/2322/2909) — mọi số của sweep phía sau kế thừa độ tin cậy này. Sau đó sweep α ∈ {0.00, 0.05, …, 1.00} trên toàn dân số + oracle theo bucket (cross-fitted: α* chọn trên nửa chẵn áp lên nửa lẻ và ngược lại; 3 họ bucket: nhóm mật độ / decile số tương tác / staleness) + policy heuristic HC port từ module adaptive.

**Kết quả — NO-GO theo bands đăng ký trước:**

| Đại lượng | Giá trị | Ngưỡng GO |
|---|---|---|
| Δ_gate (oracle bucket cross-fitted, họ tốt nhất, vs cached 0.34471) | **−0.0002** | ≥ +0.010 |
| α* toàn cục trên grid | **1.0** (cả 3 họ bucket; các bucket 0.7–1.0, tập trung 0.9–1.0) | — |
| Δ sparse (crossfit) | −0.0015 (blend làm hại đúng nhóm yếu) | ≥ +0.015 |
| Policy heuristic HC | 0.30612 = **−3.9 điểm** (đúng như phản biện 07-06 dự đoán) | — |
| Oracle per-user (upper bound, không gate) | +0.0712 — nhưng 74% user chọn α=1.0 → thuần noise-fitting trên 1 item held-out | — |

**Kết luận:** hướng "pha population head vào head cá nhân" không chứa tín hiệu khai thác được ở granularity bucket, dưới protocol đã khóa (sau calibration). Stage B/C không khởi công — tiết kiệm ~4 ngày code + ~8h GPU.

**Nhưng cùng sweep đó cho chuỗi counterfactual giá trị cao cho luận văn** (tất cả bit-paired với D-06, đã commit thành `reeval_*.json`):

| Arm | NDCG@10 | Ý nghĩa |
|---|---|---|
| Head cá nhân (α=1) | 0.34471 | ground truth |
| **Head dân số (α=0, mean 6040 head)** | **0.25099** | mất **−9.4 điểm** → **per-user head là load-bearing** — counterfactual "trong-PFedRec" mà phản biện yêu cầu, trả lời TÍCH CỰC cho câu hỏi "personalization đặt ở head có giá trị không" |
| Head của user khác (shuffled) | 0.19733 | head thực sự cá nhân hóa, không phải generic |
| Head Kaiming ngẫu nhiên | 0.07412 | sàn |

Câu chuyện gọn cho luận văn: *có head riêng cho từng user đáng giá +9.4 điểm; nội suy α giữa head riêng và head dân số không thêm được gì — giá trị nằm ở VIỆC CÓ head riêng, không phải ở việc PHA nó.*

---

## 6. Bằng chứng tính hợp lệ (Trust Gates) — cập nhật

- Corpus hiện có **7 run valid** (thêm `e81518`, `cf72e0`); 14 invalid_cold_eval; 19 diagnostics_only.
- Hai run mới đều mang bằng chứng warm mạnh: `cf72e0` có **`d06_cache_misses=0` stamped**; `e81518` miễn nhiễm cấu trúc (all-global).
- **Nâng cấp lớn:** trust gate của công cụ re-evaluation offline nay được code tường minh (trước chỉ là quy ước) — bản replay `cached` phải khớp `final_metrics.best` với |Δ| < 1e-6 và đúng số user từng nhóm, nếu không toàn bộ phiên phân tích bị hủy (exit 3). Trên cf72e0: **khớp bit-exact (max|Δ| = 0.0)**.
- **Bug hạ tầng được phát hiện + sửa:** `FIX_COMMITS` trong eval-validity giữ SHA trước đợt purge git (15/06) nên run per-user-state đầu tiên sau purge bị phân loại nhầm `invalid_cold_eval`. Fix: mỗi module giữ cặp SHA alias (trước + sau purge), hợp lệ nếu bất kỳ alias nào là ancestor. Cả 7 run valid phân loại đúng (commit `4b8d059`).

---

## 7. Giới hạn & Confounds — cập nhật

1. **Single seed (42) — vẫn là khoảng trống lớn nhất.** Chưa có khoảng tin cậy cho các ô sát nhau (personalized/adaptive/2×baseline). Đã có nền cho hướng rẻ hơn ma trận seed thuần: **per-user NDCG dump** (6040 dòng/hệ thống) cho phép kiểm định theo cặp Wilcoxon signed-rank — dump cho ô PFedRec đã sinh và commit; cần bổ sung cho các module khác.
2. **Dim confound mới (ghi nhận chủ động):** cặp baseline MSE (dim 128) vs baseline BPR (dim 64) lệch capacity — không dùng để claim hiệu ứng loss thuần. Cặp knob-for-knob hợp lệ duy nhất là BPR-baseline vs personalized.
3. Các giới hạn cũ (calibration non-uniformity của run adaptive; `contrastive_lambda=0.1` còn bật trong run adaptive; khác biệt architecture giữa các họ) — không đổi, xem bản 2026-06-15.
4. **Giới hạn của kết luận capstone:** NO-GO đo trên head ĐÃ calibrate (staleness đã bị xóa khỏi cache). Blend theo staleness trên run KHÔNG calibration về lý thuyết còn bỏ ngỏ — nhưng vì comparator trung thực nay bao gồm calibration (rẻ, hiệu quả), hướng đó khó có ý nghĩa triển khai.

---

## 8. Đề xuất bước tiếp theo / Điểm cần quyết định

Câu hỏi Path A/B của bản trước đã được giải quyết một phần **bằng dữ liệu**: biến thể Path B tham vọng nhất (capstone α-PFedRec) đã được thử và bác bỏ với chi phí ~0. Khuyến nghị hiện tại:

**Chốt narrative luận văn** = Path A làm giàu:
1. **Đóng góp phương pháp:** lớp bug cold-eval (sụt ~5×, 3/4 module) + protocol migration + trust-gate hạ tầng (replay bit-exact) — bài học chuyển giao được.
2. **Kết quả dương:** split-learning +20.9% knob-for-knob (Exp 1); PFedRec+calibration 0.3447 ≥ centralized 0.3332; calibration đóng 45% staleness gap (Exp 2).
3. **Chuỗi placement/counterfactual:** per-user head đáng giá +9.4 điểm so với head dân số; head thực sự cá nhân hóa (shuffled −14.7 điểm); nhưng α-blend không thêm giá trị (Exp 3) — một khảo sát "personalization nên đặt ở đâu" có số liệu ở mọi điểm.
4. **Kết quả phủ định trung thực:** adaptive gốc (đã báo cáo) + capstone α-blend (mới) — cả hai đều có phân tích nguyên nhân.

**Việc còn lại trước khi viết:**
- **(Bắt buộc)** Ma trận ý nghĩa thống kê: per-user dump cho 4 hệ thống chính + kiểm định theo cặp; nếu GPU cho phép, +2 seed cho 3 ô sát nhau. (~2–4 ngày GPU tùy mức.)
- **(Bắt buộc, 0 GPU)** Digest 6 paper prior-art còn thiếu (FedRecon, FedTHE, FedRAP, APFL, FedRep, GPFedRec) — FedTHE là prior art sát capstone nhất, PHẢI được cite khi kể chuyện α-blend.
- **(Bắt buộc, 0 GPU)** Mục threat-model + bảng lượng truyền per-method cho chữ "Privacy-Aware" trong tên đề tài.
- (Tùy chọn) Ablation cơ chế split-learning (frozen/random p_u) nếu muốn nâng "giả thuyết cơ chế" thành "kết quả cơ chế".

**Câu hỏi cho giáo sư:** (1) đồng ý chốt narrative trên? (2) mức đầu tư cho ý nghĩa thống kê: chỉ paired-tests trên seed 42, hay +2 seed cho các ô sát nhau? (3) thủ tục đổi tên đề tài (nếu narrative đổi) cần lead time bao lâu?

---

*Nguồn số liệu: `final_metrics.best` trong `results.json` của từng run tại `results/federated/<module>/<run_id>/`; các arm counterfactual từ `reeval_*.json` cùng thư mục (schema v2, kèm trust-gate verdict); centralized từ `results/centralized/bpr_mf_centralized_results.json`. Mọi con số trong báo cáo đã được kiểm chứng lại trực tiếp với JSON gốc ngày 2026-07-27. Bản diễn giải đầy đủ cho người ngoài chuyên ngành: `THESIS_REPORT_EXPLAINED_VI.md`.*
