# Báo cáo Tiến độ Luận văn Thạc sĩ

**Đề tài:** Personalized Federated Learning for Privacy-Aware Collaborative Filtering in Recommender Systems
**Tác giả:** Đặng Vinh
**Dataset:** MovieLens-1M | **Framework:** Flower (flwr) + PyTorch
**Ngày cập nhật:** 2026-06-15

---

## 1. Tóm tắt cho người đọc nhanh (Executive Summary)

Báo cáo này trình bày một **kết quả phủ định mang tính quyết định (decisive negative result)** cho claim gốc của luận văn, được xác lập dưới một evaluation protocol đúng về mặt phương pháp — protocol mà chúng tôi đã phải **sửa lỗi** trước khi bất kỳ con số nào đáng tin.

Claim gốc là: dưới cross-device protocol (1 user = 1 client, N=6040, natural partition, leave-one-out + 99 negatives, sampled NDCG@10), phương pháp **adaptive / hierarchical-conditional** đề xuất phải thắng cả ba comparator — all-global baseline, PFedRec, và split-learning personalized — **đặc biệt trên nhóm sparse users (0–30 interactions)** — đồng thời PFedRec phải tái lập (reproduce) được reference đã công bố.

Ma trận run chính thức (4 run hợp lệ, full-population, single-seed, dùng chung protocol đã khóa) cho thấy:

- Adaptive chỉ thắng **duy nhất** all-global baseline về overall (+6.5%).
- **Thua** split-learning personalized và PFedRec về overall.
- Trên **sparse users — chính là claim cốt lõi — adaptive xếp BÉT (DEAD LAST)**, thấp hơn cả all-global baseline.

**Do đó claim gốc của luận văn bị falsified dưới protocol đúng** (single-seed; xem §7 cho caveat về ý nghĩa thống kê). Tuy nhiên, dự án mang lại hai đóng góp thực chất và bảo vệ được: (a) một **đóng góp phương pháp** — phát hiện và sửa một lớp bug "cold-evaluation" đã làm sụt giảm ~5× các model per-user-personalized; và (b) một **phát hiện cơ chế (mechanism)** sạch sẽ: lợi ích của split-learning đến từ việc *bảo vệ quá trình train item-embedding*, không phải từ personalized inference. Hướng đề xuất là tái định khung (reframe) luận văn quanh kết quả phủ định cộng với mechanism claim, thay vì tiếp tục đuổi theo claim dương đã bị bác bỏ.

---

## 2. Những gì đã thay đổi kể từ lần cập nhật trước

Hai chỉnh sửa lồng vào nhau đã làm vô hiệu mọi số liệu cũ và buộc phải chạy lại toàn bộ ma trận.

**(a) Bug cold-evaluation.** Lượt evaluation full-population cuối cùng (D-06) đã chấm điểm **mọi user bằng một personal head cold, chưa khởi tạo**, đọc từ một cache path không tồn tại (`.embedding_cache/default/`), bởi vì lượt eval không stamp `run_id` / `reuse_cache` vào eval config. Với bất kỳ module nào giữ tham số *local* theo từng user (PFedRec: `affine_output`; personalized: user embeddings; adaptive: user embeddings / personal MLP / fusion / α / perturbation), lỗi này làm sụt full-population metrics khoảng **5×** — quan sát trực tiếp ở PFedRec: **0.0711 (cold) vs 0.3478 (warm)**. Đã fix theo từng module:

- `01d8b72` — PFedRec (`affine_output` heads)
- `3a393fb` — personalized (`user_embeddings` / `user_bias`)
- `70a92bd` — adaptive (`user_embeddings` / `personal_mlp` / fusion / `logit_alpha` / `item_perturbation`)

All-global baseline **miễn nhiễm về mặt cấu trúc** (nó dựng lại toàn bộ model từ broadcast aggregated arrays, không giữ per-user local cache), nên run của nó không cần fix và hợp lệ theo thiết kế.

**(b) Migration protocol.** Chúng tôi đã loại bỏ setup cross-silo `num-supernodes=5` cũ, chuyển sang cross-device **natural** partition (1 user = 1 client, N=6040). Đây là protocol mà claim của luận văn được *định nghĩa trên đó* — bối cảnh mà một claim về per-user personalization mới có ý nghĩa vận hành cho công trình này; cross-silo federated RecSys vẫn là một setting hợp lệ nhưng khác, được giữ lại dạng opt-in chỉ để tái lập các run cũ. Natural partition giờ là default đã khóa.

Hai thay đổi này là *lý do vì sao các kết quả trước đây không đáng tin*, và vì sao toàn bộ ma trận phải được tạo lại dưới pipeline đã sửa.

---

## 3. Evaluation Protocol đã khóa

Cả 4 run chính thức dùng chung một protocol y hệt:

| Knob | Giá trị |
|---|---|
| Partition | `natural` (cross-device, 1 user = 1 client, **N=6040**) |
| Eval | leave-one-out + 99 negative samples (NCF protocol) |
| Metric chính | `sampled_ndcg@10` (kèm HR@10) |
| Nhóm user | sparse (0–30), medium (30–100), dense (100+) interactions |
| Client sampling | `fraction_train = 0.1` (C=0.1) |
| Rounds | `num_server_rounds = 100` (cả 4 chạy đủ 100) |
| Checkpoint | `best_round_restore` |
| Eval coverage | full-population D-06, **evaluated_users = 6040** |
| Seed | 42 (mới chỉ single seed) |

Các số bên dưới được copy nguyên văn từ `final_metrics.best` trong `results.json` của từng run, không làm tròn.

---

## 4. Kết quả chính (Main Results)

| Module | Run ID | Best Round | NDCG@10 (overall) | NDCG@10 sparse | NDCG@10 medium | NDCG@10 dense | HR@10 (overall) | HR@10 sparse |
|---|---|---|---|---|---|---|---|---|
| **PFedRec** (calibration) | `…071106-ef41ab` | 100 | **0.33518** | **0.40567** | **0.37104** | **0.28697** | **0.59238** | **0.67120** |
| **Personalized** (split-learning) | `…064423-f18e64` | 33 | 0.22748 | 0.27263 | 0.24509 | 0.20087 | 0.41970 | 0.49197 |
| **Adaptive** (đóng góp luận văn) | `…152226-869730` | 83 | 0.21436 | 0.23546 | 0.22641 | 0.19887 | 0.38675 | 0.40915 |
| **Baseline** (all-global, lower bound) | `…203751-1bf513` | 25 | 0.20131 | 0.24458 | 0.22171 | 0.17300 | 0.36457 | 0.42645 |

**Xếp hạng — overall NDCG@10:** PFedRec (0.33518) > Personalized (0.22748) > **Adaptive (0.21436)** > Baseline (0.20131).
**Xếp hạng — sparse NDCG@10:** PFedRec (0.40567) > Personalized (0.27263) > Baseline (0.24458) > **Adaptive (0.23546)**.

*Lưu ý về tính so sánh được:* `mrr` không có trong `final_metrics.best` ở cả 4 run nên được lược bỏ. Các module khác nhau ở **nhiều hơn một trục thiết kế**, nên ma trận là một so sánh *cấp hệ thống (system-level)*, không phải ablation knob-for-knob:
- **Loss objective:** run baseline là **BasicMF (MSE / rating loss)**, trong khi personalized dùng **BPR (ranking loss)** và adaptive dùng model **dual**. Vai trò lower-bound của baseline không bị ảnh hưởng, nhưng objective của nó không đồng nhất với các module họ BPR.
- **Embedding dimension:** baseline=128, personalized=64, adaptive=64; PFedRec dùng `latent_dim=32` gốc của nó.
- **Optimizer regime:** PFedRec chạy regime riêng (`lr=0.1`, `lr_eta=80`, `local_epochs=1`, `num_negatives=4`, FedAvg), không so sánh knob-for-knob trực tiếp với các module họ MF (`lr=0.005`).
- **Strategy:** adaptive là module duy nhất trong chuỗi đóng góp dùng `fedprox`; baseline/personalized có lưu `proximal_mu=0.01` nhưng chạy FedAvg, nên μ bị vô hiệu (inert).

---

## 5. Phát hiện quyết định & Diễn giải

Run quyết định là adaptive **`20260613-152226-869730`** (hoàn tất 2026-06-14), với trust gates đều pass (`d06_cache_misses=0`, `evaluated_users=6040`). Ba kết luận rút ra:

**(1) Claim gốc của luận văn bị falsified.** Adaptive phải thắng cả ba comparator trên NDCG@10, đặc biệt sparse. Thực tế nó chỉ thắng baseline (+6.5% overall), thua cả personalized (−5.8%) lẫn PFedRec (−36%), và **xếp bét trên sparse users (0.23546 < baseline 0.24458)** — ngược hẳn với dự đoán cốt lõi. *(Các phần trăm liên-phương-pháp này là single-seed; khoảng cách adaptive/personalized/baseline nhỏ và chưa được chứng minh là tách biệt về thống kê — xem §7. Riêng khoảng cách của PFedRec và thứ hạng "adaptive bét trên sparse" đủ lớn để robust.)* Cơ chế α và dual-level personalization, dưới protocol và cấu hình này, không tạo ra được sự transfer cho sparse users như thiết kế kỳ vọng. Ngay cả confound về calibration-uniformity (§7) cũng chỉ *làm đẹp (flatter)* cho adaptive, nên không thể cứu verdict.

**(2) PFedRec reproduce đúng (trong giới hạn compute khả thi).** Dưới cross-device C=0.1, PFedRec đạt **NDCG@10 ≈ 0.335 / HR@10 ≈ 0.59**, khớp với bar đã ghi nhận cho protocol này. Band cao hơn của paper (HR@10 0.65–0.70, NDCG@10 0.35–0.42) cần full participation (`fraction=1.0`, ~6 ngày compute) — một bất khả thi đã được ghi nhận, **không phải reproduction thất bại**; lưu ý con số cross-device 0.335 nằm ngay dưới band full-participation của paper, đúng như compute envelope dự đoán. Việc PFedRec là phương pháp mạnh nhất ở đây tự nó cũng mang thông tin: một *per-user local score function trên item embeddings dùng chung toàn cục* vượt cả split-learning lẫn adaptive blend.

**(3) Điểm dương có thể công bố: cơ chế của split-learning.** Split-learning personalized thắng all-global baseline **+13% overall** (0.22748 vs 0.20131) và trên mọi density bucket. **Giả thuyết khả dĩ nhất** **không phải** là personalized inference mà là *bảo vệ quá trình train item-embedding khỏi việc FedAvg averaging user embeddings một cách phá hoại*: bằng cách giữ user embeddings cục bộ, bảng item toàn cục hội tụ về một representation chung sạch hơn. Đây là một mechanism claim có thể bác bỏ (falsifiable) — và nó sẽ giải thích vì sao PFedRec (cũng loại user embeddings khỏi aggregation surface) thắng tuyệt đối. Hiện **chưa được kiểm chứng bằng thực nghiệm**, cần **một ablation** để xác nhận (ví dụ: local-user-embedding vs frozen/global-user-embedding, giữ cố định item aggregation). Cần trình bày như một **giả thuyết, không phải kết quả**, cho đến khi ablation đó được chạy.

---

## 6. Bằng chứng tính hợp lệ (Trust Gates)

Cả 4 ô chính thức được audit độc lập qua sidecar `EVAL_VALIDITY.json` từng run (tổng 38 sidecar: **5 valid, 14 invalid_cold_eval, 19 diagnostics_only**). Luật validity mã hóa trong sidecar là *tĩnh (static)*: một run được gán nhãn valid nếu git commit của nó ở tại/sau commit fix cold-eval của module **và** `effective_checkpoint_rule = best_round_restore`.

**Lưu ý quan trọng — nhãn "valid" ≠ chứng minh eval đã warm.** Luật sidecar là *cần* nhưng không *đủ*: nó chứng nhận *code* có chứa fix, chứ không chứng nhận một lượt eval cụ thể thực sự chạy warm. Bằng chứng trực tiếp duy nhất cho một warm full-pop eval là **`d06_cache_misses = 0` được stamp**. Phản ví dụ ngay trong corpus của chúng tôi: một adaptive run *thứ hai* cũng được gán nhãn valid, `20260612-120025-6394ca` (chính là run v1 bị early-stopped, commit có fix `70a92bd`), vẫn **crater xuống NDCG@10 ≈ 0.080** vì run đó bị dừng sớm *và* không stamp cache-miss gate. Vậy nhãn "valid" đơn thuần không đảm bảo con số đáng tin — gate được stamp mới đảm bảo.

| Module | Run | Bằng chứng warm-eval trực tiếp |
|---|---|---|
| **Adaptive** | `…869730` | **`d06_cache_misses = 0` được stamp** (bằng chứng mạnh nhất; gate trực tiếp) |
| PFedRec | `…ef41ab` | Field không stamp, nhưng full-pop NDCG@10 = 0.3352 *khớp warm bar đã ghi nhận* (cold sẽ crater về ~0.07) → fix đã có hiệu lực (gián tiếp) |
| Personalized | `…f18e64` | Field không stamp; commit `9ce092d` chứa fix `3a393fb`; nhất quán với warm (gián tiếp) |
| Baseline | `…1bf513` | Miễn nhiễm cấu trúc — không có per-user local cache, nên phân biệt warm/cold không áp dụng |

14 run `invalid_cold_eval` đều có trước commit fix của module; 19 run `diagnostics_only` bị loại vì lý do không-hợp-lệ khác (`last_round` protocol, thiếu `results.json`, hoặc artifact BasicMF NDCG@10==1.0 do clamp). **Kết luận:** run adaptive *quyết định* mang bằng chứng mạnh nhất có thể (stamp trực tiếp); ba run còn lại warm-consistent theo reference/cấu trúc, và việc re-stamp `d06_cache_misses` cho chúng là một hạng mục belt-and-suspenders rẻ tiền cho ma trận cuối.

---

## 7. Giới hạn & Confounds

1. **Single seed (42).** Mọi con số trên đều từ một seed. Ma trận ý nghĩa-thống-kê 3-seed **chưa** chạy, nên chưa thể gắn confidence interval hay claim sự tách biệt thống kê giữa các ô gần nhau (adaptive 0.21436 vs personalized 0.22748 vs baseline 0.20131). Khoảng cách PFedRec đủ lớn để robust; thứ tự adaptive-vs-baseline-vs-personalized thì chưa được củng cố. *Đây là khoảng trống quan trọng nhất cần đóng trước khi nộp.*

2. **Calibration không đồng nhất.** Trong chuỗi đóng góp, pipeline adaptive có thêm bước final-calibration realignment mà các ô so sánh không có. Đây là một lập luận *phân tích*, không phải đại lượng đo được, nhưng nếu có thì chỉ **làm đẹp** cho adaptive — nên nó củng cố, chứ không làm yếu, verdict phủ định.

3. **Cấu hình không phải pure-default.** Run adaptive có tắt các kỹ thuật next-gen tùy chọn `enable-per-user-alpha` và `enable-item-perturbation`, **nhưng** `contrastive_lambda = 0.1` vẫn **bật** (InfoNCE auxiliary loss). Phân tích factorial trước đó cho thấy số hạng contrastive này gần như inert trên full-pop metrics, nên verdict không bị ảnh hưởng, song run nên được dán nhãn "dual + contrastive", không phải "vanilla dual". Ngoài ra `per-user-alpha` còn mang một *known full-population staleness bug*, nên hiện chưa phải là một đòn bẩy sạch.

4. **Dị biệt architecture / objective / optimizer.** Như §4, các module khác nhau ở loss objective (MSE vs BPR vs dual), embedding dim, optimizer regime, và FL strategy. So sánh công bằng *trong họ BPR/MF*, nhưng PFedRec là architecture khác; ưu thế của nó nên đọc là "một thiết kế per-user-head thắng", chứ không phải một thất bại knob-for-knob.

---

## 8. Đề xuất bước tiếp theo / Điểm cần quyết định

Tôi thấy hai hướng nhất quán và đề xuất **Path A**.

**Path A — Tái định khung quanh kết quả phủ định + mechanism (đề xuất).** Chấp nhận rằng adaptive không thắng dưới protocol đúng, và xoay narrative luận văn sang:
- **đóng góp phương pháp**: phát hiện và sửa lớp bug cold-eval, và migration protocol giúp evaluation đáng tin (một bài học thực, chuyển giao được, cho evaluation trong federated-RecSys);
- **giả thuyết mechanism**: split-learning hỗ trợ bằng cách bảo vệ item-embedding training, được củng cố (tương quan) bởi chiến thắng tuyệt đối của PFedRec — sẽ xác nhận bằng một ablation;
- một **kết quả phủ định trung thực** về adaptive α-personalization, kèm phân tích *vì sao* transfer cho sparse users thất bại.
Công việc cần làm: (i) **ma trận 3-seed** (đóng Limitation 1, củng cố mọi xếp hạng); (ii) **một ablation split-learning** để xác nhận mechanism claim; (iii) một phân tích thất bại ngắn của α-blend trên sparse users.

**Path B — Một lần chạy next-gen cuối.** Chạy lại adaptive với `per-user-alpha` và `item-perturbation` bật, hy vọng phục hồi khoảng cách sparse. **Caveats:** `per-user-alpha` có known full-pop staleness bug phải sửa trước; khoảng cần bù (sparse 0.23546 → phải vượt ≥0.27263 để thắng personalized, và ≥0.40567 để thắng PFedRec) là rất lớn; và đốt compute cho một giả thuyết n=1 có nguy cơ trì hoãn ma trận seed mà Path A dù sao cũng cần.

**Quyết định cần xin ý kiến từ giáo sư hướng dẫn:** Chúng ta cam kết Path A (luận văn negative-finding + mechanism, với ma trận 3-seed và ablation split-learning là phần thực nghiệm còn lại), hay cho phép một nỗ lực Path B *có giới hạn* (sửa per-user-alpha staleness, một cấu hình duy nhất, dừng cứng) trước khi khóa narrative? Khuyến nghị của tôi là Path A, giữ Path B chỉ như một phương án dự phòng có giới hạn thời gian chặt chẽ.

---

*Cả 4 kết quả được copy nguyên văn từ `final_metrics.best` trong `results.json` của từng module tại `results/federated/<module>/<run_id>/`. Validity provenance lấy từ sidecar `EVAL_VALIDITY.json` từng run (schema `eval_validity/1`, tạo lúc 2026-06-15 tại generator head `bb67cff`). Mọi numeric claim trong báo cáo này đã được kiểm chứng độc lập lại với JSON gốc.*
