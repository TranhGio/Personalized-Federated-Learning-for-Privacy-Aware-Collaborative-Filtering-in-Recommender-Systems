# Thí nghiệm cổng: client giữ bảng phim riêng Q_u (Giai đoạn 1, Việc 2)

**Trạng thái: ĐÃ RÚT, CHƯA CHẠY.** Phiên bản 1.3, ngày 04/11/2026.
Thí nghiệm cổng này đã rút theo quyết định của GVHD ngày 04/11/2026: ý tưởng bảng phim riêng Q_u được đưa vào
phần hướng phát triển của luận văn, không chạy lưới 8 ô và không áp dụng ngưỡng QUA / KHÔNG QUA CỔNG ở mục 7.
README được giữ nguyên làm hồ sơ đăng ký trước (các mục 1–13 không sửa); phần cài đặt `qu_grid` / `qu_diag` trong
`scripts/thesis/reeval.py` vẫn giữ: `qu_diag` (α = 0, mục 9) dùng cho việc chấm lại PFedRec có post-tuning,
`qu_grid` để dành cho hướng phát triển.

Mọi cấu hình, lưới tham số, ngưỡng và dự đoán dưới đây được ghi trước khi chạy bất kỳ ô nào.
Sau khi chạy, kết quả được điền vào `DECISION.md` cùng thư mục; README này không sửa số đã đăng ký
(thay đổi, nếu có, ghi vào mục 14 kèm ngày và lý do).

Nguồn yêu cầu: góp ý của giảng viên hướng dẫn (GVHD) trong `Vinh_lv.docx`, mục II Việc 2 và mục IV.
Thư mục kết quả: `results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/`.

## Tóm tắt (đọc 1 phút)

- **Câu hỏi:** mỗi user giữ bảng phim riêng Q_u (tinh chỉnh trên dữ liệu của mình, head cố định) rồi trộn với
  bảng chung theo α. NDCG@10 toàn bộ có đạt ≥ 0.3547 (≈ +1.0 điểm so với mốc PFedRec + hiệu chuẩn 0.3447118) không? (mục 1)
- **Lưới cố định:** α ∈ {0.3, 0.5, 0.7, 1.0} × E ∈ {1, 3} = 8 ô; hai ô α = 1.0 là mốc kiểm tra tin cậy. Không thêm bớt sau khi chạy. (mục 4)
- **Ngưỡng QUA CỔNG (mục IV của GVHD):** (i) ít nhất một ô có NDCG@10 toàn bộ ≥ 0.3547 VÀ (ii) Wilcoxon signed-rank
  theo từng user của ô đó so với mốc có p < 0.05. Không thỏa thì KHÔNG QUA CỔNG. (mục 7.1)
- **Cố định, không đổi:** bảng chung Q (`global_state_best.pt`, sha256 `279459ba…57`), head (w_u, b) đã hiệu chuẩn,
  tập ứng viên D-06 (phim test + 99 phim âm, giống hệt từng phim với số 0.3447118); chỉ các hàng của Q_u được huấn luyện,
  tốc độ học hàng phim lr_item = 29 648. (mục 2, 3, 5)
- **Dự đoán đăng ký trước:** KHÔNG QUA CỔNG (P1–P5, P7; mục 8).
- **Chi phí:** chỉ CPU, khoảng 3–5 phút cho toàn bộ 6040 user (đo trên dữ liệu tổng hợp cùng kích thước); GPU = 0. (mục 11)

---

## 1. Câu hỏi và mục đích

Câu hỏi: nếu mỗi người dùng giữ một **bản sao riêng của bảng phim** Q_u, tinh chỉnh trên dữ liệu của chính
mình (head giữ nguyên), rồi trộn với bảng chung theo hệ số α, thì NDCG@10 có tăng ít nhất +1.0 điểm so với
PFedRec + hiệu chuẩn (0.3447) hay không?

Đây là thí nghiệm **ngoại tuyến, chỉ CPU**, dùng làm cổng: chỉ đầu tư GPU cho phương pháp FL đầy đủ
(Giai đoạn 2) nếu vượt ngưỡng ở mục 7. Mọi ô của lưới đều được báo cáo, kể cả ô kém.

## 2. Đầu vào cố định

| Hạng mục | Giá trị |
|---|---|
| Run nguồn | PFedRec + hiệu chuẩn cuối, `20260726-083121-cf72e0`, module `federated-pfedrec` |
| Giao thức của run | cross-device, 1 user = 1 client, N = 6040, C = 0.1 (604 client/vòng), 100 vòng, `best_round_restore` (vòng tốt nhất 83), hiệu chuẩn 1 epoch head-only sau khôi phục, đánh giá toàn bộ 6040 user ở chỉ số vòng 101 (D-06) |
| Siêu tham số của run | latent_dim 32, SGD, lr 0.1, lr_eta 80, BCE, 4 mẫu âm/mẫu dương, batch 256, local_epochs 1, l2 = 0, seed 42 |
| Bảng phim chung q_i | `global_state_best.pt`, khóa `embedding_item.weight` (3706 × 32), sha256 `279459ba72987a98dfd1065c59ab0bb32f0e6f33323c08ccbc25211103b41c57` |
| Head cố định của user u | `affine_output.weight` (1 × 32) từ cache `federated-pfedrec/.embedding_cache/20260726-083121-cf72e0/partition_{pid}.pt` (đã qua hiệu chuẩn) **cộng** `affine_output.bias` (toàn cục, nằm trong `global_state_best.pt`) |
| Nơi có đầu vào | Hai đầu vào trên bị gitignore (`*.pt`, `.embedding_cache/`) và **không có trên laptop**; 8 ô chạy trên máy đang giữ chúng (CPU). Laptop chỉ chạy kiểm thử tổng hợp (mục 13). Trước khi chạy: `sha256sum global_state_best.pt` phải bằng giá trị trên, và cache phải đủ `manifest.json` + 6040 file |
| Dữ liệu | ML-1M, bundle `data/derived/` với `split_hash 5685bed7…`, `exclusion_sha256 b5a670c8…`; leave-one-out: phim test = tương tác mới nhất; 3706 phim, chỉ số 0–3705 |
| Nhóm user (theo số tương tác **train**, nửa mở) | sparse [0, 30): 809; medium [30, 100): 2322; dense [100, ∞): 2909 |
| Mã nguồn tham chiếu | commit của run `62961dc`; công cụ chấm lại `scripts/thesis/reeval.py` (sẽ mở rộng, mục 13); code gốc PFedRec: gitlink `IJCAI-23-PFedRec` @ `abc23809` (thư mục rỗng trong checkout laptop; đối chiếu trên GitHub Zhangcx19/IJCAI-23-PFedRec tại commit đó) |

Số tham chiếu NDCG@10 và HR@10 là khối `final_metrics.best` của run, đã được công cụ chấm lại tái lập với max|Δ| = 0.0:

| Chỉ số | Toàn bộ (6040) | sparse (809) | medium (2322) | dense (2909) |
|---|---|---|---|---|
| NDCG@10 | **0.3447118** | 0.4269673 | 0.3801614 | 0.2935401 |
| HR@10 | 0.6110927 | 0.7058096 | 0.6688200 | 0.5386731 |
| MRR¹ | 0.2821309 | 0.3551956 | 0.3082912 | 0.2409299 |

¹ MRR chỉ có trong `../reeval_cached_best.json` (biến thể `cached` của công cụ chấm lại), không nằm trong `final_metrics.best`
và không thuộc 8 chỉ số của cổng tin cậy (`trust_gate.not_compared = [sampled_mrr, eval_loss]`).

Điểm từng user của mốc này đã có sẵn: `../reeval_cached_best_per_user.csv` (6040 dòng). Phân bố NDCG@10
từng user: trung bình 0.3447, độ lệch chuẩn 0.3413, 38.9% user bằng 0 (phim test ngoài top-10), 13.9% bằng 1.

## 3. Giao thức tinh chỉnh cho từng người dùng

Ký hiệu: Q là bảng phim chung (3706 × 32), q_i là hàng i; w_u là head weight của user u (cache), b là head bias
(toàn cục); T_u là tập phim train của u (n_u = |T_u|, từ 19 đến 2313); t_u là phim test (1 phim/user);
P_u là tập mẫu âm của u; U_u là tập hàng đã cập nhật của Q_u.

Với mỗi user u (xử lý tuần tự, độc lập, trên CPU):

1. **Sao chép**: Q_u ← Q (bản sao đầy đủ, 474 KB).
2. **Tinh chỉnh E epoch**, với mỗi epoch e = 1, …, E (đánh số từ 1) dựng lại tập huấn luyện:
   - Như PFedRec: với mỗi phim dương i ∈ T_u, 1 mẫu dương (nhãn 1) + 4 mẫu âm (nhãn 0), lấy mẫu bác bỏ đều
     (rút độc lập, tức có hoàn lại) từ **tập mẫu âm P_u = {0..3705} \ T_u**, đúng hàm `_sample_train_negatives_seeded`
     của module. Phim test t_u **nằm trong P_u** (mục 12, điểm 1).
   - Seed mẫu âm epoch e: `np_rng(42, pid, 200 + e, "train_neg")`. Seed trộn batch: `torch_gen(42, pid, 200 + e, "dataloader")`.
   - Mất mát BCE trên sigmoid(w_u · Q_u[i] + b); batch 256; SGD **không momentum, không weight decay**; mỗi batch
     một lần forward/backward.
   - **Chỉ cập nhật các hàng của Q_u**. w_u và b bị đóng băng (`requires_grad=False`), không có pass cập nhật head.
   - Tốc độ học của hàng phim: **lr_item = lr × num_items × lr_eta = 0.1 × 3706 × 80 = 29 648**, đúng bằng
     bước cập nhật bảng phim trong vòng huấn luyện PFedRec của run (mục 12, điểm 2).
3. **Lưu trạng thái Q_u**: sau epoch 1 (ô E = 1) và sau epoch 3 (ô E = 3). E = 1 là trạng thái trung gian của
   cùng một quỹ đạo E = 3, nên hai ô dùng chung mẫu âm epoch 1; epoch 2 và 3 rút mẫu âm mới.
4. **Trộn** với mỗi α trong lưới, **chỉ trên tập hàng đã cập nhật** U_u = T_u ∪ (các mẫu âm đã rút trong E epoch),
   đúng nguyên văn Việc 2: Q_u^α[i] ← α · q_i + (1 − α) · Q_u[i] với i ∈ U_u; Q_u^α[i] ← q_i (gán trực tiếp) với i ∉ U_u.
   Với SGD không momentum và weight decay = 0, hàng i ∉ U_u có Q_u[i] == q_i trùng bit, nhưng phép trộn float32
   α·q + (1−α)·q **không** trả lại q trùng bit khi α ∈ {0.3, 0.7} (lệch 1 ulp ở phần lớn phần tử; chỉ đúng với 0.5 và 1.0),
   nên phải gán trực tiếp để điểm của ứng viên chưa chạm trùng bit với mốc. Về khác biệt với quy tắc trộn của
   Giai đoạn 2 (chỉ hàng T_u), xem mục 12, điểm 8.
5. **Chấm điểm** bằng head cố định (w_u, b) và bảng Q_u^α, theo mục 5.

**Quy ước α (chú ý):** α là trọng số của **bảng chung**. α = 1.0 là không cá nhân hóa (phải tái lập đúng 0.3447118);
α = 0.3 là cá nhân hóa mạnh nhất trong lưới. Quy ước này **ngược** với `reeval_alpha_grid_best.json` cũ
(ở đó α là trọng số của head riêng). Mọi file đầu ra của thí nghiệm này dùng tên cột `alpha_table` để tránh nhầm.

## 4. Lưới tham số (cố định, không thêm bớt sau khi chạy)

| | α = 0.3 | α = 0.5 | α = 0.7 | α = 1.0 |
|---|---|---|---|---|
| **E = 1** | ô 1 | ô 2 | ô 3 | ô 4 (= mốc, kiểm tra tin cậy) |
| **E = 3** | ô 5 | ô 6 | ô 7 | ô 8 (= mốc, kiểm tra tin cậy) |

Tổng 8 ô. Ô 4 và ô 8 phải cho đúng điểm từng user của mốc (mục 7.2). Không chọn lại α hay E sau khi xem kết quả;
nếu QUA CỔNG, α và E được cố định cho Giai đoạn 2 theo ô thắng (quy tắc ở mục 7.1, điểm 3), đúng yêu cầu
"không chọn lại trên tập test".

## 5. Chấm điểm

- Dùng nguyên hàm `evaluate_pfedrec_sampled` của module, với `np_rng(42, pid, 101, "eval_neg")` và tập loại trừ
  train ∪ {test} như lúc đánh giá D-06. Tập ứng viên của mỗi user (phim test + 99 phim âm) do đó **giống hệt
  từng phim** với tập đã dùng cho số 0.3447118, và giống nhau ở cả 8 ô.
- Danh sách 100 ứng viên được thu bằng forward pre-hook trên model trong lúc hàm chạy (hàm gọi model đúng hai lần:
  phim test rồi 99 phim âm); đường đi tính điểm không đổi; thu một lần mỗi user.
- Điểm = sigmoid(w_u · Q_u^α[i] + b); xếp hạng giảm dần bằng `np.argsort(-scores)` của hàm (phim test ở chỉ số 0,
  giữ nguyên hành vi); HR@10, NDCG@10 = 1/log2(hạng + 1) nếu trúng, MRR. Mốc không có hòa điểm test/âm ở mức bão hòa:
  `eval_loss ≥ 1.0` chỉ xảy ra ở 1/6040 user (NDCG = 0).
- Tổng hợp theo đúng công thức sufficient-statistics của server (tổng rồi chia một lần), toàn bộ và theo 3 nhóm.
- Ghi thêm cho mỗi (user, ô): số hàng đã cập nhật `n_touched_rows`, số hàng ứng viên đã cập nhật `n_touched_candidates`
  (0–100), hàng phim test có bị cập nhật không `test_row_touched`, độ dịch trung bình ‖Q_u[i] − q_i‖₂ trên U_u `mean_row_shift`.

## 6. Kiểm định thống kê (quy định trước)

Đơn vị quan sát là **người dùng**; mỗi ô so với mốc theo cặp trên cùng 6040 user và cùng tập ứng viên.
Kiểm định này đo biến thiên **giữa người dùng**, không đo biến thiên do seed huấn luyện.

| Hạng mục | Quy định |
|---|---|
| Thống kê chính | d_u = NDCG@10_ô(u) − NDCG@10_mốc(u) |
| Kiểm định | Wilcoxon signed-rank theo cặp, **một phía** (H₁: ô > mốc), `scipy.stats.wilcoxon(alternative="greater", zero_method="wilcox", method="approx")`; p hai phía (`alternative="two-sided"`) báo cáo kèm (mục 12, điểm 7) |
| Xử lý d_u = 0 | Loại khỏi kiểm định (chuẩn `wilcox`); báo n_nonzero. Báo thêm p theo `zero_method="pratt"` để kiểm tra độ nhạy |
| Hiệu chỉnh đa so sánh | p thô dùng cho tiêu chí QUA CỔNG (đúng văn bản). Báo thêm p hiệu chỉnh Holm trên 6 ô có α < 1 và cờ `holm_flips` (kết luận có đổi khi dùng Holm hay không; mục 12, điểm 4) |
| Độ lớn hiệu ứng | r rank-biserial theo cặp = (W⁺ − W⁻)/(W⁺ + W⁻); Δ trung bình NDCG@10 và HR@10 với khoảng tin cậy 95% bootstrap theo cặp (10 000 lần, seed 42) |
| Theo nhóm | Lặp lại cho sparse/medium/dense (mô tả; tiêu chí cổng chỉ xét toàn bộ) |
| Công suất (định hướng, không phải tiêu chí) | sd(d_u) chưa biết trước khi chạy; mốc có sd(NDCG@10_u) = 0.3413 nên sd(d_u) ≤ 0.68; mô phỏng dịch hạng trên phân bố hạng thật của mốc cho sd(d_u) ≈ 0.05–0.27. Với n = 6040, SE(Δ) ≈ 0.0006–0.0035, nên Δ = +0.010 nếu có thật cho p ≪ 0.05 ở cả hai phía. Lưu ý Wilcoxon (`wilcox`) kiểm định tính đối xứng của các d_u ≠ 0 quanh 0, không kiểm định trung bình: Δ ≥ +0.010 do ít user nhảy hạng lớn trong khi đa số d_u ≠ 0 âm vẫn có thể cho p ≥ 0.05; hai điều kiện ở 7.1 là độc lập. sd(d_u) và n_nonzero quan sát được ghi trong `stats_wilcoxon.json` |

## 7. Ngưỡng quyết định (theo mục IV của GVHD; hai diễn giải của người thực hiện ghi ở mục 12, điểm 7) và điều kiện tin cậy

> Nguyên văn: "Có ít nhất một cấu hình trong lưới đạt NDCG@10 ≥ 0.3547 (≥ +1.0 điểm so với 0.3447); và
> Wilcoxon theo từng user so với PFedRec + hiệu chuẩn có p < 0.05."

### 7.1. QUA CỔNG khi và chỉ khi thỏa cả hai

1. Có ít nhất một ô trong 8 ô đạt **NDCG@10 toàn bộ ≥ 0.3547** (ngưỡng ràng buộc là con số này, so sánh trực tiếp trên
   giá trị float64 của trung bình 6040 user theo công thức mục 5, không làm tròn; GVHD viết "+1.0 điểm so với 0.3447"
   với mốc làm tròn, so với mốc chính xác 0.3447118 thì tương đương Δ ≥ +0.0099882). Ô 4 và ô 8 là mốc (Δ ≡ 0 theo cấu trúc)
   nên thực chất chỉ 6 ô α < 1 có thể thắng; và
2. Wilcoxon theo từng user của ô đó so với mốc có **p < 0.05** (một phía H₁: ô > mốc, p thô; mục 6 và mục 12 điểm 7;
   p hai phía báo cáo kèm).
3. **Ô thắng** (chỉ dùng để cố định α, E cho Giai đoạn 2): nếu nhiều ô cùng thỏa 1 và 2, ô thắng là ô có NDCG@10 toàn bộ
   lớn nhất; hòa (chênh < 0.001, cỡ 1 SE theo mục 6) thì ưu tiên α lớn hơn (gần bảng chung hơn), rồi E nhỏ hơn.
   Quy tắc cố định trước khi chạy, không đổi sau khi xem số.

HR@10, MRR và các chỉ số theo nhóm chỉ để báo cáo (đúng mục II), không tham gia tiêu chí cổng.
Δ của ô thắng là ước lượng **sau chọn lọc** (tốt nhất trong 6 ô trên cùng tập test; độ lệch lạc quan kỳ vọng dưới
giả thuyết không ≈ 1–2 SE ≈ 0.002, nhỏ so với ngưỡng), được ghi với nhãn "ước lượng Giai đoạn 1, chưa xác nhận";
con số xác nhận chỉ đến từ Giai đoạn 2 với 2–3 seed mới.

Nếu không thỏa: KHÔNG QUA CỔNG. Kết quả được đưa vào chuỗi ablation như một mắt xích phủ định có giải thích cấu trúc (mục 8).

### 7.2. Điều kiện tin cậy

Bất kỳ điều nào không đạt thì **không báo cáo ô nào**; sửa mã rồi chạy lại. Mỗi lần chạy bị hủy được ghi một dòng vào
`DECISION.md` (ngày, kiểm tra nào hỏng, commit sửa, có đổi gì ở mã tinh chỉnh/chấm điểm không); lưới, ngưỡng, kiểm định
và dự đoán không đổi, và kết quả của lần chạy bị hủy không được dùng để điều chỉnh bất kỳ mục đã đăng ký.

| Kiểm tra | Yêu cầu |
|---|---|
| sha256 của `global_state_best.pt` | = `279459ba…57` (kiểm tra bằng tay trước khi chạy; công cụ chỉ ghi lại sha) |
| Cache head | đủ 6040 file `partition_{pid}.pt`, chữ ký `manifest.json` khớp, 0 cache miss |
| Cổng tin cậy có sẵn của công cụ | biến thể `cached` tái lập `final_metrics.best` với \|Δ\| < 1e-6 trên 8 chỉ số (NDCG@10 và HR@10 × {toàn bộ, sparse, medium, dense}) và đúng 4 số đếm user (6040/809/2322/2909) |
| Ô α = 1.0 (ô 4 và ô 8) | **điểm từng user** trùng bit với `reeval_cached_best_per_user.csv` (so trên giá trị float đã parse; CSV ghi `repr(float)` nên parse chính xác; cả NDCG@10 và HR@10, 6040/6040 user) |
| Số user | 6040 = 809 + 2322 + 2909 ở mọi ô |
| Q_u hữu hạn | `torch.isfinite(Q_u).all()` cho mọi user sau epoch 1 và 3 (sai thì dừng) |
| Hàng chưa chạm sau trộn | với một mẫu 50 user và **mọi α trong lưới**, Q_u^α[i] == q_i trùng bit với mọi i ∉ U_u |
| Bất biến d_u | với mọi ô: số user có `n_touched_candidates = 0` nhưng d_u ≠ 0 (NDCG@10 hoặc HR@10) phải bằng 0; ghi vào `trust_checks` |
| Tập mẫu âm | kiểm tra trên mẫu âm **đã rút** (hàm dựng dữ liệu không trả về P_u): (a) ô lưới gọi với `exclude_items=None`; (b) mọi mẫu âm đã rút ∉ T_u, mọi user, mọi epoch; (c) `n_hit` = số mẫu âm đã rút trùng t_u, gộp 6040 user × E epoch: lưới sạch phải có n_hit > 0 (kỳ vọng ≈ Σ_u 4·n_u·E/(3706 − n_u)), biến thể rò rỉ D2/D4 phải có n_hit = 0; ghi `n_hit` vào `trust_checks` |

## 8. Dự đoán đăng ký trước (ghi trước khi chạy)

**Lập luận cấu trúc.** Head tuyến tính và cố định nên điểm của ứng viên i chỉ phụ thuộc vào hàng Q_u^α[i].
Cả 100 ứng viên đều là phim u chưa tương tác, nên hàng của chúng chỉ thay đổi nếu bị rút làm mẫu âm; khi đó
hàng bị kéo về phía điểm thấp. Vì phim test nằm trong P_u cùng với 99 phim âm, phim test và mỗi phim âm có
**cùng xác suất bị rút** (đối xứng về xác suất). Tác dụng lên NDCG@10 **không đối xứng**: phim test bị kéo xuống làm
mất toàn bộ NDCG_u, còn một phim âm bị kéo xuống chỉ giúp khi nó đang xếp trên phim test. Tính trên 6040 user,
kỳ vọng Δ ≤ 0 với mọi phân bố hạng tương thích với MRR = 0.2821 của mốc. Phần "riêng" của Q_u chỉ nằm ở các hàng
đã xem (T_u), là các hàng không bao giờ được chấm (đây cũng là lý do quy tắc trộn của Giai đoạn 2 làm cổng thành
tầm thường, mục 12 điểm 8). Hơn nữa, với head tuyến tính, mọi biến đổi tuyến tính theo user của bảng phim đều bị
head hấp thụ; bảng riêng chỉ thêm được sai lệch theo từng phim, và sai lệch đó không lan sang phim chưa xem.

**Tỉ lệ hàng ứng viên bị cập nhật, kỳ vọng** (mỗi user: p_u = 1 − (1 − 1/(3706 − n_u))^(4·n_u·E); P(≥1) = 1 − (1 − p_u)^99):

| Nhóm | n_u trung vị | n_u trung bình | % ứng viên bị chạm, E = 1 | % ứng viên bị chạm, E = 3 | P(user có ≥ 1 ứng viên bị chạm), E = 1 / E = 3 |
|---|---|---|---|---|---|
| sparse | 23 | 23.6 | 2.5% | 7.4% | 0.92 / 1.00 |
| medium | 54 | 57.8 | 6.1% | 17.2% | 0.99 / 1.00 |
| dense | 212 | 289.0 | 27.3% | 56.2% | 1.00 / 1.00 |
| toàn bộ | 95 | 164.6 | 15.8% | 34.7% | 0.99 / 1.00 |

Mọi % và xác suất là **trung bình trên từng user của nhóm** theo n_u thật trong `../reeval_cached_best_per_user.csv`.
Hai cột n_u chỉ mô tả; tính tại n_u trung vị cho giá trị thấp hơn (sparse E = 3: 7.2%; dense E = 1: 21.6%; toàn bộ E = 1: 10.0%)
vì phân bố n_u lệch phải. Trung bình nhóm là đại lượng được so với `test_row_touched` quan sát trong P4.

**Dự đoán cụ thể** (sẽ đối chiếu trong `DECISION.md`; biên 0.001 ≈ 1 SE, chênh lệch nhỏ hơn biên ghi là "không phân biệt"):

| # | Dự đoán | Cách kiểm |
|---|---|---|
| P1 | Không ô nào đạt NDCG@10 toàn bộ ≥ 0.3547; mọi ô có Δ toàn bộ ≤ +0.005. Hai kịch bản về độ lớn: **kéo mạnh** (độ dịch logit mỗi lượt chạm = (lr_item/batch)·σ·‖w_u‖² ≫ 1, kỳ vọng với lr_item = 29 648; hàng bị chạm rơi dưới mọi hàng chưa chạm): Δ toàn bộ ≈ −0.024 (E = 1) / −0.061 (E = 3), dense ≈ −0.038 / −0.092, sparse ≈ −0.007 / −0.019 (mô phỏng trên phân bố hạng thật của mốc; cận xấu nhất −0.032 / −0.074 toàn bộ). **Kéo yếu** (độ dịch ≪ 1, hạng không đổi): Δ ∈ [−0.005, +0.005] mọi ô. Kịch bản thực tế nhận diện sau khi chạy bằng `mean_row_shift` (P7), không chọn lại sau khi xem Δ | bảng 8 ô |
| P2 | Trong mỗi hàng E, Δ(α = 0.3) ≤ Δ(α = 0.7) + 0.001; trong mỗi cột α < 1, Δ(E = 3) ≤ Δ(E = 1) + 0.001 | bảng 8 ô |
| P3 | Ở cả 6 ô α < 1, Δ NDCG@10 của nhóm dense là nhỏ nhất (âm nhất) trong 3 nhóm | bảng theo nhóm |
| P4 | Tỉ lệ `test_row_touched` quan sát theo nhóm xấp xỉ hai cột % ở bảng trên (± 1 điểm) | cột chẩn đoán |
| P5 | Tầng `test_row_touched = 1` (mỗi ô α < 1): trung bình d_u < 0, và trong số user có NDCG mốc > 0, số user d_u < 0 nhiều hơn số user d_u > 0 (user có NDCG mốc = 0 chỉ có thể có d_u ≥ 0 nên không dùng để kiểm "giảm"). Tầng `n_touched_candidates = 0`: 100% user có d_u = 0 | phân tầng |
| P7 | Kịch bản kéo mạnh xảy ra: trung vị `mean_row_shift` trên U_u lớn hơn 10 lần trung vị ‖q_i‖₂ của bảng chung, ở cả E = 1 và E = 3 | cột chẩn đoán |

(P6 thuộc biến thể chẩn đoán ở mục 9, chỉ tính nếu được duyệt và chạy.)

Kỳ vọng tổng thể: **KHÔNG QUA CỔNG**. Nếu dữ liệu nói ngược lại, đó là bằng chứng chống lại lập luận cấu trúc ở trên
(P1–P5, P7 bị bác). Kết luận QUA / KHÔNG QUA trong mọi trường hợp chỉ theo mục 7.1; điều kiện tin cậy 7.2 là bắt buộc
cho mọi lần chạy, bất kể kết quả theo hướng nào.

## 9. Biến thể chẩn đoán ngoài lưới — KHÔNG thuộc Việc 2, không dùng cho quyết định cổng, chỉ chạy sau khi GVHD duyệt

Mục đích: đo trực tiếp **rò rỉ do loại phim test khỏi mẫu âm** khi suy luận bằng bảng riêng, vì đây đúng là cách
code chính thức của PFedRec chấm điểm (commit `abc23809`: `engine.py` hàm `fed_evaluate`, dòng 170–171 nạp
`client_model_params[user]` là state dict đầy đủ sau huấn luyện cục bộ, gồm `embedding_item.weight` đã post-tune;
`data.py` dòng 77–79 tính tập mẫu âm = toàn bộ phim trừ mọi phim đã tương tác, trước khi tách leave-one-out, nên loại cả
phim test và phim validation). Biến thể này phục vụ câu hỏi của Việc 3 (khoảng cách tái lập) nhưng **không thay thế và
không gộp vào Việc 3** (ML-100K, C = 1.0 và 0.1); kết quả không trích làm kết quả của phương pháp, không ảnh hưởng mục 7,
và nếu được duyệt chỉ ghi vào `DECISION.md` mục 4, tách khỏi bảng 8 ô. Khoảng cách tái lập so với số công bố 0.4436 (ML-1M)
là ≈ 9.9 điểm với run cf72e0 này (0.4436 − 0.3447); con số "khoảng 10.8 điểm" trong góp ý ứng với run chưa hiệu chuẩn
`20260608-071106-ef41ab` (0.3352).

| Biến thể | α | E | Tập mẫu âm | Ý nghĩa |
|---|---|---|---|---|
| D1 | 0.0 | 1 | {0..3705} \ T_u (sạch) | suy luận kiểu code gốc, không rò rỉ |
| D2 | 0.0 | 1 | {0..3705} \ (T_u ∪ {t_u}) (= bảng loại trừ D-13 của ta) | suy luận kiểu code gốc, **có rò rỉ** |
| D3 | 0.0 | 3 | sạch | như D1, 3 epoch |
| D4 | 0.0 | 3 | có rò rỉ | như D2, 3 epoch |

Dự đoán P6 (tiêu chí số): (a) Δ NDCG@10 toàn bộ của D2 so với D1 ≥ +0.010 và Wilcoxon theo cặp (D2 > D1, một phía,
cùng quy định mục 6) có p < 0.05; tương tự D4 so với D3. (b) D2 hoặc D4 thu hẹp ít nhất một nửa khoảng cách tới số công bố,
tức NDCG@10 toàn bộ ≥ 0.394 (= 0.3447 + ½·(0.4436 − 0.3447)), trong khi D1 và D3 đều < 0.3547. P6 Đúng chỉ khi cả (a) và (b).
Nếu đúng, phần lớn khoảng cách tái lập nằm ở cách suy luận + rò rỉ; phần còn lại do C = 0.1 sẽ được đo riêng ở Việc 3.
Chi phí: 2 quỹ đạo tinh chỉnh bổ sung mỗi user, vài phút CPU. Sẽ xin ý kiến trong báo cáo thứ Sáu 06/11/2026.

## 10. Đầu ra

Tất cả nằm trong thư mục này, được commit vào git (JSON, CSV, MD không bị gitignore):

| File | Nội dung |
|---|---|
| `README.md` | file này |
| `reeval_qu_grid_best.json` | khối provenance như các file `reeval_*` hiện có (tool, argv, globals_sha256, cache_dir, run_seed, eval_round_num = 101, …) + `grid` (α, E), `pool = "clean"`, `lr_item = 29648`, `metrics_by_cell` khóa `"a0.30_E1"` … `"a1.00_E3"` với đủ NDCG@10, HR@10, MRR toàn bộ và theo nhóm, `trust_checks` (mục 7.2, kể cả `n_hit`, số user vi phạm bất biến d_u), thời gian chạy. Khóa `metrics` cấp cao nhất (bắt buộc theo schema) **luôn** là ô `a1.00_E1` (mốc tái lập), không phải ô tốt nhất; không có khóa `*_star` |
| `reeval_qu_grid_best_per_user.csv` | dạng dài, 6040 × 8 dòng, đúng 15 cột: `pid, user_group, n_interactions, staleness_at_best, alpha_table, epochs, pool, n_touched_rows, n_touched_candidates, test_row_touched, mean_row_shift, sampled_ndcg@10, sampled_hr@10, sampled_mrr, eval_loss` |
| `stats_wilcoxon.json` | cho mỗi ô trong 6 ô α < 1 (ô 4, 8 chỉ ghi n_nonzero = 0 làm kiểm tra): n, n_nonzero, sd(d_u), W⁺, W⁻, p một phía và p hai phía (wilcox), p (pratt), p Holm, `holm_flips`, r rank-biserial, Δ trung bình NDCG@10/HR@10 và CI 95% bootstrap, theo toàn bộ và 3 nhóm; kèm `scipy_version`, `numpy_version`, `argv`, `inputs_sha256` của hai CSV |
| `reeval_qu_diag_best.json`, `…_per_user.csv` | biến thể D1–D4 (chỉ khi được duyệt), cùng schema, cột `pool ∈ {clean, leaky}` |
| `DECISION.md` | điền sau khi chạy: bảng 8 ô, bảng Wilcoxon, đối chiếu P1–P5, P7 (P6 nếu chạy), số lần chạy bị hủy, kết luận QUA / KHÔNG QUA CỔNG, ngày, commit |

## 11. Ước lượng chi phí

| Hạng mục | Ước lượng |
|---|---|
| Tinh chỉnh + chấm 8 ô | mỗi epoch đi qua 5 × 994 169 ≈ 5.0 triệu dòng (toàn bộ user), batch 256, một lần backward vào bảng phim; E = 3 ≈ 15 triệu dòng. Đo trên CPU laptop (6 luồng) với run tổng hợp đúng kích thước (3706 phim, dim 32, 99 phim âm, 200 user lấy ngẫu nhiên theo phân bố n_u thật): 33 ms/user, ngoại suy theo tổng n_u ≈ 3.1 phút cho 6040 user. Vẫn đo lại bằng `--limit-users 200` trên dữ liệu thật và ghi vào `DECISION.md` |
| Chấm điểm | 8 ô × 6040 user; mỗi lượt toàn bộ user khoảng 3–4 giây theo nhật ký các biến thể cũ của run này |
| Bộ nhớ | tối đa hai bản sao Q_u (474 KB mỗi bản: trạng thái sau epoch 1 và bản đang huấn luyện) cộng một bản trộn Q_u^α tạm và `global_snapshot` tại một thời điểm; không lưu Q_u của mọi user |
| GPU | 0 |

## 12. Các quyết định của người thực hiện ngoài văn bản của GVHD (minh bạch)

1. **Tập mẫu âm.** Góp ý yêu cầu đồng thời "giữ nguyên như các baseline" và "không loại phim test". Hai yêu cầu
   này mâu thuẫn vì bảng loại trừ D-13 của mọi run hiện có (kể cả cf72e0) **loại phim test** khỏi mẫu âm huấn luyện
   (`task.py`, hàm `prepare_user_train_data`, hợp `exclude_items` = train ∪ {test}). Cổng chọn yêu cầu thứ hai
   (tập sạch, không rò rỉ) vì đó là lý do GVHD nêu; rò rỉ của bảng chung trong các run cũ được khai báo là giới hạn
   (pha loãng ~1/604 mỗi vòng), không chạy lại baseline trong khuôn khổ cổng.
2. **Tốc độ học của hàng phim (lr_item)** không được chỉ định; chọn đúng bước cập nhật bảng phim của PFedRec trong run
   (29 648, SGD) để giữ nghĩa "như PFedRec". Không thử lr khác trong cổng; nếu muốn thêm lr, đó là lưới mới và phải đăng ký lại.
3. **Head cố định** gồm cả `affine_output.weight` (riêng) và `affine_output.bias` (toàn cục); cả hai không đổi.
4. **Holm** chỉ là báo cáo bổ sung; tiêu chí cổng dùng p thô đúng văn bản. Nếu GVHD muốn tiêu chí dùng Holm, đổi
   trước khi chạy và ghi vào mục 14.
5. **Seed** mẫu âm và trộn batch được cố định theo (42, pid, 200 + e) để hai lần chạy cho cùng kết quả trùng bit;
   PFedRec gốc không seed trộn batch, nhưng điều này không đổi phân bố kết quả.
6. **Biến thể chẩn đoán D1–D4** (mục 9) là đề xuất thêm, chờ duyệt.
7. **Phạm vi và chiều của kiểm định** không được chỉ định. (a) "NDCG@10 ≥ 0.3547" hiểu là NDCG@10 **toàn bộ 6040 user**,
   vì mốc 0.3447 là số toàn bộ (mốc theo nhóm là 0.427/0.380/0.294 nên ngưỡng không có nghĩa theo nhóm). (b) Chọn Wilcoxon
   **một phía** (H₁: ô > mốc) vì câu hỏi của cổng là "ô có tốt hơn mốc không" và điều kiện 1 đã cố định chiều cải thiện.
   Một phía ở đây là lựa chọn **chặt hơn**: ô có Δ ≥ +0.010 nhưng thống kê hạng có dấu âm (đa số user xấu đi) sẽ có
   p một phía ≈ 1 và KHÔNG QUA, trong khi p hai phía có thể < 0.05. p hai phía được báo cáo kèm; nếu GVHD muốn tiêu chí
   dùng hai phía, đổi trước khi chạy và ghi vào mục 14.
8. **Tập hàng được trộn.** Việc 2 viết "trộn theo hàng đã bị cập nhật"; với SGD dày trên `embedding_item` (BCE, 4 mẫu âm)
   tập này là U_u = T_u ∪ (mẫu âm đã rút), và cổng theo đúng nghĩa đó (mục 3, bước 4): các hàng mẫu âm bị kéo xuống **vẫn
   được giữ trong Q_u^α** với trọng số (1 − α). Mô tả phương pháp FL ở Giai đoạn 2 (mục IV) lại dùng quy tắc khác: "chỉ trộn α
   theo hàng phim đã tương tác (T_u), hàng khác lấy bản mới nhất từ server". Nếu áp quy tắc Giai đoạn 2 vào cổng (head cố định),
   100 ứng viên của mỗi user đều ∉ T_u nên đều lấy q_i chung, điểm trùng bit với mốc và d_u ≡ 0 ở cả 8 ô, tức chính là ô 4/ô 8,
   không cần chạy thêm. Vì vậy cổng chỉ có nội dung với cách đọc Việc 2, và kết quả cổng (dương hay âm) chỉ nói về cơ chế
   "hàng mẫu âm bị dịch", là cơ chế Giai đoạn 2 sẽ loại bỏ khi lấy hàng chưa tương tác từ server. Điểm này sẽ nêu trong báo cáo
   thứ Sáu 06/11/2026 để GVHD xác nhận; không chặn việc chạy cổng.
9. **Cố định E cho Giai đoạn 2.** Văn bản chỉ yêu cầu cố định α; người thực hiện cố định thêm E theo ô thắng để Giai đoạn 2
   không có bậc tự do mới.
10. **Giả định của P1 (kịch bản kéo mạnh)** là ‖w_u‖² đủ lớn để một lượt chạm đẩy hàng xuống dưới mọi hàng chưa chạm;
    không kiểm tra được trên laptop vì không có cache head. P7 và `mean_row_shift` dùng để phân xử sau khi chạy.

## 13. Kế hoạch cài đặt và kiểm thử trước khi chạy

- Thêm biến thể `qu_grid` (và `qu_diag` cho mục 9) vào `PFEDREC_VARIANTS` của `scripts/thesis/reeval.py`
  (nhánh `feat/gate-qu-reeval`), cài đặt bằng **một driver mới `run_qu_grid(...)`** gọi từ `run_pfedrec_suite`,
  không đi qua `run_reeval`/`_resolve_local_state` (vốn chỉ chấm một lần mỗi user):
  - quét cache fail-closed và `verify_cache_signature` như biến thể `cached`; tự chạy `cached` + `trust_gate()` nội bộ và ghi verdict
    vào `trust_checks` (không ghi đè các file `reeval_cached_best*` ở run dir);
  - hàm huấn luyện mới `finetune_item_table_one_epoch(model, items, ratings, lr_item, batch_size, generator)`: đóng băng
    `affine_output.*`, một optimizer `torch.optim.SGD(model.embedding_item.parameters(), lr=lr_item, momentum=0, weight_decay=0)`,
    `DataLoader(..., shuffle=True, generator=generator)`, một lần forward/backward mỗi batch. Không dùng lại
    `train_pfedrec_single_user` (hai pass/batch, luôn bước head, loader không seed, không đóng băng được head bằng lr = 0);
  - driver gọi hàm này một lần cho mỗi epoch e ∈ {1, 2, 3} với
    `prepare_user_train_data(user_idx=pid, user_train_items=T_u, num_items=3706, num_negatives=4, run_seed=42, round_num=200 + e, exclude_items=None)`
    (biến thể rò rỉ D2/D4 dùng `exclude_items=user.exclude_items`) và `generator = torch_gen(42, pid, 200 + e, "dataloader")`;
    sau e = 1 lưu bản sao Q_u(E = 1), tiếp tục e = 2, 3;
  - theo dõi U_u (hàng đã xuất hiện trong batch); trộn bằng mask: `Q_alpha = q_global.clone(); Q_alpha[U] = alpha * q_global[U] + (1 - alpha) * Q_u[U]`,
    rồi `adapter.set_globals(model, {"embedding_item.weight": Q_alpha, "affine_output.bias": b})` và `adapter.evaluate_user`,
    cộng vào một `SuffStats` riêng cho từng ô (8 ô); khôi phục `global_snapshot` sau mỗi user;
  - thu danh sách 100 ứng viên bằng `model.register_forward_pre_hook` quanh `adapter.evaluate_user` (không sửa `task.py`),
    một lần mỗi user, để tính `n_touched_candidates`, `test_row_touched`;
  - thêm `--out-dir` (mặc định = run dir, giữ nguyên hành vi cũ); chỉ `qu_grid`/`qu_diag` ghi vào `--out-dir`;
  - ghi CSV dài bằng hàm riêng `write_qu_per_user_csv` với đúng 15 cột ở mục 10; không dùng `write_per_user_csv`
    (writer hiện có dùng `extrasaction="ignore"` nên sẽ lặng lẽ bỏ cột mới);
  - `dump_json(metrics=metrics_by_cell["a1.00_E1"], extra={"metrics_by_cell": ..., "trust_checks": ...})`;
  - thêm `scripts/thesis/gate_stats.py` (scipy) đọc `reeval_qu_grid_best_per_user.csv` và `../reeval_cached_best_per_user.csv`
    (chỉ cần hai CSV đã commit), xuất `stats_wilcoxon.json`; khai báo `scipy>=1.11` trong `scripts/foundation/pyproject.toml`
    (extra `stats`; môi trường chạy ô đã có scipy gián tiếp qua `scikit-learn`).
- Kiểm thử bắt buộc trước khi chạy thật (`scripts/foundation/tests/test_reeval_qu.py`, CPU, dữ liệu tổng hợp nhiều epoch):
  α = 1.0 trùng `cached` trùng bit; Q_u^α trùng bit q_i trên hàng ngoài U_u cho mọi α ∈ {0.3, 0.5, 0.7, 1.0}; tập sạch rút được
  phim test ít nhất một lần (n_hit > 0), tập rò rỉ không bao giờ (n_hit = 0), không mẫu âm nào thuộc T_u; cùng seed cho cùng Q_u;
  danh sách ứng viên thu được có đúng 100 phần tử, phần tử 0 là phim test, 99 phần tử còn lại ∉ train ∪ {test}; header CSV
  trùng đúng 15 cột và mỗi ô đủ 6040 dòng; `--self-test` hiện có vẫn qua.
- Thứ tự chạy thật (lệnh đăng ký trước, chạy tại thư mục gốc repo trên máy có đầu vào):
  1. pre-flight: `sha256sum results/federated/pfedrec/20260726-083121-cf72e0/global_state_best.pt` = `279459ba…57`;
     `test -f federated-pfedrec/.embedding_cache/20260726-083121-cf72e0/manifest.json`;
  2. đo thời gian: `python scripts/thesis/reeval.py --run-dir results/federated/pfedrec/20260726-083121-cf72e0 --module pfedrec --globals best --local-variant qu_grid --device cpu --limit-users 200 --no-json --no-per-user`;
  3. chạy thật: `python scripts/thesis/reeval.py --run-dir results/federated/pfedrec/20260726-083121-cf72e0 --module pfedrec --globals best --local-variant qu_grid --device cpu --out-dir results/federated/pfedrec/20260726-083121-cf72e0/gate_qu`;
  4. kiểm định: `python scripts/thesis/gate_stats.py --grid results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/reeval_qu_grid_best_per_user.csv --baseline results/federated/pfedrec/20260726-083121-cf72e0/reeval_cached_best_per_user.csv --grid-json results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/reeval_qu_grid_best.json --out results/federated/pfedrec/20260726-083121-cf72e0/gate_qu/stats_wilcoxon.json` (`--grid-json` để điều kiện 1 dùng NDCG@10 tổng hợp theo công thức mục 5, không tính lại từ CSV);
  5. điền `DECISION.md`, commit.

## 14. Nhật ký thay đổi README

| Ngày | Phiên bản | Thay đổi |
|---|---|---|
| 2026-11-03 | 1 | Đăng ký trước: cấu hình, lưới 4 × 2, ngưỡng, kiểm định, dự đoán, biến thể chẩn đoán đề xuất |
| 2026-11-03 | 1.1 | Sau vòng phản biện (41 điểm, trước khi chạy): thêm Tóm tắt; trộn chỉ trên U_u (float32 không trùng bit khi trộn mọi hàng); quy tắc ô thắng; ngưỡng ràng buộc = 0.3547; khai báo chiều/phạm vi kiểm định, tập hàng trộn, cố định E (mục 12, điểm 7–10); P1 hai kịch bản, P2/P5 có biên (P3 theo biên chung 0.001 ở mục 8), thêm P7, P6 có tiêu chí số; sửa 10.8 → 9.9 điểm với cf72e0; thêm kiểm tra tin cậy (Q_u hữu hạn, hàng chưa chạm sau trộn, bất biến d_u, n_hit) và ghi nhận lần chạy bị hủy; kế hoạch cài đặt chi tiết hơn; thứ Sáu 06/11. Không đổi lưới, ngưỡng, seed, lr_item |
| 2026-11-03 | 1.2 | Sau khi cài đặt (`qu_grid`/`qu_diag` trong `reeval.py`, `gate_stats.py`): thêm `--grid-json` vào lệnh kiểm định (mục 13); thay ước lượng chi phí bằng số đo trên dữ liệu tổng hợp (mục 11, Tóm tắt). Không đổi lưới, ngưỡng, kiểm định, seed, lr_item, dự đoán |
| 2026-11-04 | 1.3 | Đã rút theo quyết định của GVHD ngày 04/11: ý tưởng bảng phim riêng Q_u chuyển sang phần hướng phát triển; không chạy lưới 8 ô, không áp dụng ngưỡng mục 7. Chỉ đổi trạng thái ở đầu README; các mục 1–13 giữ nguyên làm hồ sơ đăng ký trước. `qu_diag` (mục 9) được dùng lại cho việc chấm lại PFedRec có post-tuning |
