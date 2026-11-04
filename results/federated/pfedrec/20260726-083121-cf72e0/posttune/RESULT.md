# Kết quả chấm lại PFedRec có post-tuning — cách (a)

**Trạng thái: ĐÃ CHẠY, 04/11/2026.** Điền theo mẫu tạo cùng README phiên bản 1 (04/11/2026); không thêm chỉ số, ô
hay tiêu chí sau khi xem số liệu. Nguồn số: `reeval_qu_diag_best.json` (`metrics_by_cell`, `trust_checks`),
`stats_wilcoxon.json` (`cells`, `contrasts`). Chạy bằng wrapper `.bug-investigation/run_pfedrec_posttune_a.sh`
qua `scripts/remote/start_run.sh`, CPU, log `posttune-a`.

| Hạng mục | Giá trị |
|---|---|
| Ngày chạy | 2026-11-04, 21:57–22:02 |
| Commit mã nguồn (máy chạy) | `d30ce88` (= `master` sau PR #16, #17) |
| Commit README được dùng | `67835a2` (README v1.1, merged in PR #16) |
| sha256 `global_state_best.pt` | `279459ba72987a98dfd1065c59ab0bb32f0e6f33323c08ccbc25211103b41c57` (khớp) |
| Cổng tin cậy `cached` | PASS, max\|Δ\| = 0.0 (NDCG/HR toàn bộ và 3 nhóm trùng số của run; 6040/809/2322/2909 user) |
| Kiểm tra tin cậy mục 7 (`trust_checks.status`) | PASS: `leaky_monotonicity_violations` = {E1: 0, E3: 0}; `n_hit.leaky` = 0; `n_hit.clean` = 3618; Q_u không NaN/Inf; hàng chưa chạm trùng bit (50 user); bất biến d_u: 0 vi phạm ở 4 ô; lỗi danh sách ứng viên: 0; lr_item = 29 648 |
| Thời gian: 200 user / toàn bộ | không ghi mốc riêng cho bước dò (log không in) / 292.3 s cho 4 ô × 6040 user (≈ 20.7 user/s) + 3.8 s chấm lại mốc; bước 2 từ 21:57:03 đến 22:02:01 |
| Trung vị `mean_row_shift` so với trung vị ‖q_i‖ | 16.32 (E1) và 13.55 (E3) so với 12.28 → tỉ số ≈ 1.3 / 1.1: **không** phải kéo mạnh (≫ 10×) theo định nghĩa mục 8, cũng không phải kéo yếu (≪ 1) |
| Lần chạy bị hủy | 0 (lần đầu `start_run.sh` dừng trước khi chạy vì nhánh `master` trên máy chưa có upstream; đặt `--set-upstream-to` rồi chạy lại, không có ô nào được tính) |

## 1. Bốn ô (NDCG@10 / HR@10 / MRR, toàn bộ và theo nhóm)

| Ô | Toàn bộ | sparse (809) | medium (2322) | dense (2909) |
|---|---|---|---|---|
| Mốc: bảng phim chung | 0.3447118 / 0.6110927 / 0.2821309 | 0.4269673 / 0.7058096 / 0.3551956 | 0.3801614 / 0.6688200 / 0.3082912 | 0.2935401 / 0.5386731 / 0.2409299 |
| `a0.00_E1_clean` (dòng 2 luận văn) | 0.3181 / 0.5584 / 0.2608 | 0.4165 / 0.6848 / 0.3474 | 0.3611 / 0.6326 / 0.2937 | 0.2564 / 0.4641 / 0.2105 |
| `a0.00_E1_leaky` (dòng 3 luận văn) | 0.3789 / 0.6642 / 0.3090 | 0.4305 / 0.7108 / 0.3583 | 0.3891 / 0.6792 / 0.3165 | 0.3564 / 0.6394 / 0.2893 |
| `a0.00_E3_clean` (phụ lục) | 0.2762 / 0.4704 / 0.2305 | 0.3981 / 0.6539 / 0.3329 | 0.3347 / 0.5792 / 0.2742 | 0.1956 / 0.3324 / 0.1672 |
| `a0.00_E3_leaky` (phụ lục) | 0.4520 / 0.7505 / 0.3738 | 0.4385 / 0.7219 / 0.3649 | 0.4128 / 0.7063 / 0.3381 | 0.4871 / 0.7937 / 0.4047 |

## 2. Wilcoxon theo cặp so với mốc (NDCG@10, toàn bộ)

| Ô | Δ trung bình | CI95 bootstrap | n≠0 | p một phía (ô > mốc) | p hai phía | p Holm | r rank-biserial |
|---|---|---|---|---|---|---|---|
| `a0.00_E1_clean` | −0.02664 | [−0.03153, −0.02173] | 1495 | 1.0 | 9.5e−4 | 1.0 | −0.099 |
| `a0.00_E1_leaky` | +0.03421 | [+0.03184, +0.03661] | 1329 | 2.9e−219 | 5.8e−219 | 8.7e−219 | +1.000 |
| `a0.00_E3_clean` | −0.06850 | [−0.07570, −0.06131] | 2441 | 1.0 | 3.0e−53 | 1.0 | −0.359 |
| `a0.00_E3_leaky` | +0.10733 | [+0.10256, +0.11218] | 2502 | < 1e−300 | < 1e−300 | < 1e−300 | +1.000 |

Theo nhóm (Δ NDCG@10, p hai phía): `E1_clean` sparse −0.0105 (0.090), medium −0.0191 (0.72), dense −0.0371 (0.0017);
`E3_clean` sparse −0.0289 (0.059), medium −0.0454 (5.7e−4), dense −0.0979 (2.0e−50);
`E1_leaky` sparse +0.0036 (5.2e−7), medium +0.0090 (8.5e−45), dense +0.0629 (7.9e−171);
`E3_leaky` sparse +0.0115 (7.8e−18), medium +0.0326 (9.3e−105), dense +0.1936 (1.2e−291).
r = +1.000 ở hai ô `leaky` là hệ quả của kiểm tra đơn điệu (không user nào giảm).

## 3. Đối chiếu leaky − clean (NDCG@10)

| Đối chiếu | Δ toàn bộ | CI95 | p một phía | Δ sparse | Δ medium | Δ dense |
|---|---|---|---|---|---|---|
| E = 1: leaky − clean | +0.06085 | [+0.05583, +0.06578] | 3.1e−107 (n≠0 = 639) | +0.0141 | +0.0281 | +0.1000 |
| E = 3: leaky − clean | +0.17583 | [+0.16774, +0.18384] | 1.9e−281 (n≠0 = 1694) | +0.0404 | +0.0781 | +0.2915 |

Ghi chú cơ chế: hai quỹ đạo `clean` và `leaky` chỉ khác nhau ở những lượt rút trúng phim test, nên hiệu leaky − clean
khác 0 ở đúng 639 user (E = 1) và 1694 user (E = 3). Δ toàn bộ +0.061 dồn vào 639 user tương ứng ≈ +0.58 NDCG mỗi user
bị ảnh hưởng: khi phim test bị rút làm mẫu âm dù chỉ một lần, NDCG của user đó gần như về 0.

## 4. Đối chiếu dự đoán A1–A5 (README mục 8)

| # | Dự đoán | Kết quả | Đúng / Sai / Không phân biệt |
|---|---|---|---|
| A1 | clean E1 < mốc; kéo mạnh ≈ 0.321 hoặc kéo yếu Δ ∈ [−0.005, 0]; E3 < E1 | clean E1 = 0.3181 (Δ −0.0266; dự đoán kéo mạnh −0.024); E3 = 0.2762 (Δ −0.0685; dự đoán −0.061) < E1. Chỉ số `mean_row_shift` lại không ở mức "kéo mạnh" (≈ 1.3× ‖q_i‖) | **Đúng** về chiều và về độ lớn (sát kịch bản kéo mạnh); nhận diện kịch bản bằng `mean_row_shift` không phân biệt được |
| A2 | leaky E1: Δ ≥ +0.010, p < 0.05; E3 ≥ E1 | Δ +0.0342, p = 2.9e−219; E3 (0.4520) > E1 (0.3789) | **Đúng** |
| A3 | leaky − clean ≥ +0.010, p < 0.05, ở cả E1 và E3 | +0.0609 (p 3.1e−107) và +0.1758 (p 1.9e−281) | **Đúng** |
| A4 | dense có \|Δ\| lớn nhất ở cả 4 ô | E1_clean 0.037 > 0.019 > 0.011; E3_clean 0.098 > 0.045 > 0.029; E1_leaky 0.063 > 0.009 > 0.004; E3_leaky 0.194 > 0.033 > 0.012 | **Đúng** |
| A5 | không ô nào ≥ 0.394 | `a0.00_E3_leaky` = 0.4520 ≥ 0.394, và vượt cả số công bố 0.4436; `E1_leaky` = 0.3789 < 0.394 | **Sai** (ở E = 3) |

## 5. Câu cho phần hạn chế (chọn theo quy tắc mục 5 README, không viết lại)

- [x] Δ(dòng 3 − dòng 2) = +0.0609 ≥ +0.010 và p = 3.1e−107 < 0.05 → "một phần khoảng cách tới số công bố đến từ việc
  code gốc loại phim test khỏi mẫu âm khi post-tuning".
- [ ] ngược lại → (không áp dụng).

Bảng PFedRec trong luận văn theo mục 5 README: dòng 1 = 0.3447 (bảng phim chung), dòng 2 = 0.3181 (post-tuning, mẫu âm
sạch), dòng 3 = 0.3789 (post-tuning, mẫu âm theo code gốc). Dòng 2 là số được gọi là "PFedRec có post-tuning".
Hai ô E = 3 ở phụ lục; `E3_leaky` = 0.4520 chỉ dùng để minh họa rằng giao thức post-tuning của code gốc, với 3 epoch,
tái tạo và vượt số công bố ngay ở C = 0.1 — không trích làm kết quả phương pháp.

Người điền: Claude (từ hai JSON), kiểm tra lại: Dang Vinh | Ngày: 2026-11-04 | Commit kết quả: `455b12d`
