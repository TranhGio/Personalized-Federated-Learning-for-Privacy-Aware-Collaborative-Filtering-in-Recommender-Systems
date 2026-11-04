# Kết quả chấm lại PFedRec có post-tuning — cách (a) (điền SAU khi chạy)

**Trạng thái: CHƯA CHẠY.** Mẫu này được tạo cùng README phiên bản 1 (04/11/2026). Chỉ điền vào các ô đã định sẵn;
không thêm chỉ số, ô hay tiêu chí sau khi xem số liệu. Nguồn số: `reeval_qu_diag_best.json` (`metrics_by_cell`,
`trust_checks`), `stats_wilcoxon.json` (`cells`, `contrasts`).

| Hạng mục | Giá trị |
|---|---|
| Ngày chạy | — |
| Commit mã nguồn (máy chạy) | — |
| Commit README được dùng | — |
| sha256 `global_state_best.pt` | — (phải = `279459ba…57`) |
| Cổng tin cậy `cached` | — (PASS, max\|Δ\|) |
| Kiểm tra tin cậy mục 7 (`trust_checks.status`) | — (PASS; `leaky_monotonicity_violations` phải toàn 0; `n_hit.leaky` = 0; `n_hit.clean` > 0) |
| Thời gian: 200 user / toàn bộ | — / — |
| Trung vị `mean_row_shift` so với trung vị ‖q_i‖ | — (kịch bản kéo mạnh / kéo yếu, mục 8) |
| Lần chạy bị hủy | — (số lần, lý do) |

## 1. Bốn ô (NDCG@10 / HR@10 / MRR, toàn bộ và theo nhóm)

| Ô | Toàn bộ | sparse (809) | medium (2322) | dense (2909) |
|---|---|---|---|---|
| Mốc: bảng phim chung | 0.3447118 / 0.6110927 / 0.2821309 | 0.4269673 / 0.7058096 / 0.3551956 | 0.3801614 / 0.6688200 / 0.3082912 | 0.2935401 / 0.5386731 / 0.2409299 |
| `a0.00_E1_clean` (dòng 2 luận văn) | — | — | — | — |
| `a0.00_E1_leaky` (dòng 3 luận văn) | — | — | — | — |
| `a0.00_E3_clean` (phụ lục) | — | — | — | — |
| `a0.00_E3_leaky` (phụ lục) | — | — | — | — |

## 2. Wilcoxon theo cặp so với mốc (NDCG@10, toàn bộ)

| Ô | Δ trung bình | CI95 bootstrap | n≠0 | p một phía (ô > mốc) | p hai phía | p Holm | r rank-biserial |
|---|---|---|---|---|---|---|---|
| `a0.00_E1_clean` | — | — | — | — | — | — | — |
| `a0.00_E1_leaky` | — | — | — | — | — | — | — |
| `a0.00_E3_clean` | — | — | — | — | — | — | — |
| `a0.00_E3_leaky` | — | — | — | — | — | — | — |

## 3. Đối chiếu leaky − clean (NDCG@10)

| Đối chiếu | Δ toàn bộ | CI95 | p một phía | Δ sparse | Δ medium | Δ dense |
|---|---|---|---|---|---|---|
| E = 1: leaky − clean | — | — | — | — | — | — |
| E = 3: leaky − clean | — | — | — | — | — | — |

## 4. Đối chiếu dự đoán A1–A5 (README mục 8)

| # | Dự đoán | Kết quả | Đúng / Sai / Không phân biệt |
|---|---|---|---|
| A1 | clean E1 < mốc; kéo mạnh ≈ 0.321 hoặc kéo yếu Δ ∈ [−0.005, 0]; E3 < E1 | — | — |
| A2 | leaky E1: Δ ≥ +0.010, p < 0.05; E3 ≥ E1 | — | — |
| A3 | leaky − clean ≥ +0.010, p < 0.05, ở cả E1 và E3 | — | — |
| A4 | dense có \|Δ\| lớn nhất ở cả 4 ô | — | — |
| A5 | không ô nào ≥ 0.394 | — | — |

## 5. Câu cho phần hạn chế (chọn theo quy tắc mục 5 README, không viết lại)

- [ ] Δ(dòng 3 − dòng 2) ≥ +0.010 và p < 0.05 → "một phần khoảng cách tới số công bố đến từ việc code gốc loại phim test khỏi mẫu âm khi post-tuning".
- [ ] ngược lại → "khác biệt về mẫu âm là không đáng kể; khoảng cách còn lại chưa tách được giữa C và các khác biệt K1–K3".

Người điền: — | Ngày: — | Commit kết quả: —
