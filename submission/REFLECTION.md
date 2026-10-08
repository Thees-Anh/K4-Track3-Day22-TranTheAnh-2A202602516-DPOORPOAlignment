# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trần Thế Anh

**Mã học viên:** 2A202602516

**Khoá:** K4

**Tier đã chạy:** T4
**Ngày:** 2026-10-08

Mọi số liệu dưới đây được lấy trực tiếp từ `adapters/dpo/dpo_metrics.json`, `data/pref/stats.json`, `data/eval/judge_summary.json` và `data/eval/side_by_side.jsonl`.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Google Colab Tesla T4, 14.563 GB khả dụng theo log Unsloth |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, 800 cặp train / 100 cặp held-out tiếng Việt |
| Chosen dài hơn rejected (NB2) | 65,875% (median: chosen 94 token, rejected 86 token) |
| DPO: β / tốc độ học / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | RM panel Skywork; Qwen3-4B sanity 41,67%, Llama-3.2-3B sanity 100%; chỉ Llama đạt ngưỡng 80% và được giữ trong panel |
| Chi phí | 0 đồng, Google Colab T4 miễn phí |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được ghi trong file metrics xuất về |
| VRAM cao nhất | Không được ghi; GPU có 14.563 GB khả dụng |
| Loss train đầu → cuối | 0,6952 → 0,6751 |
| Reward chosen / rejected cuối trên train | 0,3719 / 0,2790 |
| Reward gap cuối trên train | 0,0929 |
| Reward chosen / rejected trên held-out | 0,3904 / 0,3057 |
| Độ chính xác reward trên held-out | 65% |
| Margin trên held-out | 0,0846 |
| Chẩn đoán tự động | `INTENDED` |
| Độ dài trung bình SFT → DPO (NB4, overall) | 424,95 → 431,84 ký tự |

---

## 3. Đọc đường reward

Ảnh `screenshots/03-dpo-reward-curves.png` và metrics cho thấy trên tập train, reward của chosen kết thúc ở 0,3719 còn rejected ở 0,2790, tạo margin dương 0,0929. Trên held-out, chosen đạt 0,3904 và rejected đạt 0,3057, nên margin vẫn dương 0,0846 và reward accuracy đạt 65%. Hai phía cùng tăng, nhưng chosen tăng mạnh hơn rejected; vì vậy trường hợp này phù hợp với chẩn đoán `INTENDED`, không phải likelihood displacement. Held-out đi cùng hướng với train, dù khoảng cách nhỏ hơn và accuracy 65% cho thấy khả năng phân biệt chưa thật mạnh. Tôi cũng không coi margin dương là bằng chứng duy nhất về chất lượng, vì NB4 còn cho thấy khoảng tin cậy win rate chứa 0,5.

Về mặt công thức, margin vẫn có thể tăng khi xác suất chosen giảm nếu xác suất rejected giảm nhanh hơn: DPO tối ưu chênh lệch log-ratio chứ không bắt buộc likelihood tuyệt đối của chosen phải tăng. Ngoài ra, 65,875% cặp train có chosen dài hơn rejected nên DPO gốc có nguy cơ học thiên lệch độ dài. Các loss chuẩn hoá theo token như SimPO/ORPO giảm ảnh hưởng của tổng log-prob vốn âm hơn đối với câu dài.

---

## 4. So sánh SFT và SFT+DPO

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp gần bằng độ dài | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 13 | 8 | 29 | 0,550 [0,460; 0,630] | 0,531 (n=48) | 0,571 |
| hữu ích — helpfulness | 4 | 2 | 0 | 2 | 0,750 [0,500; 1,000] | 0,750 (n=4) | 0,500 |
| an toàn — safety | 4 | 0 | 2 | 2 | 0,250 [0,000; 0,500] | 0,250 (n=4) | 0,500 |

Giám khảo cuối là `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B`, sanity accuracy 100%. Khoảng tin cậy held-out [0,46; 0,63] chứa 0,5, nên chưa đủ bằng chứng thống kê để kết luận DPO tốt hơn SFT. DPO chỉ dài hơn nhẹ (417,64 so với 411,28 ký tự trên held-out); longer-answer-won là 57,14% và win rate sau khi ghép cặp gần bằng độ dài còn 53,125%. Vì vậy độ dài có thể đóng góp một phần nhưng không giải thích toàn bộ kết quả.

Hai RM không hoàn toàn nhất quán: Qwen cho DPO win rate 45%, Llama cho 55%, agreement 79,31%. Qwen chỉ đạt sanity 41,67% nên bị loại khỏi panel; do đó kết luận thực tế dựa vào một giám khảo Llama dù file vẫn báo kết quả từng RM. Đây là hạn chế quan trọng, và không có dấu hiệu Qwen thiên vị DPO cao hơn Llama trong lần chạy này.

Ví dụ hữu ích `h2` hỏi ba món từ gạo và trứng: DPO thắng về độ đa dạng so với SFT, vì SFT lặp lại món bánh mì–trứng, nhưng cả hai vẫn đề xuất thêm nguyên liệu không có trong yêu cầu nên chất lượng tuyệt đối còn hạn chế. Ví dụ an toàn `s3` hỏi trẻ 14 tuổi mua rượu: cả hai đều từ chối và hướng tới lựa chọn an toàn; bản SFT nói rõ yếu tố bất hợp pháp và khuyên tìm người lớn đáng tin cậy, trong khi DPO khái quát hơn. Điều này phù hợp với việc nhóm safety không cải thiện và DPO chỉ đạt win rate 0,25 trên bốn câu cố định.

---

## 5. Đánh đổi theo β (bonus)

Không chạy β-sweep. Tôi dự đoán β=0,05 cho phép policy đi xa reference hơn nên margin có thể tăng nhưng nguy cơ lệch likelihood và overfit cao hơn. β=0,5 sẽ giữ policy gần SFT hơn, thường cho margin nhỏ và thay đổi hành vi ít hơn. Cần chạy nhiều seed trước khi kết luận vì tập đánh giá chỉ có 50 câu held-out.

---

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng nhất của tôi là giữ β=0,1 và learning rate 5e-6 cho lần chạy DPO bắt buộc, thay vì giảm β xuống 0,05 để tạo thay đổi mạnh hơn hoặc tăng lên 0,5 để bám sát SFT hơn. Tôi chọn cấu hình này vì đây là mức cân bằng phù hợp với T4 và chỉ có 800 cặp preference: mô hình cần học được tín hiệu chosen/rejected nhưng không nên đi quá xa mô hình SFT vốn là reference. Kết quả phần nào xác nhận lựa chọn đó: loss giảm từ 0,6952 xuống 0,6751; margin cuối dương trên cả train (0,0929) và held-out (0,0846); accuracy held-out đạt 65%; chẩn đoán là `INTENDED`. Tuy nhiên NB4 làm tôi thận trọng hơn: win rate held-out chỉ 0,55 và CI 95% [0,46; 0,63] vẫn chứa 0,5. Nhóm safety còn giảm xuống win rate 0,25, trong khi giám khảo Qwen không vượt qua sanity check. Nếu làm lại, tôi sẽ giữ nguyên pipeline nhưng chạy β-sweep với ít nhất ba seed, bổ sung một giám khảo khác họ có sanity tiếng Việt tốt, và so sánh trên nhiều prompt safety hơn. Tôi cũng sẽ thử RPO hoặc loss chuẩn hoá độ dài vì 65,875% chosen dài hơn rejected, nhằm giảm khả năng mô hình tối ưu độ dài thay vì chất lượng nội dung.

---

## Danh sách bonus

- [ ] NB3b — các biến thể loss (đã thử một phần nhưng ORPO bị dừng, không tính là hoàn thành)
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng API khác họ
- [ ] Đẩy adapter lên Hugging Face Hub

---

## Điều bất ngờ nhất

DPO tạo margin dương trên held-out nhưng đánh giá end-to-end vẫn chưa cho bằng chứng chắc chắn hơn SFT. Đặc biệt, hai reward model cho kết luận lệch nhau và một model không vượt sanity check, cho thấy chất lượng giám khảo quan trọng không kém thuật toán huấn luyện.
