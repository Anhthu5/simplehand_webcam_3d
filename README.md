# Ước lượng và tái tạo bàn tay 3D theo thời gian thực

Ứng dụng webcam trên Windows dựa trên mô hình [SimpleHand (CVPR 2024)](https://github.com/patienceFromZhou/simpleHand) đã được huấn luyện sẵn. MediaPipe tìm vùng bàn tay, SimpleHand dự đoán lưới bàn tay 3D (mesh), sau đó OpenGL hiển thị bề mặt để người dùng xoay và quan sát.

## Chức năng chính

- Sử dụng backbone FastViT-MA36 và bộ giải mã mesh chính thức của SimpleHand.
- Dự đoán 778 đỉnh mesh, ghép thành 1.538 tam giác và suy ra 21 khớp 3D.
- Hiển thị bề mặt có ánh sáng, hỗ trợ xoay, phóng to và xem lưới tam giác.
- Xuất OBJ/NPZ, lưu ảnh đầu vào và ảnh render, ghi video MP4 gồm webcam và mesh cạnh nhau.
- Thống kê thời gian xử lý và FPS từng phiên, tách riêng các khung có suy luận.
- Hỗ trợ suy luận trên FreiHAND và tính PA-MPJPE, PA-MPVPE.

Phần triển khai của dự án gồm xử lý vùng tay từ webcam, nạp trọng số, đổi hệ tọa độ, hiển thị mesh, xuất kết quả, ghi hình và kiểm thử. Kiến trúc mô hình và trọng số huấn luyện được kế thừa từ SimpleHand. Xem [thông tin nguồn](NOTICE.md).

## Quy trình xử lý

```text
Webcam / video
  -> MediaPipe tìm vùng bàn tay
  -> crop vuông, resize về 224 x 224
  -> FastViT-MA36 trích xuất đặc trưng và dự đoán 21 khớp 2D
  -> lấy đặc trưng tại khớp để tạo 21 token
  -> mở rộng 21 -> 84 -> 336 -> 778 đỉnh mesh
  -> hiển thị bề mặt bàn tay 3D và suy ra 21 khớp 3D
```

Khi không tìm thấy tay, chương trình bỏ qua suy luận và xóa mesh của khung hình trước.

## Cài đặt trên Windows

Chuẩn bị Python 3.10 x64, Git, VSCode và GPU NVIDIA có driver tương thích. Bộ phiên bản CUDA được cố định để phù hợp mã nguồn gốc và đã được kiểm tra trên RTX 3050 Laptop GPU. GPU thuộc kiến trúc mới hơn có thể cần phiên bản PyTorch mới hơn.

Mở thư mục dự án bằng VSCode, sau đó chạy lần lượt trong Terminal PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_upstream.ps1
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Trong VSCode, dùng **Python: Select Interpreter** để chọn `.venv\Scripts\python.exe`. Kết quả kiểm tra môi trường cần có `CUDA available: True`. Chỉ cài gói OpenCV đã chỉ định trong requirements để tránh xung đột.

## Tải trọng số đã huấn luyện

Làm theo [hướng dẫn của tác giả](https://github.com/patienceFromZhou/simpleHand#training). Tải checkpoint `epoch_200_rerun1` hoặc `epoch_200_rerun2` từ [thư mục tải](https://drive.google.com/drive/folders/1BfHjNjxQj3MdsGoq5irCrOskyCA9a64l?usp=drive_link), rồi đặt file checkpoint thực tại:

```text
weights/simplehand_fastvit_ma36.pth
```

File gốc có thể không có phần mở rộng. Đổi tên file không chuyển đổi nội dung của nó. Cần dùng checkpoint SimpleHand đầy đủ; trọng số FastViT chỉ huấn luyện trên ImageNet không chứa phần dự đoán bàn tay. Chương trình kiểm tra toàn bộ tham số bằng `strict=True` khi nạp.

Script cài đặt tải repo chính thức tại phiên bản cố định và tài nguyên `MANO_RIGHT_C.pkl` vào `third_party/simpleHand`. Checkpoint, dataset và tài nguyên MANO không được đóng gói trong repository này. Tuân thủ điều khoản của từng nguồn và chỉ nạp checkpoint/pickle từ nguồn đáng tin cậy.

## Chạy chương trình

```powershell
.\.venv\Scripts\python.exe main.py --device cuda --camera 0
```

Chương trình mở hai cửa sổ: ảnh camera đầu vào và bề mặt bàn tay 3D dự đoán.

| Thao tác | Chức năng |
|---|---|
| Kéo chuột trái | Xoay mesh |
| Lăn chuột | Phóng to hoặc thu nhỏ |
| W | Bật/tắt lưới tam giác |
| R | Đặt lại góc nhìn |
| S | Lưu OBJ, NPZ, ảnh đầu vào JPG và ảnh render PNG |
| Q / Esc | Thoát |

Một số lệnh tùy chọn:

```powershell
# Ghi webcam và mesh cạnh nhau; dùng tên file mới cho mỗi lần chạy.
.\.venv\Scripts\python.exe main.py --device cuda --record captures/demo_01.mp4

# Xử lý video có sẵn.
.\.venv\Scripts\python.exe main.py --video hand_video.mp4 --device cuda

# Thử FP16, hiển thị camera như gương và thêm khung xương 2D.
.\.venv\Scripts\python.exe main.py --device cuda --amp --mirror --show-joints
```

Nếu không mở được camera, thử `--backend msmf` hoặc `--camera 1`. Bỏ `--amp` nếu dự đoán FP16 không ổn định. `--canonicalize-left` là tùy chọn thử nghiệm lật crop tay trái trước khi suy luận.

Kết quả được lưu trong `captures/`. Mỗi phiên hoàn tất tạo một file JSON thống kê thời gian. FPS mã hóa MP4 khác FPS xử lý đo được; ghi video cũng làm tăng thời gian xử lý. Khi đo tốc độ webcam, cần có bàn tay hiện diện và ghi rõ chế độ ghi hình, độ chính xác số học đi kèm kết quả.

## Đánh giá và kiểm thử

Với FreiHAND, chuẩn bị các cặp ảnh/JSON theo hướng dẫn tác giả và một file JSON chứa danh sách đường dẫn ảnh. Mỗi annotation cần có `uv` (21,2), `xyz` (21,3), `vertices` (778,3) và thông số camera `K` (3,3). Dùng nhãn chuẩn thật, có cùng đơn vị và thứ tự điểm với dự đoán.

```powershell
.\.venv\Scripts\python.exe infer_freihand.py --images D:/datasets/FreiHAND/eval_windows.json --data-root D:/datasets/FreiHAND --limit 100 --output captures/freihand_predictions.npz
.\.venv\Scripts\python.exe evaluate.py captures/freihand_predictions.npz --unit m --output-json captures/freihand_metrics.json
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Thay đường dẫn dataset theo máy đang dùng. Bỏ `--limit` để xử lý toàn bộ danh sách. Quy trình đánh giá dùng crop có hiệu chỉnh phối cảnh và phép xoay 3D ngược theo mã chính thức, không dùng tăng cường dữ liệu khi suy luận (TTA).

Các chỉ số PA căn chỉnh tỉ lệ, tịnh tiến và hướng cho từng mẫu trước khi tính sai số. Phép căn chỉnh trực giao mặc định theo mã nguồn gốc có thể cho phép phản chiếu; `--proper-rotation` loại bỏ phản chiếu. `--unit m` nghĩa là cả tọa độ dự đoán và nhãn chuẩn đều ở đơn vị mét. Sai số cuối cùng được báo theo milimét.

Các kiểm thử phần mềm bao gồm hình học crop, căn chỉnh PA, tính pháp tuyến và xuất mesh, xử lý mất tay, ghi video và thống kê. Việc nạp checkpoint thật và hiển thị mesh đã được kiểm tra trên Windows/CUDA. Độ chính xác trên toàn bộ FreiHAND và tốc độ webcam khi hiển thị bề mặt cần được đo riêng; repository chưa công bố điểm benchmark.

## Giới hạn

- Xử lý một bàn tay trong mỗi lần suy luận. Che khuất, góc nhìn bất thường và đầu vào tay trái có thể làm giảm độ chính xác.
- Mesh dùng tọa độ tương đối với khớp gốc và được phóng to đều để hiển thị; không đo khoảng cách tuyệt đối đến webcam.
- Màu bề mặt phục vụ hiển thị hình học, không phải màu da được tái tạo.
- Crop webcam chỉ là phép xấp xỉ so với crop có hiệu chuẩn camera trong benchmark.

## Cấu trúc mã nguồn

```text
main.py                 Ứng dụng webcam/video
models/                 Mô hình tương thích checkpoint và decoder gốc có ghi nguồn
utils/                  Vùng tay, hình học, suy luận, hiển thị và ghi hình
scripts/                Tải tài nguyên gốc và kiểm tra môi trường
tests/                  Kiểm thử tự động
weights/README.md       Hướng dẫn đặt checkpoint
infer_freihand.py        Suy luận các mẫu đánh giá có nhãn
evaluate.py             Tính chỉ số PA và xuất JSON
```

## Tài liệu tham khảo và giấy phép

- Zhou và cộng sự, [A Simple Baseline for Efficient Hand Mesh Reconstruction](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_A_Simple_Baseline_for_Efficient_Hand_Mesh_Reconstruction_CVPR_2024_paper.pdf), CVPR 2024.
- [Mã nguồn SimpleHand chính thức](https://github.com/patienceFromZhou/simpleHand), phiên bản cố định được ghi trong [NOTICE.md](NOTICE.md).
- [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/).

Mã ứng dụng được phát hành theo [giấy phép MIT](LICENSE). Giấy phép và thông tin tác giả của decoder gốc được giữ trong `models/upstream/`. Các tài nguyên mô hình và dataset bên thứ ba tuân theo điều khoản riêng của nguồn cung cấp.
