# CVAT local + Colab GPU

Chạy CVAT bằng Docker trên máy cá nhân và dùng GPU Colab để chạy segmentation.
Notebook có ô nhập repo/checkpoint, **không cố định model**. Mỗi người tự dùng tài khoản
Colab, ngrok và token riêng. Thư mục này không chứa bộ cài CVAT đầy đủ.

## Chuẩn bị và vị trí thư mục

Cần có CVAT Docker hoạt động, Docker Desktop dùng Linux containers, tài khoản Google
được cấp GPU Colab và tài khoản [ngrok](https://dashboard.ngrok.com/).

Đặt `colab-gpu` ngang hàng với `docker-compose.yml` của CVAT:

```text
D:\cvat\
├── docker-compose.yml
└── colab-gpu\
    ├── CVAT_Colab_GPU_Repo_Host.ipynb
    ├── docker-compose.colab.yml
    ├── bridge.py
    ├── runtime.py
    ├── build_notebook.py
    ├── test_gateway.py
    ├── README.md
    ├── .gitignore
    └── colab.env                 # Tạo khi chạy, không đưa lên GitHub
```

Các lệnh PowerShell dưới đây chạy từ `D:\cvat`. Đổi đường dẫn nếu bạn cài CVAT ở nơi khác.
Nếu tải bộ công cụ từ repo GitHub riêng, vẫn đặt các file vào thư mục `colab-gpu` như trên.

Mở [CVAT_Colab_GPU_Repo_Host.ipynb](CVAT_Colab_GPU_Repo_Host.ipynb) bằng
Colab: **File → Upload notebook**. Notebook chứa toàn bộ gateway, không cần upload các file Python.
Hướng dẫn, ô nhập repo, thử ảnh, tunnel và lệnh Docker đều nằm trong notebook.

## Cấu hình EoMT Small

Repo bạn đưa dùng được cho **segmentation tự động**, không phải tương tác bấm điểm kiểu SAM.
Điền cấu hình ví dụ sau tại cell 2; notebook không điền sẵn hoặc tự chọn model:

| Trường | Giá trị |
| --- | --- |
| `BACKEND` | `transformers` |
| `SOURCE_REPO` | `https://github.com/tue-mps/EoMT` |
| `SOURCE_REVISION` | Để trống |
| `MODEL_REPO` | `tue-mps/coco_panoptic_eomt_small_640_2x` |
| `MODEL_REVISION` | `main` |
| `MODEL_CLASS` | `EomtForUniversalSegmentation` |
| `SEGMENTATION_TASK` | `panoptic` |
| `FUNCTION_NAME` | `eomt-small` |
| `THRESHOLD` | `0.5` |
| `MIN_MASK_AREA` | `16` |
| `FUNCTION_YAML` | Để trống |
| `ADAPTER_FILE` | `cvat_adapter.py` (không dùng trong chế độ Transformers) |
| `WORKING_DIR` | Để trống |
| `EXTRA_CONFIG_JSON` | `{}` |

Đây là EoMT-S DINOv2, đầu vào 640×640, checkpoint panoptic cho vật thể và vùng nền.
Cấu hình này đã chạy thử trên Tesla T4 trong phiên thực hành.
Phần đầu notebook còn ví dụ EoMT Large; dùng bảng Small ở README này để chạy bản nhẹ đã thử.

`SOURCE_REPO` clone mã nguồn để tham khảo hoặc dùng với custom adapter.
Trong chế độ `transformers`, inference dùng implementation Transformers và weights từ
`MODEL_REPO`, theo [hướng dẫn tác giả](https://github.com/tue-mps/EoMT).
`MODEL_CLASS` có thể để trống để đọc từ config checkpoint.
Ví dụ DINOv2 này được tài liệu Transformers hỗ trợ; GPU thực tế còn phụ thuộc Colab cấp máy nào.
Không cần cài requirements training của EoMT. Nhãn dự đoán giới hạn theo checkpoint, ví dụ COCO.

## Chạy lần đầu

1. Bật Docker Desktop, kiểm tra CVAT local mở được.
2. Upload notebook lên Colab, chọn GPU runtime.
3. Chạy cell 1, điền cell 2; cell 3 để trống với cấu hình EoMT trên.
4. Chạy cell 4–6 để nạp model và kiểm tra mask trên một ảnh.
5. Chạy cell 7, nhập ngrok authtoken qua ô ẩn, tải `colab.env`.
6. Đặt file tại `D:\cvat\colab-gpu\colab.env` rồi chạy các lệnh cell 8.
7. Trong CVAT Tasks, chạy Automatic annotation, chọn model và map nhãn.
8. Khi dùng xong, tick `STOP_SESSION` tại cell 9 và chạy cell đó để dừng host.

Ô 5 thành công sẽ hiện `Ready: eomt-small`. Ô 6 phải trả được ảnh phủ mask trước khi nối Docker.
Ô 7 lấy ngrok authtoken qua ô nhập ẩn hoặc Colab Secrets tên `NGROK_AUTHTOKEN`.
**Ngrok authtoken** dùng đăng nhập ngrok; **COLAB_TOKEN** do notebook sinh để bảo vệ API model.
Không dùng hai token này thay cho nhau.

### Cách A: Kết nối bằng file env

File `colab-gpu/colab.env` phải có hai dòng riêng theo định dạng dưới đây.
Các giá trị này chỉ là placeholder; dùng URL/token thực từ notebook của bạn:

```dotenv
COLAB_URL=https://YOUR-TUNNEL.ngrok-free.app
COLAB_TOKEN=YOUR_BRIDGE_TOKEN
```

Không có `$env:`, không thêm khoảng trắng quanh `=`, không dán ký tự `\n` thay cho xuống dòng thật.

Mở Docker Desktop, đợi Engine chạy rồi chạy trong PowerShell:

```powershell
cd D:\cvat
docker compose --env-file colab-gpu/colab.env -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml up -d
```

Nếu CVAT đang dùng file `.env` ở gốc repo, dùng cả hai để giữ cấu hình hiện có:

```powershell
docker compose --env-file .env --env-file colab-gpu/colab.env -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml up -d
```

Nếu đang dùng thêm compose override khác, giữ chúng và đặt override Colab cuối cùng.

### Cách B: In lệnh PowerShell, không tải file

Trong cell 7, thay đoạn từ `env_path = ...` đến hết cell bằng:

```python
print("\n=== Chạy trong PowerShell tại thư mục CVAT ===\n")
print(f"$env:COLAB_URL = '{COLAB_URL}'")
print(f"$env:COLAB_TOKEN = '{BRIDGE_TOKEN}'")
print("docker compose -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml up -d")
```

Chạy cell 7 rồi dán **đủ ba dòng output** vào cùng một cửa sổ PowerShell tại thư mục CVAT.
Không dán cú pháp `$env:...` vào file `.env`. Output chứa token, cần xóa trước khi chia sẻ notebook.
Hai biến chỉ tồn tại trong cửa sổ PowerShell đó.

Tránh trộn hai cách: biến môi trường PowerShell được ưu tiên hơn file env.
Mở cửa sổ PowerShell mới khi chuyển từ cách B về cách A để tránh dùng nhầm token cũ.

### Kiểm tra kết nối

```powershell
docker compose -f docker-compose.yml exec cvat_server python -c "import requests; r=requests.get('http://colab_bridge:8070/api/functions', timeout=30); print(r.status_code); print(r.text[:1500]); r.raise_for_status()"
```

Mong đợi `200` và nội dung có `eomt-small`. Đây là kiểm tra kết nối/danh sách model;
chạy Automatic annotation ở bước tiếp theo để kiểm tra toàn bộ luồng.

## Sử dụng model trong CVAT

1. Mở <http://localhost:8080>, vào Tasks và tạo một task thử.
2. Trong Labels chọn **From model +**, chọn EoMT và thêm nhãn cần dùng.
3. Ví dụ ảnh giao thông: chọn `person`, `car`, `bus`, `truck`, `motorcycle`, `bicycle`.
4. Upload vài ảnh, đặt tên và tạo task.
5. Tại task, chọn **Actions → Automatic annotation**.
6. Chọn `tue-mps/coco_panoptic_eomt_small_640_2x`, ghép nhãn cùng tên, ví dụ `car → car`.
7. Đặt threshold `0.5`, tắt **Convert masks to polygons** nếu muốn giữ mask.
8. Không bật xóa annotation cũ nếu muốn giữ dữ liệu đã có. Chạy rồi mở job để xem kết quả.
9. Kiểm tra và sửa mask khi cần; bấm **Save** sau khi chỉnh sửa.

**From model chỉ lấy nhãn; Automatic annotation mới chạy model tạo mask.**

## Dừng và sử dụng lại

Lưu annotation trước khi dừng. Với cách A, dừng cả CVAT và bridge bằng:

```powershell
docker compose --env-file colab-gpu/colab.env -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml stop
```

Với cách B, trong cửa sổ PowerShell còn hai biến môi trường:

```powershell
docker compose -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml stop
```

`docker compose stop` vẫn dừng CVAT, nhưng không dừng `colab_bridge` vì bridge thuộc file bổ sung.
Các lệnh `stop` giữ nguyên dữ liệu. **Không dùng `down -v`**, vì sẽ xóa volume dữ liệu.

Trên Colab, tick `STOP_SESSION` ở cell 9 rồi chạy, sau đó chọn
**Runtime → Disconnect and delete runtime** để giải phóng GPU.
Dừng Docker không tự dừng Colab và ngược lại.

### Lần sau dùng lại EoMT

1. Mở Docker Desktop.
2. Chạy lại notebook trên GPU Colab.
3. Lấy URL/token phiên mới ở cell 7, cập nhật file env hoặc biến PowerShell.
4. Chạy lại lệnh **`up -d` có override Colab** ở phần kết nối.

`docker compose start` chỉ bật container với cấu hình cũ, **không cập nhật URL/token mới**.
Task và annotation đã lưu nằm trên máy local, không mất khi Colab kết thúc phiên.

### Chỉ dùng CVAT thủ công, không dùng Colab

Nếu CVAT còn trỏ vào một phiên Colab đã tắt, bước tải danh sách model có thể chậm hoặc lỗi.
Quay về danh sách compose trước khi thêm Colab. Nếu ban đầu chỉ dùng compose gốc:

```powershell
docker compose -f docker-compose.yml up -d
```

Nếu ban đầu dùng Nuclio local:

```powershell
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

Giữ các override riêng khác nếu có. Các lệnh này không xóa dữ liệu annotation.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
| --- | --- |
| Không kết nối được `dockerDesktopLinuxEngine` | Mở Docker Desktop, đợi Linux engine sẵn sàng; kiểm tra `docker info` |
| Thiếu `COLAB_URL`/`COLAB_TOKEN` | Dùng đúng `--env-file` hoặc chạy đủ hai dòng `$env:...` trong cùng terminal |
| File env báo ký tự `$` không hợp lệ | Dùng `COLAB_URL=...`, không phải cú pháp PowerShell |
| `401` từ gateway | Token cũ/sai; cập nhật từ đúng phiên và chạy `up -d` |
| `502` hoặc timeout | Kiểm tra Colab còn chạy, tunnel và URL mới; thử lại lệnh kiểm tra kết nối |
| `500` khi inference | Đọc `/content/cvat-model.log` trên Colab |
| CUDA out of memory | Dùng checkpoint nhỏ hơn hoặc input phù hợp với model |
| `No deployed models found` | Kiểm tra kết nối trước, tải lại CVAT một lần; nếu còn lỗi xem F12 Console/Network |
| Cả trang CVAT quay mãi | Kiểm tra API, phiên Colab và F12 Console/Network; tránh liên tục refresh hoặc restart tất cả |
| Không có mask | Kiểm tra threshold, ảnh thử, lớp checkpoint đã học và mapping nhãn task |
| Cảnh báo orphan container `nuclio` | Nuclio cũ nằm ngoài danh sách compose hiện tại; riêng cảnh báo này không có nghĩa bridge lỗi |


## Đổi repo/model

Notebook có chế độ `nuclio` cho repo chứa CVAT handler và chế độ `custom` cho adapter riêng.
URL repo bất kỳ không đủ để biết cách nạp weights hay chuyển output; xem hợp đồng adapter trong notebook.
Đổi backend/thư viện: nên dùng runtime mới. Đổi phiên: cập nhật URL/token trong env và recreate bridge.

## Kết nối

`CVAT Docker → colab_bridge (nội bộ Docker) → HTTPS/ngrok → GPU Colab`.
Không cần expose CVAT hoặc mở port inbound trên Windows. Bridge chỉ nằm trong mạng Docker CVAT.
API Colab yêu cầu bearer token ngẫu nhiên; token lưu trong env đã được gitignore.
Không đưa `colab.env` vào Git hoặc chia sẻ notebook có output chứa bí mật.
Model chỉ load một lần; gateway trả 503 nếu có inference khác đang chạy.
Override chuyển danh sách model sang Colab, không gộp với Nuclio local.

Colab là phiên tính toán tạm thời, không đảm bảo uptime/GPU.
Gói miễn phí có hạn chế dùng chủ yếu qua giao diện web ngoài notebook;
xem [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Kiểm tra và giới hạn

Kiểm tra offline:

```powershell
python colab-gpu/build_notebook.py
python -m unittest discover -s colab-gpu -p test_*.py -v
```

Các bài kiểm tra xác minh notebook, mask, gateway và relay bằng model giả, không tải weights.
Kiểm tra local cần `numpy`, `Pillow`, `fastapi`, `uvicorn`, `httpx` và `nbformat`.
Chạy inference GPU và kiểm tra CVAT UI cần hoàn thành các bước Colab/Docker ở trên.
`runtime.py` là nguồn gateway; chạy builder sau khi sửa để notebook nhận thay đổi.
Không có thay đổi vào mã nguồn CVAT hay dữ liệu annotation.

Tài liệu: [EoMT inference](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/eomt),
[EoMT Small checkpoint](https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x),
[CVAT serverless](https://docs.cvat.ai/docs/guides/serverless-tutorial/),
[pyngrok](https://pyngrok.readthedocs.io/en/latest/).
