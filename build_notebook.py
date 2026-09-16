"""Build the self-contained notebook from the reviewed gateway implementation."""

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
cells = []


def cell(kind, source):
    entry = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
             "source": dedent(source).strip() + "\n"}
    if kind == "code":
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell("markdown", """
# CVAT local + GPU Colab: repo tùy chọn

Luồng dữ liệu: **CVAT Docker trên Windows → bridge nội bộ → HTTPS tunnel → model trên GPU Colab**.
Không cần mở CVAT ra Internet. Ảnh được gửi sang Colab khi bạn gọi model.
Notebook độc lập, không cần upload file Python đi kèm. Mỗi phiên phục vụ một model.

## Chọn cách nạp model
* `transformers`: checkpoint Hugging Face với processor hỗ trợ `post_process_*_segmentation`.
  Có sẵn chuyển mask về CVAT. Repo/checkpoint do bạn nhập, không có model mặc định.
* `nuclio`: repo có `function.yaml`, `init_context` và `handler` Python theo CVAT.
  Bạn cần cài dependencies, tải weights và thiết lập đường dẫn mà handler yêu cầu.
  Đây là lớp tương thích tối thiểu, không phải đầy đủ Nuclio/Docker build.
* `custom`: repo bất kỳ có file adapter theo hợp đồng ở cuối notebook.

**EoMT dùng được cho automatic segmentation.** Repo GitHub chứa mã nguồn; checkpoint Hugging Face
chứa weights/config. Hai ô này có mục đích khác nhau. Chế độ Transformers chạy bản triển khai
Transformers được tác giả giới thiệu, không thực thi pipeline training của repo GitHub.
EoMT không nhận positive/negative clicks như SAM. Nhãn dự đoán phụ thuộc checkpoint đã huấn luyện.

Ví dụ để điền bằng tay cho EoMT DINOv2 (không tự áp dụng):
```
BACKEND = transformers
SOURCE_REPO = https://github.com/tue-mps/EoMT
MODEL_REPO = tue-mps/coco_instance_eomt_large_640
MODEL_CLASS = EomtForUniversalSegmentation
SEGMENTATION_TASK = instance
```
Có thể dùng `tue-mps/coco_panoptic_eomt_large_640` với `panoptic`, hoặc
`tue-mps/ade20k_semantic_eomt_large_512` với `semantic`. Chọn checkpoint khớp loại bài toán.
Bộ thư viện ghim dưới đây dành cho DINOv2, không cam kết chạy các checkpoint DINOv3 mới hơn.

## Trước khi chạy
1. Colab: **Runtime → Change runtime type → GPU**. Chạy từng cell theo thứ tự.
2. Chuẩn bị tài khoản ngrok và authtoken, nhập qua ô ẩn hoặc Colab Secrets `NGROK_AUTHTOKEN`.
3. Máy Windows: Docker Desktop đang chạy, CVAT local hoạt động.

Colab không phải host cố định: GPU/VRAM và thời lượng không được đảm bảo, runtime có thể bị ngắt.
Gói miễn phí không cho dùng chủ yếu qua web UI ngoài notebook; hãy kiểm tra điều kiện tài khoản
trong [Colab FAQ](https://research.google.com/colaboratory/faq.html).
Không có keep-alive hay cơ chế tránh giới hạn. Nếu cần máy chủ ổn định, dùng GPU VM phù hợp.

Nguồn: [EoMT](https://github.com/tue-mps/EoMT),
[EoMT Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/eomt),
[CVAT serverless](https://docs.cvat.ai/docs/guides/serverless-tutorial/).
""")

cell("code", """
#@title 1. Cài thư viện cầu nối và kiểm tra GPU
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.57.1", "fastapi==0.116.1", "uvicorn==0.35.0",
                "pyngrok==7.3.0", "PyYAML==6.0.2", "pillow", "numpy", "requests"], check=True)
import torch
assert torch.cuda.is_available(), "Chọn GPU runtime rồi chạy lại."
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
""")

cell("code", """
#@title 2. Nhập repo và cấu hình model (không có model mặc định)
BACKEND = "transformers" #@param ["transformers", "nuclio", "custom"]
SOURCE_REPO = "" #@param {type:"string"}
SOURCE_REVISION = "" #@param {type:"string"}
MODEL_REPO = "" #@param {type:"string"}
MODEL_REVISION = "main" #@param {type:"string"}
MODEL_CLASS = "" #@param {type:"string"}
SEGMENTATION_TASK = "instance" #@param ["instance", "panoptic", "semantic"]
FUNCTION_NAME = "colab-segmentation" #@param {type:"string"}
THRESHOLD = 0.5 #@param {type:"number"}
MIN_MASK_AREA = 16 #@param {type:"integer"}
# Hai đường dẫn dưới đây tính từ gốc SOURCE_REPO.
FUNCTION_YAML = "" #@param {type:"string"}
ADAPTER_FILE = "cvat_adapter.py" #@param {type:"string"}
WORKING_DIR = "" #@param {type:"string"}
EXTRA_CONFIG_JSON = "{}" #@param {type:"string"}

import hashlib, json, re
from pathlib import Path
from urllib.parse import urlsplit
assert re.fullmatch(r"[a-z0-9][a-z0-9.-]*", FUNCTION_NAME), "Tên function chỉ dùng a-z, 0-9, dấu chấm/gạch ngang."
assert 0 <= THRESHOLD <= 1 and MIN_MASK_AREA >= 1
assert BACKEND != "transformers" or MODEL_REPO.strip(), "Nhập MODEL_REPO (owner/checkpoint)."
assert BACKEND == "transformers" or SOURCE_REPO.strip(), "Nhập SOURCE_REPO HTTPS."
source_dir = Path("/content/model-source")
if SOURCE_REPO:
    parsed = urlsplit(SOURCE_REPO)
    assert parsed.scheme == "https" and parsed.hostname and not parsed.username, "Dùng URL Git HTTPS không chứa token."
    source_dir = Path("/content/repos") / hashlib.sha256((SOURCE_REPO + SOURCE_REVISION).encode()).hexdigest()[:16]
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--", SOURCE_REPO, str(source_dir)], check=True)
    if SOURCE_REVISION:
        assert not SOURCE_REVISION.startswith("-")
        subprocess.run(["git", "-C", str(source_dir), "checkout", "--detach", SOURCE_REVISION], check=True)
    subprocess.run(["git", "-C", str(source_dir), "rev-parse", "HEAD"], check=True)
    if BACKEND == "nuclio":
        print("Các file cấu hình tìm thấy:")
        print("\\n".join(str(p.relative_to(source_dir)) for p in source_dir.rglob("function*.yaml")))
else:
    source_dir.mkdir(parents=True, exist_ok=True)

config = dict(backend=BACKEND, source_dir=str(source_dir), model_repo=MODEL_REPO.strip(),
              model_revision=MODEL_REVISION, model_class=MODEL_CLASS.strip(),
              task=SEGMENTATION_TASK, function_name=FUNCTION_NAME, threshold=THRESHOLD,
              min_area=MIN_MASK_AREA, function_yaml=FUNCTION_YAML, adapter_file=ADAPTER_FILE,
              working_dir=WORKING_DIR, extra=json.loads(EXTRA_CONFIG_JSON))
Path("/content/cvat-config.json").write_text(json.dumps(config, indent=2))
print("Source directory:", source_dir)
""")

cell("markdown", """
## 3. Dependencies và weights riêng của repo (chỉ khi cần)

Với EoMT chạy `transformers`, bỏ trống cell sau: weights được tải từ MODEL_REPO.
Không cài toàn bộ requirements training của EoMT vào Colab chỉ để inference.

Với `nuclio`/`custom`, đọc README và `spec.build` trong function YAML, rồi nhập các lệnh cài đặt,
tải checkpoint, đặt biến môi trường/đường dẫn cần thiết. Không tự thực thi Dockerfile hoặc các
lệnh RUN trong repo. Handler có đường dẫn `/opt/nuclio/...` cần weights ở đúng đường dẫn đó.
Các lệnh dưới chạy bằng Bash, trong SOURCE_REPO. Chỉ chạy mã từ repo bạn tin cậy.
Sau khi đổi model/backend hoặc thư viện GPU, khởi động runtime mới để tránh xung đột dependencies.
""")
cell("code", '''
#@title 3. Cài đặt riêng của repo (EoMT Transformers: để trống)
SETUP_COMMANDS = r"""
"""
if SETUP_COMMANDS.strip():
    subprocess.run(["bash", "-e", "-c", SETUP_COMMANDS], cwd=source_dir, check=True)
''')

cell("markdown", """
## 4. Ghi gateway độc lập
Cell này chứa toàn bộ implementation. Không cần tải file `runtime.py` riêng.
""")
cell("code", "%%writefile /content/cvat_colab_runtime.py\n" + (ROOT / "runtime.py").read_text(encoding="utf-8"))

cell("code", """
#@title 5. Nạp model và khởi động gateway (weights có thể tải vài phút)
import os, secrets, time, requests
from getpass import getpass
if "model_process" in globals() and model_process.poll() is None:
    model_process.terminate()
    model_process.wait(timeout=30)
if "model_log" in globals():
    model_log.close()
BRIDGE_TOKEN = secrets.token_hex(32)
model_log = open("/content/cvat-model.log", "w")
model_process = subprocess.Popen(
    [sys.executable, "-u", "/content/cvat_colab_runtime.py", "/content/cvat-config.json"],
    env={**os.environ, "CVAT_COLAB_TOKEN": BRIDGE_TOKEN}, stdout=model_log, stderr=subprocess.STDOUT)
AUTH_HEADERS = {"Authorization": "Bearer " + BRIDGE_TOKEN}
for attempt in range(900):
    if model_process.poll() is not None:
        print(Path("/content/cvat-model.log").read_text()[-12000:])
        raise RuntimeError("Model không khởi động được. Xem lỗi bên trên.")
    try:
        health = requests.get("http://127.0.0.1:8000/healthz", headers=AUTH_HEADERS, timeout=2)
        if health.ok:
            break
    except requests.RequestException:
        pass
    if attempt % 30 == 0:
        print("Đang nạp model...", attempt * 2, "giây; log: /content/cvat-model.log")
    time.sleep(2)
else:
    model_process.terminate()
    model_process.wait(timeout=30)
    raise TimeoutError("Hết 30 phút nạp model. Kiểm tra log và checkpoint.")
FUNCTIONS = requests.get("http://127.0.0.1:8000/api/functions", headers=AUTH_HEADERS, timeout=10).json()
ACTIVE_FUNCTION = next(iter(FUNCTIONS))
print("Ready:", ACTIVE_FUNCTION)
labels = json.loads(FUNCTIONS[ACTIVE_FUNCTION]["metadata"]["annotations"].get("spec") or "[]")
print("Nhãn model:", [item["name"] for item in labels])
""")

cell("code", """
#@title 6. Thử ảnh ngay trong notebook trước khi nối CVAT (detector)
from google.colab import files
import base64, io
import numpy as np
from PIL import Image
assert FUNCTIONS[ACTIVE_FUNCTION]["metadata"]["annotations"]["type"] == "detector", "Interactor: thử qua CVAT với prompts phù hợp."
uploaded = files.upload()
assert uploaded, "Chọn một ảnh để thử."
image_bytes = next(iter(uploaded.values()))
preview_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
response = requests.post("http://127.0.0.1:8000/api/function_invocations",
    headers={**AUTH_HEADERS, "x-nuclio-function-name": ACTIVE_FUNCTION},
    json={"image": base64.b64encode(image_bytes).decode(), "threshold": THRESHOLD}, timeout=180)
response.raise_for_status()
predictions = response.json()
overlay = np.asarray(preview_image).copy()
rng = np.random.default_rng(42)
for annotation in predictions:
    if annotation.get("type") != "mask":
        continue
    pixels = annotation["mask"]
    x0, y0, x1, y1 = map(int, pixels[-4:])
    mask = np.array(pixels[:-4], dtype=bool).reshape(y1 - y0 + 1, x1 - x0 + 1)
    crop = overlay[y0:y1 + 1, x0:x1 + 1]
    crop[mask] = (0.55 * crop[mask] + 0.45 * rng.integers(40, 255, size=3)).astype(np.uint8)
display(Image.fromarray(overlay))
print("Số kết quả:", len(predictions))
print("Nhãn:", sorted({p.get("label", "") for p in predictions}))
""")

cell("code", """
#@title 7. Mở HTTPS tunnel, tạo file kết nối cho Windows
from pyngrok import ngrok
from google.colab import files
try:
    from google.colab import userdata
    ngrok_token = userdata.get("NGROK_AUTHTOKEN")
except Exception:
    ngrok_token = getpass("ngrok authtoken (ẩn): ")
assert ngrok_token.strip(), "Cần ngrok authtoken."
ngrok.set_auth_token(ngrok_token.strip())
if "tunnel" in globals():
    ngrok.disconnect(tunnel.public_url)
tunnel = ngrok.connect(addr=8000, proto="http", bind_tls=True)
COLAB_URL = tunnel.public_url
assert COLAB_URL.startswith("https://")
remote = requests.get(COLAB_URL + "/healthz",
    headers={**AUTH_HEADERS, "ngrok-skip-browser-warning": "1"}, timeout=30)
remote.raise_for_status()
print("Tunnel:", COLAB_URL)
print("Remote health:", remote.json())
env_path = Path("/content/colab.env")
env_path.write_text("COLAB_URL=" + COLAB_URL + "\\nCOLAB_TOKEN=" + BRIDGE_TOKEN + "\\n")
os.chmod(env_path, 0o600)
files.download(str(env_path))
print("File colab.env chứa token: giữ riêng, không commit hoặc chia sẻ.")
""")

cell("markdown", r"""
## 8. Nối CVAT Docker trên Windows

Đặt `colab.env` vừa tải vào `D:\cvat\colab-gpu\colab.env`.
Repo local đã có `bridge.py` và `docker-compose.colab.yml` trong thư mục này.
Chạy PowerShell từ `D:\cvat` (nếu đang dùng thêm compose override riêng, giữ chúng và đặt override Colab cuối cùng):

```powershell
cd D:\cvat
docker compose --env-file .env --env-file colab-gpu/colab.env -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml up -d
```

Nếu repo không có `.env`, bỏ cặp `--env-file .env`. Lần đầu image bridge cần được tải về.
Override áp dụng cho cả server và annotation worker; không cần deploy Nuclio Docker trên Colab.
Khi dùng override này, CVAT liệt kê model Colab thay cho danh sách model Nuclio local.

Kiểm tra kết nối từ chính container CVAT:

```powershell
docker compose -f docker-compose.yml exec cvat_server python -c "import requests; r=requests.get('http://colab_bridge:8070/api/functions', timeout=30); print(r.status_code); print(r.text[:3000]); r.raise_for_status()"
```

Mở CVAT local, refresh trang Models nếu có; tại Tasks chọn **Automatic annotation**,
chọn model, map nhãn model sang nhãn task và chạy thử một task nhỏ.
Giữ output dạng mask; đừng bật chuyển sang polygon nếu bạn cần bảo toàn lỗ và các vùng rời nhau.
Với EoMT, dùng Automatic annotation, không tìm nó trong công cụ positive/negative clicks.
Với `nuclio` interactor tương thích, dùng công cụ AI trong trang annotation.

Runtime restart làm model/tunnel ngừng chạy. Chạy lại notebook, tải `colab.env` mới và chạy lại
lệnh `up -d` để bridge nhận URL/token mới. Không cần mở port CVAT ra Internet.

Quay về backend local ban đầu: dùng đúng danh sách compose trước khi thêm Colab,
recreate `cvat_server` và `cvat_worker_annotation`, rồi dừng bridge.
Ví dụ nếu trước đây chạy serverless local:

```powershell
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d --force-recreate cvat_server cvat_worker_annotation
docker compose --env-file colab-gpu/colab.env -f docker-compose.yml -f colab-gpu/docker-compose.colab.yml stop colab_bridge
```
Không dùng `down -v`, vì lệnh đó xóa volume dữ liệu.

## Lỗi thường gặp
* `401`: token cũ hoặc sai. Tải lại env, chạy `up -d` lại.
* `502`/timeout: runtime/tunnel đã dừng, URL cũ hoặc mạng bị chặn.
* `500`: đọc `/content/cvat-model.log`, thường là weights/dependencies hoặc VRAM.
* Model không hiện: kiểm tra `/api/functions` từ container, refresh CVAT và kiểm tra loại adapter.
* Kết quả rỗng: kiểm tra threshold, nhãn checkpoint và ảnh thử; COCO chỉ có các lớp đã học.
* CUDA out of memory: dùng checkpoint nhỏ hơn hoặc cấu hình input phù hợp với model.
* Không kết nối được Docker: mở Docker Desktop và đợi Linux engine sẵn sàng.

## Repo tùy chỉnh
`custom` yêu cầu file `cvat_adapter.py` do repo cung cấp hoặc do bạn viết:

```python
def create_adapter(config):
    # Load the chosen repo's model and checkpoint using config["extra"].
    return adapter

# adapter.metadata: Nuclio function dictionary, see metadata() in cell 4.
# adapter.predict(payload): JSON-serializable CVAT result.
# payload["image"]: base64-encoded image; detector output: list of annotations.
# Mask annotation: {"label": "class-name", "type": "mask", "mask": pixels_and_bbox}.
# pixels_and_bbox: cropped row-major binary pixels + [x0, y0, x1, y1], inclusive bounds.
# Interactor output must match the specific CVAT interactor protocol/version.
```

Chỉ URL GitHub không xác định được cách load weights, preprocessing và postprocessing.
Notebook không thể tự biến mọi repo research thành một model CVAT; adapter là phần cần bổ sung
khi repo không dùng giao diện Transformers đã hỗ trợ hoặc CVAT Nuclio.
""")

cell("code", """
#@title 9. Dừng phiên khi kết thúc bài thực hành (chỉ chạy khi dùng xong)
STOP_SESSION = False #@param {type:"boolean"}
if STOP_SESSION:
    if "tunnel" in globals():
        ngrok.disconnect(tunnel.public_url)
    if "model_process" in globals() and model_process.poll() is None:
        model_process.terminate()
        model_process.wait(timeout=30)
    if "model_log" in globals():
        model_log.close()
    print("Đã dừng gateway và tunnel. Có thể disconnect Colab runtime.")
else:
    print("Host đang giữ nguyên. Khi dùng xong, tick STOP_SESSION rồi chạy cell này.")
""")

notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "colab": {"name": "CVAT_Colab_GPU_Repo_Host.ipynb", "provenance": [], "private_outputs": True},
    "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}, "cells": cells}
(ROOT / "CVAT_Colab_GPU_Repo_Host.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print("Built", ROOT / "CVAT_Colab_GPU_Repo_Host.ipynb")
