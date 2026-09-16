"""Small Nuclio-compatible gateway for an interactive GPU notebook session."""

import base64
import importlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response


def cvat_mask(mask):
    """Nuclio uses cropped, row-major pixels + inclusive bbox, NOT CVAT REST RLE."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8).ravel().tolist() + [x0, y0, x1, y1]


def metadata(name, labels, display_name=None):
    return {
        "metadata": {"name": name, "annotations": {
            "name": display_name or name, "type": "detector",
            "spec": json.dumps(labels), "framework": "pytorch",
        }},
        "spec": {"description": "GPU inference from a configurable Colab repository"},
        "status": {"state": "ready", "httpPort": 8000},
    }


class TransformersAdapter:
    def __init__(self, config):
        import torch
        import transformers

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime in Colab first")
        repo = config["model_repo"]
        if not repo:
            raise ValueError("MODEL_REPO must be a Hugging Face checkpoint repository")
        kwargs = {"revision": config.get("model_revision") or "main", "trust_remote_code": False}
        model_config = transformers.AutoConfig.from_pretrained(repo, **kwargs)
        class_name = config.get("model_class") or (model_config.architectures or [""])[0]
        model_class = getattr(transformers, class_name, None)
        if model_class is None or not hasattr(model_class, "from_pretrained"):
            raise ValueError(f"Unsupported model class {class_name!r}; supply MODEL_CLASS or a custom adapter")
        self.processor = transformers.AutoImageProcessor.from_pretrained(repo, **kwargs)
        self.mode = config["task"]
        self.postprocess = getattr(self.processor, f"post_process_{self.mode}_segmentation", None)
        if not callable(self.postprocess):
            raise ValueError(f"This checkpoint processor does not support {self.mode} segmentation")
        self.model = model_class.from_pretrained(repo, **kwargs).to("cuda").eval()
        self.labels = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        if not self.labels or len(set(self.labels.values())) != len(self.labels):
            raise ValueError("Checkpoint must have a nonempty, unique id2label mapping")
        self.threshold = config.get("threshold", 0.5)
        self.min_area = config.get("min_area", 16)
        self.metadata = metadata(config["function_name"], [
            {"id": k, "name": v, "type": "mask"} for k, v in self.labels.items()
        ], repo)

    def predict(self, payload):
        import torch

        image = Image.open(io.BytesIO(base64.b64decode(payload["image"], validate=True))).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        kwargs = {"target_sizes": [(image.height, image.width)]}
        threshold = float(payload.get("threshold", self.threshold))
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.mode != "semantic":
            kwargs["threshold"] = threshold
        processed = self.postprocess(outputs, **kwargs)[0]
        if self.mode == "semantic":
            segmentation = processed.detach().cpu().numpy()
            segments = [{"id": int(k), "label_id": int(k)} for k in np.unique(segmentation)]
        else:
            segmentation = processed["segmentation"].detach().cpu().numpy()
            segments = processed["segments_info"]
        result = []
        for segment in segments:
            label_id = int(segment["label_id"])
            if label_id not in self.labels:
                continue
            mask = segmentation == segment["id"]
            if np.count_nonzero(mask) < self.min_area:
                continue
            annotation = {"label": self.labels[label_id], "type": "mask", "mask": cvat_mask(mask)}
            if "score" in segment:
                annotation["confidence"] = str(float(segment["score"]))
            result.append(annotation)
        return result


def load_adapter(config):
    backend = config["backend"]
    if backend == "transformers":
        return TransformersAdapter(config)
    repo = Path(config["source_dir"]).resolve()
    if backend == "custom":
        path = (repo / config["adapter_file"]).resolve()
        spec = importlib.util.spec_from_file_location("user_cvat_adapter", path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo))
        spec.loader.exec_module(module)
        return module.create_adapter(config)
    if backend != "nuclio":
        raise ValueError(f"Unknown backend: {backend}")
    import yaml

    path = (repo / config["function_yaml"]).resolve()
    doc = yaml.safe_load(path.read_text())
    os.chdir(config.get("working_dir") or path.parent)
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(repo))
    module_name, handler_name = doc["spec"]["handler"].split(":")
    module = importlib.import_module(module_name)
    context = SimpleNamespace(user_data=SimpleNamespace(), logger=logging.getLogger("model"))
    context.Response = lambda body, headers=None, content_type="application/json", status_code=200: Response(
        content=body, headers=headers, media_type=content_type, status_code=status_code
    )
    if hasattr(module, "init_context"):
        module.init_context(context)
    doc.setdefault("status", {}).update(state="ready", httpPort=8000)
    doc["spec"].setdefault("description", doc["metadata"]["name"])
    annotations = doc["metadata"]["annotations"]
    if not isinstance(annotations.get("spec"), str):
        annotations["spec"] = json.dumps(annotations.get("spec") or [])
    handler = getattr(module, handler_name)
    return SimpleNamespace(metadata=doc, predict=lambda body: handler(context, SimpleNamespace(body=body)))


def create_app(adapter, token):
    if len(token) < 32:
        raise ValueError("Use a random token of at least 32 characters")
    info = adapter.metadata
    function_name = info["metadata"]["name"]
    if info["metadata"]["annotations"]["type"] not in {"detector", "interactor"}:
        raise ValueError("This gateway supports detector and interactor adapters")
    json.loads(info["metadata"]["annotations"].get("spec") or "[]")

    def authorize(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid bridge token")

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, dependencies=[Depends(authorize)])
    inference_lock = threading.Lock()

    @app.get("/healthz")
    def health():
        return {"status": "ready", "function": function_name}

    @app.get("/api/functions")
    def functions():
        return {function_name: info}

    @app.get("/api/functions/{name}")
    def function(name: str):
        if name != function_name:
            raise HTTPException(404, "Unknown function")
        return info

    @app.post("/api/function_invocations")
    def invoke(payload: dict, x_nuclio_function_name: str = Header(default="")):
        if x_nuclio_function_name != function_name:
            raise HTTPException(404, "Unknown function")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(503, "GPU busy; retry after the current inference finishes")
        try:
            return adapter.predict(payload)
        except Exception:
            logging.exception("Model inference failed")
            raise HTTPException(500, "Model inference failed; inspect the Colab runtime log")
        finally:
            inference_lock.release()

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    config = json.loads(Path(sys.argv[1]).read_text())
    app = create_app(load_adapter(config), os.environ["CVAT_COLAB_TOKEN"])
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
