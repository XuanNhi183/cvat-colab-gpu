import ast
import base64
import contextlib
import io
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import nbformat
import numpy as np
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from bridge import Bridge, ThreadingHTTPServer
from runtime import TransformersAdapter, create_app, cvat_mask, metadata


TOKEN = "a" * 64


def fake_adapter():
    return SimpleNamespace(
        metadata=metadata("test-model", [{"id": 0, "name": "object", "type": "mask"}]),
        predict=lambda payload: [{"label": "object", "type": "mask", "mask": cvat_mask(payload["mask"])}],
    )


class GatewayTests(unittest.TestCase):
    def test_transformers_postprocessing_at_original_size(self):
        image_data = io.BytesIO()
        Image.new("RGB", (3, 2)).save(image_data, format="PNG")
        payload = {"image": base64.b64encode(image_data.getvalue()).decode(), "threshold": 0.7}

        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([[0, 2, 2], [0, 0, 2]])

        class Batch(dict):
            def to(self, device):
                self.device = device
                return self

        adapter = TransformersAdapter.__new__(TransformersAdapter)
        batch = Batch()
        adapter.processor = lambda **kwargs: batch
        adapter.model = lambda **kwargs: "logits"
        adapter.labels = {0: "background", 2: "object"}
        adapter.threshold = 0.5
        adapter.min_area = 1
        calls = []

        def postprocess(outputs, **kwargs):
            calls.append(kwargs)
            if adapter.mode == "semantic":
                return [Tensor()]
            return [{"segmentation": Tensor(), "segments_info": [{"id": 2, "label_id": 2, "score": 0.9}]}]

        adapter.postprocess = postprocess
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            for mode in ("instance", "panoptic", "semantic"):
                adapter.mode = mode
                results = adapter.predict(payload)
                self.assertEqual(batch.device, "cuda")
                self.assertEqual(calls[-1]["target_sizes"], [(2, 3)])
                self.assertEqual("threshold" in calls[-1], mode != "semantic")
                found = next(x for x in results if x["label"] == "object")
                self.assertEqual(found["mask"], [1, 1, 0, 1, 1, 0, 2, 1])
                self.assertEqual("confidence" in found, mode != "semantic")

    def test_masks_preserve_holes_and_inclusive_bounds(self):
        mask = np.zeros((7, 8), dtype=bool)
        mask[1:6, 2:7] = True
        mask[3, 4] = False
        encoded = cvat_mask(mask)
        self.assertEqual(encoded[-4:], [2, 1, 6, 5])
        decoded = np.zeros_like(mask)
        decoded[1:6, 2:7] = np.asarray(encoded[:-4]).reshape(5, 5)
        np.testing.assert_array_equal(decoded, mask)
        self.assertEqual(cvat_mask([[1]]), [1, 0, 0, 0, 0])
        self.assertIsNone(cvat_mask(np.zeros((3, 4))))

    def test_authentication_discovery_and_routing(self):
        with TestClient(create_app(fake_adapter(), TOKEN)) as client:
            self.assertEqual(client.get("/api/functions").status_code, 401)
            self.assertEqual(client.get("/healthz", headers={"Authorization": "Bearer wrong"}).status_code, 401)
            client.headers["Authorization"] = "Bearer " + TOKEN
            info = client.get("/api/functions").json()["test-model"]
            self.assertEqual(json.loads(info["metadata"]["annotations"]["spec"])[0]["type"], "mask")
            self.assertEqual(client.get("/api/functions/test-model").json(), info)
            self.assertEqual(client.get("/api/functions/unknown").status_code, 404)
            self.assertEqual(client.post("/api/function_invocations", json={}).status_code, 404)
            response = client.post("/api/function_invocations", json={"mask": [[1, 0], [0, 1]]},
                                   headers={"x-nuclio-function-name": "test-model"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["mask"], [1, 0, 0, 1, 0, 0, 1, 1])

    def test_notebook_is_valid_self_contained_and_not_model_locked(self):
        root = Path(__file__).resolve().parent
        notebook = nbformat.read(root / "CVAT_Colab_GPU_Repo_Host.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                source = cell.source
                if source.startswith("%%writefile"):
                    source = source.split("\n", 1)[1]
                    self.assertEqual(source.strip(), (root / "runtime.py").read_text(encoding="utf-8").strip())
                ast.parse(source)
                self.assertEqual(cell.outputs, [])
        config = next(c.source for c in notebook.cells if c.cell_type == "code" and "SOURCE_REPO =" in c.source)
        self.assertIn('SOURCE_REPO = ""', config)
        self.assertIn('MODEL_REPO = ""', config)


class RelayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.socket = socket.socket()
        cls.socket.bind(("127.0.0.1", 0))
        port = cls.socket.getsockname()[1]
        cls.gpu = uvicorn.Server(uvicorn.Config(create_app(fake_adapter(), TOKEN), log_level="error"))
        cls.gpu_thread = threading.Thread(target=cls.gpu.run, kwargs={"sockets": [cls.socket]}, daemon=True)
        cls.gpu_thread.start()
        for _ in range(100):
            if cls.gpu.started:
                break
            time.sleep(0.05)
        if not cls.gpu.started:
            raise RuntimeError("Test gateway failed to start")
        cls.relay = ThreadingHTTPServer(("127.0.0.1", 0), Bridge)
        cls.relay.upstream = f"http://127.0.0.1:{port}"
        cls.relay.token = TOKEN
        cls.relay_thread = threading.Thread(target=cls.relay.serve_forever, daemon=True)
        cls.relay_thread.start()
        cls.origin = f"http://127.0.0.1:{cls.relay.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.relay.shutdown()
        cls.relay.server_close()
        cls.relay_thread.join(timeout=5)
        cls.gpu.should_exit = True
        cls.gpu_thread.join(timeout=5)
        cls.socket.close()

    def test_cvat_request_through_relay_to_gateway(self):
        with urlopen(self.origin + "/api/functions", timeout=5) as response:
            self.assertIn("test-model", json.load(response))
        request = Request(self.origin + "/api/function_invocations",
                          data=json.dumps({"mask": [[0, 1]]}).encode(),
                          headers={"Content-Type": "application/json", "x-nuclio-function-name": "test-model"})
        with urlopen(request, timeout=5) as response:
            self.assertEqual(json.load(response)[0]["mask"], [1, 1, 0, 1, 0])

    def test_upstream_errors_are_preserved(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.origin + "/api/functions/nonexistent", timeout=5)
        self.assertEqual(error.exception.code, 404)
        error.exception.close()


if __name__ == "__main__":
    unittest.main()
