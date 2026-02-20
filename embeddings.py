from __future__ import annotations
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

log = logging.getLogger(__name__)

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_SIZE = 224
ONNX_PATH = Path(__file__).parent / "clip_visual.onnx"


class CLIPEmbedder:
    def __init__(self) -> None:
        self.session = None
        self.input_name = None
        self.model_size_mb = 0
        self._try_load()

    def _try_load(self) -> None:
        if not ONNX_PATH.exists():
            log.warning("No model file at %s", ONNX_PATH)
            return
        self.model_size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
        try:
            from auto_scale import apply_tune, get_threads
            apply_tune(self.model_size_mb)
            intra, inter = get_threads()
        except Exception:
            intra, inter = 4, 2
        try:
            log.info("Loading CLIP ONNX (%.0f MB) threads=%d/%d …",
                     self.model_size_mb, intra, inter)
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = intra
            opts.inter_op_num_threads = inter
            self.session = ort.InferenceSession(
                str(ONNX_PATH), opts, providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            log.info("CLIP ready. Input: %s", self.input_name)
        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            self.session = None

    def is_ready(self) -> bool:
        return self.session is not None

    def reload(self) -> bool:
        self.session = None
        self._try_load()
        return self.is_ready()

    def get_dim(self) -> int:
        if not self.session:
            return 0
        shape = self.session.get_outputs()[0].shape
        return shape[-1] if shape else 0

    @staticmethod
    def _preprocess(img):
        img = img.convert("RGB")
        w, h = img.size
        if w < h:
            nw, nh = _SIZE, int(h * _SIZE / w + 0.5)
        else:
            nw, nh = int(w * _SIZE / h + 0.5), _SIZE
        img = img.resize((nw, nh), Image.BICUBIC)
        left = (nw - _SIZE) // 2
        top = (nh - _SIZE) // 2
        img = img.crop((left, top, left + _SIZE, top + _SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        return arr.transpose(2, 0, 1)

    @staticmethod
    def _norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    def _run(self, batch):
        outs = self.session.run(None, {self.input_name: batch})
        emb = outs[-1] if len(outs) > 1 else outs[0]
        if len(emb.shape) == 3:
            emb = emb[:, 0, :]
        return self._norm(emb)

    def _check(self):
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Use /model to download one.")

    def embed_path(self, path):
        self._check()
        inp = self._preprocess(Image.open(path))[np.newaxis]
        return self._run(inp).flatten().tolist()

    def embed_bytes(self, data):
        self._check()
        inp = self._preprocess(Image.open(BytesIO(data)))[np.newaxis]
        return self._run(inp).flatten().tolist()

    def embed_batch(self, paths):
        self._check()
        good = []
        for i, p in enumerate(paths):
            try:
                good.append((i, self._preprocess(Image.open(p))))
            except Exception as exc:
                log.warning("Skip %s: %s", p, exc)
        result = [None] * len(paths)
        if not good:
            return result
        indices, arrays = zip(*good)
        out = self._run(np.stack(arrays).astype(np.float32))
        for pos, idx in enumerate(indices):
            result[idx] = out[pos].flatten().tolist()
        return result
