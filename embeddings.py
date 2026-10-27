from __future__ import annotations
import logging
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

log = logging.getLogger(__name__)

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
ONNX_PATH = Path(__file__).parent / "clip_visual.onnx"

# Server-side guard for untrusted uploads. Pillow's built-in decompression-bomb
# check only *warns* below 2x the limit, and the full-size resize below would
# allocate multiple GB for an image in the 89M-179M px band — an OOM kill here
# takes down every other service on the box.
_MAX_PIXELS = 89_478_485
_MAX_SIDE = 8192

# Version of the preprocessing pipeline. Changing the resize path (e.g. adding
# Image.draft() or reducing_gap) changes the decoded pixels, and therefore the
# embeddings. Bumping this invalidates every existing vector, so it is folded
# into the model fingerprint to force the "re-index required" warning.
PREP_VERSION = 2

# Fallback when the model declares nothing and the session shape is dynamic.
_DEFAULT_INPUT = 224


class CLIPEmbedder:
    def __init__(self) -> None:
        self.session = None
        self.input_name = None
        self.model_size_mb = 0
        # (intra, inter) the current session was built with; None until a
        # load is attempted. /tune compares this to decide whether the
        # session must be rebuilt for new thread counts to take effect.
        self._thread_tune = None
        # Per-model preprocessing, resolved in _try_load().
        self.input_size = _DEFAULT_INPUT
        self.crop_mode = "center"
        self._lock = threading.Lock()
        self._try_load()

    def _apply_model_settings(self) -> None:
        """
        Decide the input resolution and cropping from the active model.

        The resolution is read from the ONNX input shape when it is a fixed
        integer, so any model (224px, 336px, other) works without a code
        change. Only when the shape is dynamic do we fall back to what the
        model registry declares, and finally to 224.
        """
        settings = {"crop": "center", "input": _DEFAULT_INPUT}
        try:
            from models import get_active_settings
            settings = get_active_settings()
        except Exception:
            pass
        self.crop_mode = str(settings.get("crop", "center"))
        declared = int(settings.get("input", _DEFAULT_INPUT) or _DEFAULT_INPUT)

        detected = None
        try:
            shape = self.session.get_inputs()[0].shape
            # Typical CLIP shape: [batch, channels, height, width]
            dims = [d for d in (shape or [])][2:]
            fixed = [int(d) for d in dims if isinstance(d, int) and d > 0]
            if len(fixed) >= 2 and fixed[0] == fixed[1]:
                detected = fixed[0]
        except Exception:
            detected = None
        if detected:
            self.input_size = detected
        else:
            self.input_size = declared
        log.info("CLIP preprocessing: %dpx input, crop=%s",
                 self.input_size, self.crop_mode)

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
        # Record BEFORE building: /tune compares this even when the build
        # below fails, so it can offer a retry once the file is fixed.
        self._thread_tune = (intra, inter)
        try:
            log.info("Loading CLIP ONNX (%.0f MB) threads=%d/%d …",
                     self.model_size_mb, intra, inter)
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = intra
            opts.inter_op_num_threads = inter
            # Documented benchmark winner for CPU inference (onnxruntime docs,
            # "Threading control"): reduces spin-wait overhead between ops.
            try:
                opts.add_session_config_entry(
                    "session.intra_op.spin_duration_us", "1000")
                opts.add_session_config_entry(
                    "session.intra_op.spin_backoff_max", "8")
            except Exception:
                pass
            self.session = ort.InferenceSession(
                str(ONNX_PATH), opts, providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            # Resolve resolution/cropping once the session exists.
            self._apply_model_settings()
            log.info("CLIP ready. Input: %s (%dpx, %s crop)",
                     self.input_name, self.input_size, self.crop_mode)
        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            self.session = None

    def is_ready(self) -> bool:
        return self.session is not None

    def reload(self) -> bool:
        """
        Swap in a new session.

        The lock is held for the whole build so _run() can never observe the
        intermediate session=None state (which used to make an entire batch
        silently produce no vectors). onnxruntime session *construction* is not
        thread-safe either, so serialising it is required regardless.
        """
        with self._lock:
            old_session, old_name = self.session, self.input_name
            self.session = None
            self.input_name = None
            try:
                self._try_load()
            except Exception:
                self.session = old_session
                self.input_name = old_name
                raise
        return self.is_ready()

    def get_dim(self) -> int:
        if not self.session:
            return 0
        shape = self.session.get_outputs()[0].shape
        return shape[-1] if shape else 0

    def _preprocess(self, img):
        """
        Scale an image to this model's CLIP input tensor.

        Resolution comes from the loaded ONNX session (auto-detected, so 224px
        and 336px models both work). Two crop modes:

          "center"  — OpenAI CLIP standard: resize the shortest edge to N, then
                      centre-crop N×N. Used by every CLIP model here.
          "stretch" — resize straight to N×N. Chinese-CLIP's feature extractor
                      disables centre-cropping, so this matches it exactly.
        """
        target = max(16, int(self.input_size or _DEFAULT_INPUT))
        # 1. Bomb guard from the header, before anything that allocates.
        w, h = img.size
        if w * h > _MAX_PIXELS or max(w, h) > _MAX_SIDE:
            raise ValueError(
                f"Image too large to embed: {w}x{h}px (limit {_MAX_SIDE}px/side)")

        # 2. JPEG/MPO only: let libjpeg decode at a reduced DCT scale instead of
        #    full resolution. MUST happen before any load/convert, after which it
        #    is a no-op. Measured ~7x faster on 4032x3024 JPEG (81ms -> 12ms).
        img.draft("RGB", (target, target))

        # 3. `img.size` may have changed after draft() — re-read it.
        img = img.convert("RGB")
        if self.crop_mode == "stretch":
            img = img.resize((target, target), Image.BICUBIC, reducing_gap=2.0)
        else:
            w, h = img.size
            if w < h:
                nw, nh = target, int(h * target / w + 0.5)
            else:
                nw, nh = int(w * target / h + 0.5), target
            # reducing_gap only engages for large downscales; a free win then.
            img = img.resize((nw, nh), Image.BICUBIC, reducing_gap=2.0)
            left = (nw - target) // 2
            top = (nh - target) // 2
            img = img.crop((left, top, left + target, top + target))

        # np.asarray avoids Pillow's copy; in-place ops avoid two more temporaries.
        arr = np.asarray(img, dtype=np.float32)
        arr *= np.float32(1.0 / 255.0)
        arr -= MEAN
        arr /= STD
        # transpose() is a view; ascontiguousarray makes the actual (3,H,W) copy.
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    @staticmethod
    def _norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

    def _run(self, batch, images=None):
        """
        Run inference on a pre-stacked batch.

        `images` (a list of raw PIL images) is an alternative to `batch` for
        convenience. onnxruntime's Run is thread-safe for concurrent use, but
        NOT for being swapped mid-call, so the session reference is copied
        once and an in-flight run keeps its own object alive.
        """
        with self._lock:
            session, name = self.session, self.input_name
        if session is None or name is None:
            raise RuntimeError("Model not loaded. Use /model to download one.")
        if images is not None:
            batch = np.stack([self._preprocess(im) for im in images]).astype(np.float32)
        outs = session.run(None, {name: batch})
        emb = outs[-1] if len(outs) > 1 else outs[0]
        if len(emb.shape) == 3:
            emb = emb[:, 0, :]
        return self._norm(emb)

    def _check(self):
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Use /model to download one.")

    def embed_path(self, path):
        self._check()
        return self._run(None, [Image.open(path)]).flatten().tolist()

    def embed_bytes(self, data):
        self._check()
        return self._run(None, [Image.open(BytesIO(data))]).flatten().tolist()

    def embed_batch(self, paths):
        self._check()
        indices, images = [], []
        for i, p in enumerate(paths):
            try:
                # Preprocess eagerly so the bomb guard fires here and the image
                # is fully decoded before we hand it to the session.
                images.append(self._preprocess(Image.open(p)))
                indices.append(i)
            except Exception as exc:
                log.warning("Skip %s: %s", p, exc)
        result = [None] * len(paths)
        if not images:
            return result
        out = self._run(np.stack(images).astype(np.float32))
        for pos, idx in enumerate(indices):
            result[idx] = out[pos].flatten().tolist()
        return result
