from __future__ import annotations
import logging
import threading
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

log = logging.getLogger(__name__)

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
ONNX_PATH = Path(__file__).parent / "clip_visual.onnx"

# Server-side guard for untrusted uploads. Pillow's built-in decompression-bomb
# check only *warns* below 2x the limit; the full-size resize below would
# allocate multiple GB for an image in the 89M-179M px band — an OOM kill here
# takes down every other service on the box. Derived from available RAM so a
# small box refuses huge images before decode instead of OOMing.
try:
    import psutil as _psutil
    _avail_px = _psutil.virtual_memory().available // (4 * 3)
    _MAX_PIXELS = min(89_478_485, max(4_000_000, _avail_px))
except Exception:
    _MAX_PIXELS = 89_478_485
_MAX_SIDE = 8192

# Version of the preprocessing pipeline. Changing the resize path (e.g. adding
# Image.draft() or reducing_gap) changes the decoded pixels, and therefore the
# embeddings. Bumping this invalidates every existing vector, so it is folded
# into the model fingerprint to force the "re-index required" warning.
#
# PREP_VERSION history:
#   1  full JPEG decode, BICUBIC resize, centre crop (original reliable
#      behaviour).
#   2  added Image.draft() fast decode -> reduced DCT precision -> less
#      distinctive embeddings -> "always finds the wrong image". Reverted.
#   3  back to full decode.
#   4  honour EXIF Orientation via ImageOps.exif_transpose (rotated photos
#      previously embedded sideways). Re-index required.
PREP_VERSION = 4

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
        self.mean = MEAN.copy()
        self.std = STD.copy()
        self.resample = 3
        self._lock = threading.Lock()
        # Inference / index-run counters. reload() refuses to swap the session
        # while either is non-zero: swapping mid-batch would mix two feature
        # spaces into one index with no error at all.
        self._active_runs = 0
        self._index_depth = 0
        # Why the last reload() was refused or failed, for the /model reply.
        self.last_error = None
        self._try_load()

    def _resolve_settings(self, session) -> tuple[str, int]:
        """
        Decide the input resolution and cropping for `session`.

        The resolution is read from the ONNX input shape when it is a fixed
        integer, so any model (224px, 336px, other) works without a code
        change. Only when the shape is dynamic do we fall back to what the
        model registry declares, and finally to 224.
        """
        settings = {"crop": "center", "input": _DEFAULT_INPUT,
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "std": [0.26862954, 0.26130258, 0.27577711],
                    "resample": 3}
        try:
            from models import get_active_settings
            settings = get_active_settings()
        except Exception:
            pass
        crop = str(settings.get("crop", "center"))
        declared = int(settings.get("input", _DEFAULT_INPUT) or _DEFAULT_INPUT)
        mean = np.asarray(settings.get("mean", [0.48145466, 0.4578275, 0.40821073]),
                          dtype=np.float32)
        std = np.asarray(settings.get("std", [0.26862954, 0.26130258, 0.27577711]),
                         dtype=np.float32)
        resample = int(settings.get("resample", 3) or 3)

        detected = None
        try:
            shape = session.get_inputs()[0].shape
            # Typical CLIP shape: [batch, channels, height, width]
            dims = [d for d in (shape or [])][2:]
            fixed = [int(d) for d in dims if isinstance(d, int) and d > 0]
            if len(fixed) >= 2 and fixed[0] == fixed[1]:
                detected = fixed[0]
        except Exception:
            detected = None
        size = detected or declared
        log.info("CLIP preprocessing: %dpx input, crop=%s, resample=%d", size, crop, resample)
        return crop, size, mean, std, resample

    @staticmethod
    def _options(intra: int, inter: int):
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
        return opts

    def _try_load(self) -> bool:
        """Load the model from disk. Kept for callers that expect no refusal."""
        return self.reload(force=True)

    def reload(self, force: bool = False) -> bool:
        """
        Swap in a new session, keeping the old one when the build fails.

        The new session is built into locals and committed in a single step, so
        a corrupt or unreadable model leaves the previous session — and its
        preprocessing settings — fully intact instead of leaving the bot with
        no model until the next restart.

        Refuses while an inference batch or an index run is in progress (see
        is_busy()/indexing()) unless `force` is set: a swap between batches
        would put two incompatible feature spaces in the same index. Returns
        True when a usable session is live afterwards; `last_error` says why
        when it is False.
        """
        with self._lock:
            if not force and self.is_busy():
                self.last_error = ("busy: an index run or inference batch is "
                                   "in progress")
                log.warning("Model reload refused — %s", self.last_error)
                return False
            if not ONNX_PATH.exists():
                self.last_error = f"no model file at {ONNX_PATH}"
                log.warning("No model file at %s", ONNX_PATH)
                return False
            size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
            try:
                from auto_scale import apply_tune, get_threads
                apply_tune(size_mb)
                intra, inter = get_threads()
            except Exception:
                intra, inter = 4, 1
            # Record BEFORE building: /tune compares this even when the build
            # below fails, so it can offer a retry once the file is fixed.
            prev_tune = self._thread_tune
            self._thread_tune = (intra, inter)
            try:
                log.info("Loading CLIP ONNX (%.0f MB) threads=%d/%d …",
                         size_mb, intra, inter)
                # onnxruntime session construction is not thread-safe, which is
                # the other reason this whole block holds the lock.
                session = ort.InferenceSession(
                    str(ONNX_PATH), self._options(intra, inter),
                    providers=["CPUExecutionProvider"],
                )
                name = session.get_inputs()[0].name
                crop, input_size, mean, std, resample = self._resolve_settings(session)
            except Exception as exc:
                self._thread_tune = prev_tune
                self.last_error = str(exc)
                log.error("Failed to load model (previous session kept): %s",
                          exc)
                return False
            # Commit — nothing above this line mutated the live session.
            self.session, self.input_name = session, name
            self.crop_mode, self.input_size = crop, input_size
            self.mean, self.std = mean, std
            self.resample = resample
            self.model_size_mb = size_mb
            self.last_error = None
            log.info("CLIP ready. Input: %s (%dpx, %s crop)",
                     name, input_size, crop)
            return True

    def is_ready(self) -> bool:
        return self.session is not None

    def is_busy(self) -> bool:
        """True while an inference batch or an index run is in flight."""
        return bool(self._active_runs or self._index_depth)

    @contextmanager
    def indexing(self):
        """
        Mark the whole span of an index run so /model cannot swap the model
        between two batches. indexer.py should wrap its run in:

            with embedder.indexing():
                ...download/embed batches...

        reload() then refuses (returning False) until the run finishes.
        """
        with self._lock:
            self._index_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._index_depth -= 1

    def get_dim(self) -> int:
        if not self.session:
            return 0
        outs = self.session.get_outputs()
        names = [o.name for o in outs]
        chosen = None
        for preferred in ("image_embeds", "embedding", "pooler_output"):
            if preferred in names:
                chosen = outs[names.index(preferred)]
                break
        if chosen is None and outs:
            chosen = outs[-1]
        shape = chosen.shape if chosen is not None else None
        return int(shape[-1]) if shape else 0

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

        # Honour EXIF Orientation so rotated photos are embedded upright.
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        resample = (Image.LANCZOS if self.resample == 1
                    else Image.BICUBIC if self.resample >= 3
                    else Image.BILINEAR)
        if self.crop_mode == "stretch":
            img = img.resize((target, target), resample, reducing_gap=2.0)
        else:
            w, h = img.size
            if w < h:
                nw, nh = target, int(h * target / w + 0.5)
            else:
                nw, nh = int(w * target / h + 0.5), target
            img = img.resize((nw, nh), resample, reducing_gap=2.0)
            left = (nw - target) // 2
            top = (nh - target) // 2
            img = img.crop((left, top, left + target, top + target))

        arr = np.asarray(img, dtype=np.float32)
        arr *= np.float32(1.0 / 255.0)
        arr -= self.mean
        arr /= self.std
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    @staticmethod
    def _norm(v):
        v = np.asarray(v, dtype=np.float32)
        if not np.all(np.isfinite(v)):
            return v
        nrm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(nrm, 1e-8)

    def _run(self, batch, images=None):
        """
        Run inference on a pre-stacked batch.

        `images` (a list of raw PIL images) is an alternative to `batch` for
        convenience. onnxruntime's Run is thread-safe for concurrent use, but
        NOT for being swapped mid-call, so the session reference is copied
        once and an in-flight run keeps its own object alive. The run is also
        counted, so reload() refuses to swap the session under it.
        """
        with self._lock:
            session, name = self.session, self.input_name
            if session is not None and name is not None:
                self._active_runs += 1
        if session is None or name is None:
            raise RuntimeError("Model not loaded. Use /model to download one.")
        try:
            if images is not None:
                batch = np.stack([self._preprocess(im) for im in images]).astype(np.float32)
            outs = session.run(None, {name: batch})
        finally:
            with self._lock:
                self._active_runs -= 1
        emb = outs[-1] if len(outs) > 1 else outs[0]
        # Prefer a pooled 2-D embedding when the model exposes several outputs
        # (e.g. SigLIP ships last_hidden_state + pooler_output). Fall back to
        # the last output, which is the pooled one for every model here.
        names = [o.name for o in session.get_outputs()]
        for preferred in ("image_embeds", "embedding", "pooler_output"):
            if preferred in names:
                emb = outs[names.index(preferred)]
                break
        if len(emb.shape) == 3:
            emb = emb.mean(axis=1)
        return self._norm(emb)

    def _check(self):
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Use /model to download one.")

    def embed_path(self, path):
        self._check()
        with Image.open(path) as im:
            return self._run(None, [im]).flatten().tolist()

    def embed_bytes(self, data):
        self._check()
        with Image.open(BytesIO(data)) as im:
            return self._run(None, [im]).flatten().tolist()

    def embed_bytes_query(self, data):
        """
        Embed a QUERY image with light test-time augmentation (horizontal flip)
        averaged into a single vector. The index stores single (non-TTA) vectors,
        so this is "query expansion": it makes the query embedding more robust to
        Telegram's recompression and small alignment differences, improving
        recall without requiring a re-index.
        """
        self._check()
        with Image.open(BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            variants = [im, im.transpose(Image.FLIP_LEFT_RIGHT)]
            out = self._run(None, variants)   # (2, d), already unit-normalised
            return self._norm(out.mean(axis=0)).flatten().tolist()

    def embed_path_query(self, path):
        self._check()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            variants = [im, im.transpose(Image.FLIP_LEFT_RIGHT)]
            out = self._run(None, variants)
            return self._norm(out.mean(axis=0)).flatten().tolist()

    def embed_batch(self, paths):
        self._check()
        indices, images = [], []
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    images.append(self._preprocess(im))
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
