import os
import sys
import urllib.request

URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx"
OUT = "clip_visual.onnx"

if os.path.exists(OUT):
    print(f"✅ Exists: {OUT} ({os.path.getsize(OUT) // 1024 // 1024} MB)")
    sys.exit(0)

print(f"Downloading → {OUT}")


def prog(c, b, t):
    mb = c * b / 1024 / 1024
    sys.stdout.write(f"\r  {mb:.0f} / {t / 1024 / 1024:.0f} MB")
    sys.stdout.flush()


urllib.request.urlretrieve(URL, OUT, reporthook=prog)
print(f"\n✅ {OUT} ({os.path.getsize(OUT) // 1024 // 1024} MB)")
