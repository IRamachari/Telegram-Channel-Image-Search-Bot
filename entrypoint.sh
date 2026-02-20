#!/bin/bash
set -e
echo "=== Starting ==="
if [ -n "$TELETHON_SESSION_B64" ]; then
    echo "$TELETHON_SESSION_B64" | base64 -d > /app/indexer_session.session
    echo "✅ Session restored"
fi
if [ -f /app/clip_visual.onnx ]; then
    echo "✅ Model: $(du -h /app/clip_visual.onnx | cut -f1)"
fi
mkdir -p /app/chroma_db /app/temp_images
python /app/health.py &
cd /app
exec python main.py bot
