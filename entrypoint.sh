#!/bin/bash
set -e
echo "=== Starting ==="

SESSION_NAME="${TELEGRAM_SESSION:-indexer_session}"

# Preferred: TELETHON_STRING_SESSION (no files needed) — handled in-app.
if [ -n "$TELETHON_STRING_SESSION" ]; then
    echo "✅ Using TELETHON_STRING_SESSION from environment (no session file needed)"
# Fallback: restore a base64-encoded .session file into place.
elif [ -n "$TELETHON_SESSION_B64" ]; then
    echo "$TELETHON_SESSION_B64" | base64 -d > "/app/${SESSION_NAME}.session"
    echo "✅ Session file restored from TELETHON_SESSION_B64"
else
    echo "ℹ️  No session provided — use /login in Telegram, or set TELETHON_STRING_SESSION"
fi

if [ -f /app/clip_visual.onnx ]; then
    echo "✅ Model: $(du -h /app/clip_visual.onnx | cut -f1)"
fi
mkdir -p /app/chroma_db /app/temp_images
python /app/health.py &
cd /app
exec python main.py bot
