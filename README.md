# Telegram Channel Image Search Bot

A self-hosted Telegram bot that indexes images from your Telegram channels and lets you find visually similar images by simply sending a photo. Powered by OpenAI CLIP for image embeddings and ChromaDB as the vector database.

---

## How It Works

1. The bot connects to your Telegram account via the Telethon API
2. It downloads images from your specified channels and converts them into vector embeddings using CLIP
3. Embeddings are stored in a local ChromaDB vector database
4. When you send a photo to the bot, it embeds your image and searches the database for the closest visual matches
5. A real-time monitor watches your channels for new images and indexes them automatically as they are posted

---

## Features

- **Visual similarity search** — send any photo and get back the most visually similar images from your indexed channels, with match percentage and direct links
- **Auto-monitor** — starts automatically on every boot, watches all indexed channels in real time for new and deleted images
- **Startup sync** — on reboot the bot catches up on any images posted while it was offline
- **Live pinned status** — pin a message in any chat that auto-updates with current index count, monitor state, and activity
- **Duplicate finder** — scans the entire database and groups near-identical images
- **Cleanup** — checks indexed messages still exist and removes vectors for any that were deleted
- **Multi-channel support** — index as many public or private channels as needed
- **DB backup & restore** — export the full database as a zip and reimport it later
- **Multiple CLIP models** — choose between faster smaller models or slower higher-accuracy ones depending on your hardware
- **Auto-tune** — automatically adjusts batch size and parallelism based on available RAM
- **Private channel support** — works with private channels as long as your Telegram account is a member

---

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Show current status |
| `/index` | Index a channel |
| `/monitor` | View or toggle the real-time monitor |
| `/checkupdates` | Manually scan all channels for new images |
| `/setpinned` | Pin a live status message in the current chat |
| `/cleanup` | Remove vectors for deleted messages |
| `/duplicates` | Find near-duplicate images across the database |
| `/stats` | Vector count per channel |
| `/system` | Server CPU, RAM, and disk usage |
| `/autotune` | Auto-optimize performance settings |
| `/export` | Download a backup of the database |
| `/import` | Restore the database from a backup |
| `/model` | Download and switch CLIP model |
| `/session` | Check Telegram session status |
| `/login` | Authenticate your Telegram account |

**Search:** just send any photo directly to the bot.

---

## Requirements

- Python 3.10+
- Telegram API credentials from [my.telegram.org](https://my.telegram.org)
- A bot token from [@BotFather](https://t.me/BotFather)
- ~500 MB RAM minimum (depends on model chosen)
- Disk space proportional to number of images indexed (~1–2 KB per image vector)

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create a `.env` file**
```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_SESSION=indexer_session
BOT_TOKEN=your_bot_token
ADMIN_IDS=your_telegram_user_id
```

Get `API_ID` and `API_HASH` from [my.telegram.org](https://my.telegram.org).  
Get `BOT_TOKEN` from [@BotFather](https://t.me/BotFather).  
Get your `ADMIN_IDS` user ID from [@userinfobot](https://t.me/userinfobot).

**3. Download a CLIP model**
```bash
python download_model.py
```

**4. Start the bot**
```bash
python main.py bot
```

**5. First-time flow inside Telegram**
```
/login  →  /model  →  /index
```

After the first index is complete the monitor runs automatically on every subsequent start.

---

## Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_DIR` | `chroma_db` | Where the vector database is stored |
| `TEMP_DIR` | `temp_images` | Temporary download directory |
| `BATCH_SIZE` | `32` | Embedding batch size |
| `PARALLEL_DOWNLOADS` | `10` | Concurrent image downloads |
| `SIMILARITY_THRESHOLD` | `0.85` | Minimum similarity score for results |
| `MAX_RESULTS` | `10` | Maximum search results returned |
