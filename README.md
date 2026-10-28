# Telegram Channel Image Search Bot

A self-hosted Telegram bot that indexes images from your Telegram channels and lets you find visually similar images by simply sending a photo. Powered by OpenAI CLIP for image embeddings and a compact built-in vector store (HNSW + SQLite) instead of heavyweight vector-database packages.

---

## How It Works

1. The bot connects to your Telegram account via **Telethon (MTProto)**.
2. It downloads images from your specified channels and converts them into vector embeddings using CLIP.
3. Embeddings are stored in a local vector store — an SQLite database with a usearch HNSW graph on top (~25 packages instead of ChromaDB's ~120).
4. When you send a photo to the bot, it embeds your image and searches the database for the closest visual matches.
5. A real-time monitor watches your channels for new images and indexes them automatically as they are posted.

---

## Features

- **Visual similarity search** — send any photo and get back the most visually similar images from your indexed channels, with match percentage and direct links.
- **Large-file support over MTProto** — photos, image documents, and database backups are downloaded over MTProto, so the Bot API's 20 MB limit no longer applies.
- **String-session login** — authenticate from a single environment variable; no interactive `/login` and no session file required (great for Docker/cloud).
- **Auto-monitor** — starts automatically on every boot, watches all indexed channels in real time for new and deleted images.
- **Startup sync** — on reboot the bot catches up on any images posted while it was offline.
- **Live pinned status** — pin a message in any chat that auto-updates with current index count, monitor state, and activity.
- **Duplicate finder** — scans the entire database and groups near-identical images.
- **Cleanup** — checks indexed messages still exist and removes vectors for any that were deleted.
- **Multi-channel support** — index as many public or private channels as needed.
- **DB backup & restore** — export the full database as a zip and reimport it later (auto-split for large databases).
- **Multiple CLIP models** — choose between faster smaller models or slower higher-accuracy ones.
- **Auto-tune** — automatically adjusts batch size and parallelism based on available RAM.
- **Up to 336px input** — the high-accuracy models embed at 336×336 instead of 224×224; the resolution is read from the model file itself, so no config change is needed. Chinese-CLIP is included for multilingual / CJK channel content.

---

## Choosing a CLIP model

| Key | Model | Size | RAM | Input | Notes |
|-----|-------|------|-----|-------|-------|
| `vit-b-32` | CLIP ViT-B/32 | 85 MB | 350 MB | 224 px | Fastest, general use |
| `vit-b-16` | CLIP ViT-B/16 | 83 MB | 500 MB | 224 px | Better detail, handles edits |
| `vit-l-14` | CLIP ViT-L/14 | 293 MB | 1.2 GB | 224 px | High accuracy |
| `vit-l-14-336` | CLIP ViT-L/14 @336 | 307 MB | 1.8 GB | 336 px | **Best general accuracy (CLIP)** |
| `siglip-base-patch16-224` | SigLIP Base/16 | 95 MB | 400 MB | 224 px | Stronger retrieval, fast, small servers |
| `siglip-large-patch16-384` | SigLIP Large/16 @384 | 314 MB | 1.4 GB | 384 px | **Highest retrieval quality** |
| `siglip2-base-patch16-224` | SigLIP2 Base/16 | 90 MB | 400 MB | 224 px | Newer SigLIP2, fast, small servers |

Use `/model` to download and switch. The input resolution and crop mode are read
from the model file (and the recorded model settings), so 224 px and 336 px models
both work without any configuration. SigLIP models use their own normalisation
(mean/std 0.5) and square resize, which is handled automatically per model.

**Changing the model invalidates the existing vectors.** A different embedding
dimension makes the store reject the next insert outright; a same-dimension model
produces a different (incomparable) feature space with no error at all. The bot
stores a model fingerprint inside the database directory and warns on startup, on
`/import`, and on `/model` whenever the recorded model no longer matches what is
on disk — **run `/index` again after any model change.**

`auto-tune` recommends the 336 px model whenever RAM permits; you can always
override with `/model`.

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
| `/tune` | Manually override tuning: `intra=`, `inter=`, `batch=`, `parallel=`; `/tune reset` for auto |
| `/export` | Download a backup of the database |
| `/import` | Restore the database from a backup |
| `/model` | Download and switch CLIP model |
| `/session` | Check Telegram session status |
| `/login` | Authenticate your Telegram account (only if not using a string session) |

**Search:** just send any photo (or image file) directly to the bot.

---

## Requirements

- Python 3.11+
- Telegram API credentials from [my.telegram.org](https://my.telegram.org)
- A bot token from [@BotFather](https://t.me/BotFather)
- ~500 MB RAM minimum (depends on the model chosen)
- Disk space proportional to the number of images indexed (~1–2 KB per image vector)

---

## Environment Variables

Every variable, what it's for, and whether you need it.

### Required

| Variable | Purpose |
|----------|---------|
| `TELEGRAM_API_ID` | Your Telegram **app** API ID from [my.telegram.org](https://my.telegram.org). Identifies the MTProto application used to read channels **and** to download user-sent files larger than 20 MB. Without it the bot can't talk to Telegram over MTProto. |
| `TELEGRAM_API_HASH` | The secret paired with `TELEGRAM_API_ID`. Required for the same reasons. Keep it private. |
| `BOT_TOKEN` | The token from [@BotFather](https://t.me/BotFather). Used for (a) the normal bot interface (commands, replies) and (b) logging the bot in over MTProto so it can download large files you send it. |
| `ADMIN_IDS` | Comma-separated numeric Telegram user IDs allowed to run admin commands (`/index`, `/login`, `/import`, etc.). The **first** ID also receives startup notifications. Get yours from [@userinfobot](https://t.me/userinfobot). If left empty, every user is treated as admin — not recommended. |

### Session (choose ONE approach)

| Variable | Purpose |
|----------|---------|
| `TELETHON_STRING_SESSION` | **Recommended.** A **Telethon** `StringSession` for your *user* account. When set, the bot authenticates directly from this string — no `/login`, no `.session` file. Ideal for Docker/cloud. Generate it with `python generate_session.py`. **⚠️ Must be a Telethon string — a Pyrogram session string is a different, incompatible format and will silently fail (the bot will keep asking you to `/login`).** Treat it as a password: it grants full access to your account. |
| `TELEGRAM_SESSION` | The base **file name** (not a string!) for the on-disk session created by `/login`. Default `indexer_session` → produces `indexer_session.session`. Only relevant when you are **not** using `TELETHON_STRING_SESSION`. A second file, `<name>_bot.session`, is auto-created for the bot's MTProto download client. |
| `TELETHON_SESSION_B64` | Optional. Base64 of an existing `<name>.session` file. `entrypoint.sh` decodes it to disk on container start. A file-based alternative to `TELETHON_STRING_SESSION`. Leave blank if you use the string session. |

> Precedence: if `TELETHON_STRING_SESSION` is set it always wins; the bot ignores any session file and `/login` is unnecessary.

### Optional tuning (sensible defaults)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIMILARITY_THRESHOLD` | `0.85` | Minimum cosine similarity for a result to be shown when searching. Higher = stricter matches. |
| `MAX_RESULTS` | `10` | Maximum number of matches returned per search. |
| `BATCH_SIZE` | `32` | How many images are embedded by CLIP per batch during indexing. Larger = faster but more RAM. (Auto-tune may override this.) |
| `PARALLEL_DOWNLOADS` | `10` | How many images are downloaded concurrently while indexing. Higher = faster but more likely to hit Telegram flood-waits. (Auto-tune may override this.) |
| `CHROMA_DIR` | `chroma_db` | Directory where the vector store is kept (`vectors.sqlite3` + `hnsw.bin` + state files). The name is kept for compatibility with existing deployments. |
| `TEMP_DIR` | `temp_images` | Scratch directory for images while they're being embedded; files are deleted afterward. |

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Create a `.env` file** (copy `.env.example`)
```env
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
BOT_TOKEN=your_bot_token
ADMIN_IDS=your_telegram_user_id
# Recommended — paste a Telethon string session (see step 3):
TELETHON_STRING_SESSION=
```

**3. Authenticate (pick one)**

*Option A — string session (recommended, no interactive login later):*
```bash
python generate_session.py
# follow the prompts, then paste the printed string into .env as
# TELETHON_STRING_SESSION=...
```

*Option B — interactive login inside Telegram:* leave `TELETHON_STRING_SESSION` blank and run `/login` after the bot starts.

**4. Download a CLIP model**
```bash
python download_model.py
```

**5. Start the bot**
```bash
python main.py bot
```

**6. First-time flow inside Telegram**
```
(/login if you skipped the string session)  →  /model  →  /index
```

After the first index completes, the monitor runs automatically on every subsequent start.

---

## Notes on the MTProto downloader

User-sent files (search photos, image documents, and `/import` backups) are downloaded through a Telethon client logged in as the bot, **not** through the Bot HTTP API. This is what removes the old `File is too big` error: the Bot API caps `getFile` downloads at 20 MB, while MTProto has no such limit. If the MTProto client can't start (missing `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` / `BOT_TOKEN`), small files (≤20 MB) still fall back to the Bot API automatically.

---

## Troubleshooting

- **"It keeps asking me to `/login` even though I set a session string."** You most likely pasted a **Pyrogram** session string, or put a string into `TELEGRAM_SESSION` (which is only a file *name*). Generate a **Telethon** string with `python generate_session.py` and put it in `TELETHON_STRING_SESSION`. Verify with `/session` — it should report `Mode: String session (env)` and `Auth: ✅`.
- **"File is too big."** Ensure `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `BOT_TOKEN` are all set so the MTProto downloader can start.
- **"Search results look wrong after I switched models."** Vectors from different CLIP models (or different preprocessing) are not comparable. The bot warns when the stored model fingerprint no longer matches the model on disk — run `/index` to rebuild the collection with the current model.
- **"`/export` says the WAL checkpoint was busy."** A checkpoint could not complete because a reader was active (usually a long index run). The backup may be missing the newest transactions. Retry when indexing is idle.
- **"`/import` says the backup has no model fingerprint."** The archive predates model tracking and may have been indexed with a different model or an older preprocessing version. Import works, but re-index for reliable results.
- **"`/import` says the backup uses the old ChromaDB format."** Backups made before the storage migration can't be imported. Every such vector also predates the current model fingerprint, so they'd be rebuilt anyway — run `/index` to create a fresh store instead.
- **"Botkeep: dependency installation failed; No space left on device."** Your host panel's disk quota is smaller than the dependency install. Delete the stale `~/.cache/pip` folder via the panel's file manager to reclaim space from previous failed attempts, do **not** set any `PIP_*` environment variable (Botkeep's installer already runs pip with `--no-cache-dir`, and its dependency-reuse check is disabled whenever `PIP_*` variables or a `pip.conf` are present), and consider switching to the `vit-b-32` model (85 MB instead of ~300 MB) if the quota remains tight. After one successful install Botkeep caches the venv and later boots skip the install entirely.
