#!/usr/bin/env python3
"""
Generate a **Telethon** StringSession for this bot.

Why this exists
---------------
This project talks to Telegram with **Telethon**. A session string produced by
**Pyrogram** (or any other library) is a *different, incompatible format* and
will NOT work here — pasting one into TELETHON_STRING_SESSION just makes the
bot think it has no session and keep asking you to /login.

Run this once, locally, then copy the printed string into your .env as:

    TELETHON_STRING_SESSION=1Bx...<long string>...

After that the bot authenticates straight from the environment — no .session
file and no in-Telegram /login needed. Ideal for Docker / cloud hosts.

Usage
-----
    python generate_session.py

You will be asked for your API_ID / API_HASH (or it reads them from .env),
your phone number, the login code Telegram sends you, and your 2FA password
if you have one.
"""

from __future__ import annotations

import asyncio
import os

from telethon import TelegramClient
from telethon.sessions import StringSession

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


async def main() -> None:
    api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()

    if not api_id:
        api_id = input("API_ID (from https://my.telegram.org): ").strip()
    if not api_hash:
        api_hash = input("API_HASH (from https://my.telegram.org): ").strip()

    if not api_id or not api_hash:
        print("❌ API_ID and API_HASH are required.")
        return

    print("\nConnecting to Telegram … you'll be asked for your phone + code.\n")
    async with TelegramClient(StringSession(), int(api_id), api_hash) as client:
        me = await client.get_me()
        session_str = client.session.save()
        print("\n" + "=" * 70)
        print(f"✅ Logged in as: {me.first_name} (id {me.id})")
        print("=" * 70)
        print("\nAdd this line to your .env file:\n")
        print(f"TELETHON_STRING_SESSION={session_str}\n")
        print("Keep it SECRET — it grants full access to your Telegram account.")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
