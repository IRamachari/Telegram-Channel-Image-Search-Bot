from __future__ import annotations
import asyncio
import logging
import os

from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaPhoto,
    MessageMediaDocument,
    InputDocumentFileLocation,
    InputPhotoFileLocation,
)
from telethon.tl.functions.upload import GetFileRequest

log = logging.getLogger(__name__)

CHUNK_SIZE = 512 * 1024
WORKERS = 4


async def download_file(client, message, output_path, workers=WORKERS):
    if not message.media:
        raise ValueError("No media")
    size = _get_file_size(message)
    if size is None or size < CHUNK_SIZE * 2:
        await message.download_media(file=output_path)
        return output_path
    location = _get_input_location(message)
    if location is None:
        await message.download_media(file=output_path)
        return output_path
    try:
        await _parallel_download(client, location, size, output_path, workers)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        await message.download_media(file=output_path)
    return output_path


def _get_file_size(message):
    media = message.media
    if isinstance(media, MessageMediaPhoto):
        if media.photo and media.photo.sizes:
            for s in reversed(media.photo.sizes):
                sz = getattr(s, "size", None)
                if sz:
                    return sz
                szs = getattr(s, "sizes", None)
                if szs:
                    return max(szs)
        return None
    if isinstance(media, MessageMediaDocument):
        doc = media.document
        return doc.size if doc else None
    return None


def _get_input_location(message):
    media = message.media
    if isinstance(media, MessageMediaPhoto):
        photo = media.photo
        if not photo:
            return None
        # thumb_size="" is only valid for documents ("the whole document").
        # For photos Telegram requires the `type` of the PhotoSize to download;
        # an empty string yields FILE_ID_INVALID and every photo download fails.
        from indexer import _pick_photo_size
        thumb = _pick_photo_size(message)
        if not thumb:
            return None
        return InputPhotoFileLocation(
            id=photo.id, access_hash=photo.access_hash,
            file_reference=photo.file_reference, thumb_size=thumb,
        )
    if isinstance(media, MessageMediaDocument):
        doc = media.document
        if not doc:
            return None
        return InputDocumentFileLocation(
            id=doc.id, access_hash=doc.access_hash,
            file_reference=doc.file_reference, thumb_size="",
        )
    return None


async def _parallel_download(client, location, size, output_path, workers):
    chunks = []
    offset = 0
    while offset < size:
        chunk_size = min(CHUNK_SIZE, size - offset)
        chunk_size = max(4096, (chunk_size + 4095) // 4096 * 4096)
        chunks.append((offset, chunk_size))
        offset += chunk_size
    results = {}
    sem = asyncio.Semaphore(workers)

    async def dl(off, lim):
        async with sem:
            r = await client(GetFileRequest(location=location, offset=off, limit=lim))
            results[off] = r.bytes

    await asyncio.gather(*[dl(o, l) for o, l in chunks])
    with open(output_path, "wb") as f:
        for off, _ in chunks:
            f.write(results[off])
    if os.path.getsize(output_path) > size:
        with open(output_path, "r+b") as f:
            f.truncate(size)
