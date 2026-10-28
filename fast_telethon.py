"""
Chunk-parallel MTProto downloader.

Telethon's own ``download_media`` walks a file one request at a time. This
helper splits the byte range into ``upload.getFile`` requests, keeps a few in
flight and writes every chunk straight to its offset on disk, so a 2 GB (or
4 GB premium) document never has to fit in RAM.

It is a *fast path* only: any media it does not understand, anything smaller
than one chunk, and any failure falls back to Telethon's sequential downloader.
"""

from __future__ import annotations

import asyncio
import logging
import os

from telethon import errors
from telethon.tl.functions.upload import GetFileRequest
from telethon.tl.types import (
    InputDocumentFileLocation,
    InputPhotoFileLocation,
    MessageMediaDocument,
    MessageMediaPhoto,
    MessageMediaWebPage,
)

from media import photo_size_bytes, pick_photo_size, thumb_size_bytes

log = logging.getLogger(__name__)

# upload.getFile allows at most 1 MB per request and requires both the offset
# and the limit to be multiples of 4096 — the planner below rounds them.
CHUNK_SIZE = 1024 * 1024
WORKERS = 4
_MAX_ATTEMPTS = 5


class _State:
    """Per-download mutable state shared by the chunk workers."""

    __slots__ = ("client", "location", "lock")

    def __init__(self, client, location):
        self.client = client
        self.location = location
        self.lock = asyncio.Lock()


def _unlink(path) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _resolve_location(message, thumb=None):
    """
    Return ``(input_location, size)`` for the exact object being downloaded.

    The size must belong to the same PhotoSize the location points at: chunking
    a 3 MB photo while requesting its 320 px thumbnail asks for offsets past
    EOF and silently produces a corrupt file.
    """
    media = getattr(message, "media", None)
    if isinstance(media, MessageMediaWebPage):
        page = getattr(media, "webpage", None)
        media = getattr(page, "photo", None) or getattr(page, "document", None)
    if isinstance(media, MessageMediaPhoto):
        photo = media.photo
        stype = thumb or pick_photo_size(photo)
        if not photo or not stype:
            return None, 0
        size = photo_size_bytes(photo, stype)
        if not size:
            return None, 0
        return InputPhotoFileLocation(
            id=photo.id, access_hash=photo.access_hash,
            file_reference=photo.file_reference, thumb_size=stype,
        ), size
    if isinstance(media, MessageMediaDocument):
        doc = media.document
        if not doc:
            return None, 0
        if thumb:
            size = thumb_size_bytes(doc, thumb)
            if not size:
                return None, 0
        else:
            size = int(getattr(doc, "size", 0) or 0)
            if not size:
                return None, 0
        return InputDocumentFileLocation(
            id=doc.id, access_hash=doc.access_hash,
            file_reference=doc.file_reference, thumb_size=thumb or "",
        ), size
    return None, 0


async def download_file(client, message, output_path, workers=WORKERS, thumb=None):
    """
    Download a message's media to ``output_path``.

    ``thumb`` selects a specific thumbnail type (photos: ``PhotoSize.type``,
    documents: ``doc.thumbs[].type``). Falls back to Telethon when the media is
    not a plain photo/document, when it is smaller than one chunk, or when the
    parallel path fails for any reason.
    """
    location, size = _resolve_location(message, thumb)
    if location is None or size < CHUNK_SIZE:
        await message.download_media(file=output_path, thumb=thumb)
        return output_path

    state = _State(client, location)

    async def refresh():
        """Re-fetch the message so the short-lived file_reference is new."""
        try:
            peer = getattr(message, "chat_id", None) or getattr(message, "peer_id", None)
            fresh = await client.get_messages(peer, ids=message.id)
            fresh = fresh[0] if isinstance(fresh, list) else fresh
            if fresh is not None:
                new_location, _ = _resolve_location(fresh, thumb)
                return new_location
        except Exception as exc:
            log.debug("file_reference refresh failed: %s", exc)
        return None

    try:
        await _parallel_download(state, size, output_path, workers, refresh)
    except asyncio.CancelledError:
        # A cancelled download leaves a partially written file behind.
        _unlink(output_path)
        raise
    except Exception as exc:
        log.warning("Parallel download failed (%s); falling back to Telethon", exc)
        _unlink(output_path)
        await message.download_media(file=output_path, thumb=thumb)
    return output_path


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    """Write ``data`` at ``offset`` without touching the shared file position."""
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise IOError("pwrite returned 0")
        view = view[written:]
        offset += written


async def _parallel_download(state, size, output_path, workers, refresh=None):
    chunks = []
    offset = 0
    while offset < size:
        limit = min(CHUNK_SIZE, size - offset)
        limit = max(4096, (limit + 4095) // 4096 * 4096)
        chunks.append((offset, limit))
        offset += limit
    if not chunks:
        raise ValueError("empty download")

    sem = asyncio.Semaphore(max(1, int(workers)))
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        # Pre-allocate so out-of-order chunk writes always land in range.
        os.ftruncate(fd, size)

        async def worker(off, limit):
            async with sem:
                data = await _get_chunk(state, off, limit, refresh)
                if len(data) != limit:
                    raise IOError(
                        f"short read at offset {off}: {len(data)}/{limit} bytes")
                _pwrite_all(fd, data, off)

        results = await asyncio.gather(
            *[worker(off, limit) for off, limit in chunks],
            return_exceptions=True)
        failure = next((r for r in results if isinstance(r, BaseException)), None)
        if failure is not None:
            raise failure
        written = os.fstat(fd).st_size
        if written != size:
            raise IOError(f"incomplete download: {written}/{size} bytes")
    finally:
        os.close(fd)


async def _get_chunk(state, offset, limit, refresh=None):
    last = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            result = await state.client(GetFileRequest(
                location=state.location, offset=offset, limit=limit))
            return result.bytes
        except errors.FloodWaitError as exc:
            last = exc
            wait = max(int(exc.seconds), 1)
            log.info("GetFile flood wait %ds (offset=%d)", wait, offset)
            await asyncio.sleep(wait)
        except errors.FileReferenceExpiredError as exc:
            last = exc
            if refresh is None:
                raise
            async with state.lock:
                new_location = await refresh()
            if new_location is None:
                raise
            state.location = new_location
        except errors.FileMigrateError as exc:
            last = exc
            async with state.lock:
                try:
                    state.client = await state.client._get_exported_client(exc.new_dc)
                except AttributeError:
                    raise
        except (errors.ServerError, errors.TimedOutError) as exc:
            last = exc
            await asyncio.sleep(min(2 ** attempt, 30))
    raise last if last is not None else IOError("GetFile failed")
