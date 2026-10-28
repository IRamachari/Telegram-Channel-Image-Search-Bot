"""
Shared media classification for the bulk indexer and the real-time monitor.

Both paths must agree on what counts as an image. They used to disagree: the
monitor accepted anything whose mime type started with ``image/`` and therefore
indexed webp/svg stickers the indexer deliberately skips — and advanced the
channel watermark over them. This module is the single source of truth for
"is this message an image", for which thumbnail to download, and for the
scratch-file naming/cleanup rules.
"""

from __future__ import annotations

import logging
import os
import re
import time

from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto,
    MessageMediaWebPage,
    PhotoCachedSize,
    PhotoStrippedSize,
)

log = logging.getLogger(__name__)

MIN_PHOTO_SIZE = 224

# MIME types worth embedding as-is. A sticker/webp document also matches
# "image/", but stickers are not channel photos and indexing them pollutes the
# counts and the duplicate finder, so they are excluded separately below.
_IMAGE_MIME = ("image/jpeg", "image/png", "image/webp", "image/bmp",
               "image/gif", "image/tiff", "image/heic", "image/heif")

# Documents whose *thumbnail* is worth embedding: videos and animations (a GIF
# arrives as an mp4 with DocumentAttributeAnimated) have no still image PIL can
# open, but Telegram always renders a poster frame for them.
_VIDEO_MIME_PREFIX = "video/"

_STICKER_ATTRS = frozenset(("DocumentAttributeSticker",))
_POSTER_ATTRS = frozenset((
    "DocumentAttributeAnimated",
    "DocumentAttributeVideo",
    "DocumentAttributeImageSize",
))

# Scratch files are named ``<channel>_<message>.jpg`` (monitor) or
# ``idx<pid>_<channel>_<message>.jpg`` (indexer).
_TEMP_RE = re.compile(r"^(?:idx\d+_)?\d+_\d+\.jpg$")
TEMP_MAX_AGE_S = 6 * 3600


def _attr_names(doc) -> set[str]:
    return {type(a).__name__ for a in (getattr(doc, "attributes", None) or [])}


def photo_sizes(photo) -> list[tuple[str, int, int]]:
    """Usable ``(type, max_side_px, bytes)`` triples for a Photo, smallest first."""
    out: list[tuple[str, int, int]] = []
    for s in getattr(photo, "sizes", None) or []:
        if isinstance(s, (PhotoStrippedSize, PhotoCachedSize)):
            continue
        w = getattr(s, "w", 0)
        h = getattr(s, "h", 0)
        nbytes = getattr(s, "size", None)
        if not nbytes:
            progressive = getattr(s, "sizes", None)
            nbytes = max(progressive) if progressive else 0
        if w > 0 and h > 0 and nbytes:
            out.append((s.type, max(w, h), int(nbytes)))
    out.sort(key=lambda item: item[1])
    return out


def pick_photo_size(photo, min_px: int = MIN_PHOTO_SIZE):
    """Smallest PhotoSize that is still >= min_px, else the largest one."""
    usable = photo_sizes(photo)
    if not usable:
        return None
    for stype, side, _ in usable:
        if side >= min_px:
            return stype
    return usable[-1][0]


def photo_size_bytes(photo, stype=None) -> int:
    """
    Byte size of one specific PhotoSize (0 when it does not exist).

    Returning 0 rather than silently falling back to the largest size keeps a
    chunk plan from ever being computed against a different object than the
    one being downloaded.
    """
    usable = photo_sizes(photo)
    if not usable:
        return 0
    if stype is None:
        return usable[-1][2]
    for name, _, nbytes in usable:
        if name == stype:
            return nbytes
    return 0


def thumb_size_bytes(obj, stype) -> int:
    """Byte size of a thumbnail (a PhotoSize from ``doc.thumbs``), 0 if absent."""
    for s in getattr(obj, "thumbs", None) or []:
        if getattr(s, "type", None) == stype:
            nbytes = getattr(s, "size", None)
            if not nbytes:
                progressive = getattr(s, "sizes", None)
                nbytes = max(progressive) if progressive else 0
            return int(nbytes or 0)
    return 0


def best_thumb_type(doc):
    """Largest still thumbnail available on a document, or None."""
    best = None
    best_side = -1
    for s in getattr(doc, "thumbs", None) or []:
        w = getattr(s, "w", 0)
        h = getattr(s, "h", 0)
        if w > 0 and h > 0 and max(w, h) > best_side:
            best_side = max(w, h)
            best = getattr(s, "type", None)
    if best is None:
        # Some documents only carry a stripped preview; Telethon can still
        # download it by type string, which is all we need.
        for s in getattr(doc, "thumbs", None) or []:
            stype = getattr(s, "type", None)
            if stype:
                return stype
    return best


def _document_kind(doc) -> str:
    """``image`` | ``thumb`` | ``sticker`` | ``other`` for a document."""
    if not doc:
        return "other"
    attrs = _attr_names(doc)
    mime = (getattr(doc, "mime_type", "") or "").lower()
    if attrs & _STICKER_ATTRS or mime == "application/x-tgsticker":
        return "sticker"
    if mime in _IMAGE_MIME:
        return "image"
    # Video / animated GIF / image uploaded without a usable mime: embed the
    # poster frame instead of skipping the message.
    if attrs & _POSTER_ATTRS or mime.startswith(_VIDEO_MIME_PREFIX) or not mime:
        if best_thumb_type(doc):
            return "thumb"
    return "other"


def media_kind(msg) -> str:
    """Classify a message's media for stats and for the download path."""
    media = getattr(msg, "media", None)
    if media is None:
        return "none"
    if isinstance(media, MessageMediaPhoto):
        return "photo"
    if isinstance(media, MessageMediaDocument):
        return _document_kind(media.document)
    if isinstance(media, MessageMediaWebPage):
        return "webpage"
    if type(media).__name__ == "MessageMediaPaidMedia":
        # Paid (Stars) posts are not downloadable without a purchase; they are
        # counted as skipped instead of vanishing silently.
        return "paid"
    return "other"


def is_image(msg) -> bool:
    """True when this message should be embedded."""
    return media_kind(msg) in ("photo", "image", "thumb")


def download_thumb_type(msg):
    """
    Thumbnail type to pass to ``download_media(thumb=...)``.

    ``None`` means "download the media itself". Photos always use a sized
    thumbnail (bandwidth), documents only when their own bytes are not an
    embeddable image (video/GIF poster frames).
    """
    media = getattr(msg, "media", None)
    if isinstance(media, MessageMediaPhoto):
        return pick_photo_size(media.photo)
    if isinstance(media, MessageMediaDocument) and media_kind(msg) == "thumb":
        return best_thumb_type(media.document)
    return None


def media_size(msg) -> int:
    """Bytes that will actually be downloaded for this message."""
    media = getattr(msg, "media", None)
    kind = media_kind(msg)
    if kind == "photo":
        return photo_size_bytes(media.photo, pick_photo_size(media.photo))
    if kind == "image":
        return int(getattr(media.document, "size", 0) or 0)
    if kind == "thumb":
        doc = media.document
        return (thumb_size_bytes(doc, best_thumb_type(doc))
                or int(getattr(doc, "size", 0) or 0))
    return 0


def sweep_temp_dir(temp_dir, max_age_s: int = TEMP_MAX_AGE_S) -> int:
    """
    Delete stale scratch files left behind by a crash, OOM or SIGKILL.

    Age-gated on purpose: ``/cleanup`` constructs a ChannelMonitor while the
    long-lived monitor is running, and a fresh file may be a download that is
    still in flight.
    """
    now = time.time()
    removed = 0
    try:
        entries = os.scandir(temp_dir)
    except OSError:
        return 0
    with entries:
        for entry in entries:
            try:
                if not entry.is_file() or not _TEMP_RE.match(entry.name):
                    continue
                if now - entry.stat().st_mtime < max_age_s:
                    continue
                os.remove(entry.path)
                removed += 1
            except OSError:
                continue
    if removed:
        log.info("Swept %d stale temp file(s) from %s", removed, temp_dir)
    return removed
