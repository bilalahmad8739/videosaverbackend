# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from urllib.parse import quote, urlparse
import yt_dlp
import hashlib
import ipaddress
import re
import os
import shutil
import socket
import tempfile
import threading
import time
import uuid

app = FastAPI(title="VideoSaver API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# Blocked platforms
# -----------------------------------------
BLOCKED_DOMAINS = ["youtube.com", "youtu.be"]

def is_blocked(url: str) -> bool:
    return any(domain in url.lower() for domain in BLOCKED_DOMAINS)

def is_safe_url(url: str) -> bool:
    """Only public http(s) links: stops the server being used to reach
    internal addresses (localhost, cloud metadata, private network)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        for info in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except Exception:
        return False

# Cookies file path (put cookies.txt in same folder as main.py)
COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cookies.txt")

# ffmpeg is needed to merge video-only + audio streams. Render's Python
# runtime has no system ffmpeg, so use the binary bundled with imageio-ffmpeg.
try:
    import imageio_ffmpeg
    FFMPEG_LOCATION = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_LOCATION = shutil.which("ffmpeg")
print(f"Using ffmpeg: {FFMPEG_LOCATION}")

# Finished proxy downloads are cached here (one folder per video+format),
# and kept until unused for STALE_DOWNLOAD_SECONDS so paused downloads can resume
DOWNLOAD_ROOT = os.path.join(tempfile.gettempdir(), "videosaver_downloads")
STALE_DOWNLOAD_SECONDS = 60 * 60
BUILDING_PREFIX = "building-"
LAST_ACCESS_MARKER = ".last_access"
MAX_FILESIZE_BYTES = 1024 * 1024 * 1024  # 1 GB - protects Render's small disk
FORMAT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-+.]{1,100}$")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


_build_locks: dict[str, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def _get_build_lock(cache_key: str) -> threading.Lock:
    """One lock per video+format so two requests never build the same file at once."""
    with _build_locks_guard:
        return _build_locks.setdefault(cache_key, threading.Lock())


def _mark_accessed(cache_dir: str):
    # A separate marker file: touching video.mp4 itself would change its
    # mtime and therefore its ETag, which would break If-Range resumes
    marker = os.path.join(cache_dir, LAST_ACCESS_MARKER)
    with open(marker, "w"):
        pass


def cleanup_stale_downloads():
    """Remove cached videos unused for an hour, and failed/abandoned builds."""
    if not os.path.isdir(DOWNLOAD_ROOT):
        return
    now = time.time()
    for name in os.listdir(DOWNLOAD_ROOT):
        path = os.path.join(DOWNLOAD_ROOT, name)
        marker = os.path.join(path, LAST_ACCESS_MARKER)
        try:
            last_used = os.path.getmtime(marker if os.path.exists(marker) else path)
            if now - last_used > STALE_DOWNLOAD_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass


def friendly_error(err: Exception) -> str:
    """Turn yt-dlp's technical errors into a short message for app users."""
    msg = ANSI_PATTERN.sub("", str(err))
    lower = msg.lower()
    if "unsupported url" in lower:
        return "This link is not supported. Please use a video link from a supported site."
    if "private" in lower:
        return "This video is private and cannot be downloaded."
    # Checked before "login": Instagram says "rate-limit reached or login required"
    if "rate-limit" in lower or "rate limit" in lower or "429" in lower:
        return "This site is limiting requests right now. Please try again in a few minutes."
    if any(k in lower for k in ("login", "logged-in", "authentication", "sign in")):
        return "This video needs a login to view, so it cannot be downloaded."
    if "geo" in lower or "deleted" in lower:
        return "This video was removed or is not available in our server's region."
    if "file is larger than max-filesize" in lower or "max-filesize" in lower:
        return "This video is too large to download (limit 1 GB)."
    if "404" in lower or "not available" in lower or "removed" in lower:
        return "Video not found. It may have been removed."
    # Strip "ERROR: [Extractor] id: " prefix and yt-dlp CLI advice
    msg = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*[^:]*:\s*)?", "", msg)
    msg = msg.split(" Use --")[0].split("; please report")[0]
    return msg.strip() or "Could not process this video."

# -----------------------------------------
# Request / Response models
# -----------------------------------------
class ExtractRequest(BaseModel):
    url: str

class VideoFormat(BaseModel):
    quality: str
    url: str
    ext: str
    filesize: str

class VideoResponse(BaseModel):
    title: str
    thumbnail: str
    duration: str
    uploader: str
    platform: str
    video_url: str
    formats: list[VideoFormat]


# -----------------------------------------
# Health check
# -----------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "VideoSaver API is running"}

@app.get("/ping")
def ping():
    return {"status": "alive"}


# -----------------------------------------
# Stream / M3U8 / Protected CDN Download Proxy
# -----------------------------------------
@app.get("/download-proxy")
def download_proxy(video_url: str, format_id: str = "best"):
    if is_blocked(video_url):
        raise HTTPException(status_code=403, detail="YouTube is not supported.")
    if not is_safe_url(video_url):
        raise HTTPException(status_code=400, detail="Invalid video link.")
    if format_id and format_id != "best" and not FORMAT_ID_PATTERN.match(format_id):
        raise HTTPException(status_code=400, detail="Invalid format.")

    cleanup_stale_downloads()

    # The finished file is cached per (video, format) so a paused download
    # can resume: the next request gets the exact same bytes (and ETag)
    # instead of a freshly rebuilt file that would differ.
    cache_key = hashlib.sha256(f"{video_url}|{format_id}".encode()).hexdigest()[:32]
    cache_dir = os.path.join(DOWNLOAD_ROOT, cache_key)
    cached_file = os.path.join(cache_dir, "video.mp4")

    with _get_build_lock(cache_key):
        if not os.path.exists(cached_file):
            _build_video(video_url, format_id, cache_dir, cached_file)
        _mark_accessed(cache_dir)

    # FileResponse serves Range requests (206) and honours If-Range with the
    # ETag, so a client resuming against a rebuilt file gets the full file
    return FileResponse(cached_file, media_type="video/mp4", filename="video.mp4")


def _build_video(video_url: str, format_id: str, cache_dir: str, cached_file: str):
    # Build in a private folder so yt-dlp's partial/intermediate files
    # (.part, .f137.mp4 before merging) never appear in the cache
    work_dir = os.path.join(DOWNLOAD_ROOT, f"{BUILDING_PREFIX}{uuid.uuid4().hex}")
    os.makedirs(work_dir, exist_ok=True)
    out_file = os.path.join(work_dir, "video.mp4")

    def cleanup():
        shutil.rmtree(work_dir, ignore_errors=True)

    fmt = f"{format_id}/bestvideo+bestaudio/best" if format_id and format_id != "best" else "bestvideo+bestaudio/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": out_file,
        # Video-only + audio-only streams get merged; keep the result as .mp4
        "merge_output_format": "mp4",
        "max_filesize": MAX_FILESIZE_BYTES,
        "quiet": True,
        "no_warnings": True,
    }
    if FFMPEG_LOCATION:
        ydl_opts["ffmpeg_location"] = FFMPEG_LOCATION
    if os.path.exists(COOKIES_FILE):
        ydl_opts["cookiefile"] = COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        cleanup()
        raise HTTPException(status_code=500, detail=f"Download failed: {friendly_error(e)}")

    # yt-dlp skips (without raising) files over max_filesize
    if not os.path.exists(out_file):
        cleanup()
        raise HTTPException(
            status_code=500,
            detail="Downloaded video file could not be created. It may be larger than 1 GB.",
        )

    os.makedirs(cache_dir, exist_ok=True)
    os.replace(out_file, cached_file)
    cleanup()


# -----------------------------------------
# Main extract endpoint
# -----------------------------------------
@app.post("/extract", response_model=VideoResponse)
def extract_video(request: ExtractRequest):
    url = request.url.strip()

    # Block YouTube
    if is_blocked(url):
        raise HTTPException(
            status_code=403,
            detail="YouTube is not supported. Please use Instagram, TikTok, Facebook, Twitter, or other platforms."
        )
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Please enter a valid video link (http or https).")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }

    # Use cookies file if it exists
    if os.path.exists(COOKIES_FILE):
        ydl_opts["cookiefile"] = COOKIES_FILE
        print(f"Using cookies from: {COOKIES_FILE}")
    else:
        print("No cookies.txt found - running without cookies")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=friendly_error(e))
    except Exception as e:
        print(f"Extract failed for {url}: {e}")
        raise HTTPException(status_code=500, detail="Server error. Please try again.")

    # Profile/story links come back as a playlist with no video of their own
    if info.get("_type") == "playlist" or (info.get("entries") and not info.get("formats")):
        raise HTTPException(
            status_code=400,
            detail="This link is a profile or playlist. Please share a single video link.",
        )

    # Some extractors (e.g. Snapchat Spotlight) return one direct video URL
    # on the result itself instead of a formats list
    raw_formats = info.get("formats") or ([info] if info.get("url") else [])
    formats = []
    heights = {}  # format url -> pixel height, used for sorting
    seen_qualities = set()

    # Size of the audio track that gets merged into video-only formats
    audio_sizes = [
        a.get("filesize") or a.get("filesize_approx") or 0
        for a in raw_formats
        if a.get("vcodec") == "none" and a.get("acodec") not in (None, "none")
    ]
    best_audio_size = max(audio_sizes, default=0)

    for f in raw_formats:
        raw_url = f.get("url")
        if not raw_url:
            continue
        if f.get("vcodec") == "none":
            continue

        quality = f.get("format_note") or f.get("height") or "Unknown"
        quality = str(quality)
        ext = f.get("ext", "mp4")
        filesize_bytes = f.get("filesize") or f.get("filesize_approx") or 0
        format_id = f.get("format_id", "")

        # Higher qualities are often video-only (DASH) with audio in a separate
        # stream. Ask yt-dlp to merge the best audio in, or the file is silent.
        if format_id and f.get("acodec") == "none":
            format_id = f"{format_id}+bestaudio"
            ext = "mp4"  # merged output is always mp4 (merge_output_format)
            if filesize_bytes:
                filesize_bytes += best_audio_size

        if filesize_bytes and filesize_bytes > 0:
            if filesize_bytes >= 1_000_000:
                filesize = f"{filesize_bytes / 1_000_000:.1f} MB"
            elif filesize_bytes >= 1_000:
                filesize = f"{filesize_bytes / 1_000:.1f} KB"
            else:
                filesize = f"{filesize_bytes} B"
        else:
            filesize = "Unknown size"

        key = f"{quality}-{ext}"
        if key in seen_qualities:
            continue
        seen_qualities.add(key)

        # Route through backend download proxy so all platforms (TikTok, Dailymotion, Instagram, etc.)
        # are downloaded with the exact cookies/headers/muxing without 403 Forbidden errors
        download_url = f"/download-proxy?video_url={quote(url)}&format_id={quote(str(format_id))}"

        formats.append(VideoFormat(
            quality=quality,
            url=download_url,
            ext=ext,
            filesize=filesize,
        ))
        heights[download_url] = f.get("height") or 0

    if not formats:
        raise HTTPException(status_code=404, detail="No downloadable video formats found.")

    def quality_sort_key(f: VideoFormat):
        # Prefer the real pixel height; parsing "1080p60" as text gives 108060
        if heights.get(f.url):
            return heights[f.url]
        match = re.search(r"\d+", f.quality)
        return int(match.group()) if match else 0

    formats.sort(key=quality_sort_key, reverse=True)

    duration_sec = info.get("duration", 0) or 0
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    # `or` (not a .get default): sites often return these keys set to None,
    # which would fail response validation with a 500
    return VideoResponse(
        title=info.get("title") or "Unknown Title",
        thumbnail=info.get("thumbnail") or "",
        duration=duration_str,
        uploader=info.get("uploader") or "Unknown",
        platform=info.get("extractor_key") or "Unknown",
        video_url=url,
        formats=formats,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
