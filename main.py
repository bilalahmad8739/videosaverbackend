# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from urllib.parse import quote
import yt_dlp
import re
import os
import tempfile
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

# Cookies file path (put cookies.txt in same folder as main.py)
COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cookies.txt")

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
    temp_dir = tempfile.gettempdir()
    out_file = os.path.join(temp_dir, f"video_{uuid.uuid4().hex[:8]}.mp4")

    fmt = f"{format_id}/bestvideo+bestaudio/best" if format_id and format_id != "best" else "bestvideo+bestaudio/best"

    ydl_opts = {
        "format": fmt,
        "outtmpl": out_file,
        # Video-only + audio-only streams get merged; keep the result as .mp4
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    if os.path.exists(COOKIES_FILE):
        ydl_opts["cookiefile"] = COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

    if not os.path.exists(out_file):
        raise HTTPException(status_code=500, detail="Downloaded video file could not be created.")

    def cleanup():
        try:
            if os.path.exists(out_file):
                os.remove(out_file)
        except Exception:
            pass

    return FileResponse(
        out_file,
        media_type="video/mp4",
        filename="video.mp4",
        background=BackgroundTask(cleanup),
    )


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
        raise HTTPException(status_code=400, detail=f"Could not extract video: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    raw_formats = info.get("formats", [])
    formats = []
    seen_qualities = set()

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

    if not formats:
        raise HTTPException(status_code=404, detail="No downloadable video formats found.")

    def quality_sort_key(f: VideoFormat):
        try:
            return int(re.sub(r"[^0-9]", "", f.quality) or 0)
        except:
            return 0

    formats.sort(key=quality_sort_key, reverse=True)

    duration_sec = info.get("duration", 0) or 0
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    return VideoResponse(
        title=info.get("title", "Unknown Title"),
        thumbnail=info.get("thumbnail", ""),
        duration=duration_str,
        uploader=info.get("uploader", "Unknown"),
        platform=info.get("extractor_key", "Unknown"),
        video_url=url,
        formats=formats,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
