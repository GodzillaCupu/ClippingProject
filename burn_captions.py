#!/usr/bin/env python3
"""
burn_captions.py — word-level karaoke caption burn-in untuk output
AI-Youtube-Shorts-Generator (mode local).

Repo aslinya TIDAK punya caption sama sekali. Script ini menutup gap itu:
transkrip ulang video sumber dengan word timestamps, potong per klip,
bikin file .ass karaoke (kata aktif di-highlight), lalu burn pakai ffmpeg.

Pakai:
    # 1. jalankan generator dulu dengan --output-json
    python main.py "URL" --mode local --num-clips 3 --output-json result.json

    # 2. burn caption ke semua short
    python burn_captions.py result.json --style hormozi

Output: output/short_01_cap.mp4, short_02_cap.mp4, ...

Bisa juga tanpa result.json (satu file, caption dari awal video):
    python burn_captions.py --video output/short_01.mp4 --standalone
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------- style presets

# Warna ASS = &HAABBGGRR (BGR terbalik, bukan RGB!)
STYLES = {
    "hormozi": {  # kuning-hijau nyala, huruf besar semua, outline tebal
        "font": "Arial Black",
        "size": 92,
        "base_color": "&H00FFFFFF",      # putih
        "active_color": "&H0000E5FF",    # kuning-oranye
        "outline_color": "&H00000000",   # hitam
        "outline": 7,
        "shadow": 4,
        "margin_v": 300,
        "uppercase": True,
        "pop": 112,
        "max_words": 3,
        "max_chars": 20,
    },
    "clean": {  # putih rapi, kata aktif sedikit lebih terang, cocok konten edukasi
        "font": "Arial",
        "size": 74,
        "base_color": "&H00D8D8D8",
        "active_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline": 5,
        "shadow": 2,
        "margin_v": 260,
        "uppercase": False,
        "pop": 104,
        "max_words": 5,
        "max_chars": 32,
    },
    "neon": {  # hijau neon, gaya gaming/hype
        "font": "Arial Black",
        "size": 88,
        "base_color": "&H00FFFFFF",
        "active_color": "&H0000FF7A",
        "outline_color": "&H00202020",
        "outline": 6,
        "shadow": 5,
        "margin_v": 320,
        "uppercase": True,
        "pop": 115,
        "max_words": 3,
        "max_chars": 18,
    },
}

# ---------------------------------------------------------------- transkrip kata


def word_cache_path(media_path: str, out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / (Path(media_path).stem + ".words.json")


def transcribe_words(media_path: str, out_dir: str, model_name: str,
                     device: str, language: Optional[str],
                     force: bool = False) -> List[Dict]:
    """faster-whisper dengan word_timestamps=True. Hasil di-cache sebagai JSON.

    Cache .srt bawaan repo tidak dipakai — format SRT tidak bisa menyimpan
    timestamp per kata, jadi kita butuh pass sendiri.
    """
    cache = word_cache_path(media_path, out_dir)
    if cache.exists() and not force:
        if cache.stat().st_mtime >= os.path.getmtime(media_path):
            words = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[words] pakai cache: {cache} ({len(words)} kata)")
            return words

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper belum terinstal. Jalankan: pip install faster-whisper")

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[words] faster-whisper model={model_name} device={device} (word_timestamps=True)")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        media_path,
        language=language,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=True,
    )

    words: List[Dict] = []
    for seg in segments:
        for w in (seg.words or []):
            text = (w.word or "").strip()
            if not text:
                continue
            words.append({"start": float(w.start), "end": float(w.end), "text": text})

    cache.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    print(f"[words] {len(words)} kata → cache: {cache}")
    return words


# ---------------------------------------------------------------- pengelompokan


def group_words(words: List[Dict], max_words: int, max_chars: int,
                max_dur: float = 1.6, gap_break: float = 0.55) -> List[List[Dict]]:
    """Pecah aliran kata jadi baris pendek ala TikTok.

    Baris diputus kalau: kena batas jumlah kata / karakter / durasi,
    ada jeda bicara panjang, atau kata sebelumnya diakhiri tanda baca.
    """
    chunks: List[List[Dict]] = []
    cur: List[Dict] = []

    def flush():
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = []

    for w in words:
        if cur:
            prev = cur[-1]
            chars = sum(len(x["text"]) + 1 for x in cur) + len(w["text"])
            dur = w["end"] - cur[0]["start"]
            if (
                len(cur) >= max_words
                or chars > max_chars
                or dur > max_dur
                or (w["start"] - prev["end"]) > gap_break
                or prev["text"].endswith((".", "!", "?", ",", "…", ":", ";"))
            ):
                flush()
        cur.append(w)
    flush()
    return chunks


# ---------------------------------------------------------------- generator ASS


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def override_color(style_color: str) -> str:
    """&HAABBGGRR (field style) -> &HBBGGRR& (bentuk untuk tag \\c inline)."""
    hexpart = style_color.replace("&H", "").replace("&", "")
    return "&H" + hexpart[-6:] + "&"


def build_ass(chunks: List[List[Dict]], style: Dict,
              play_w: int, play_h: int, karaoke: bool = True) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style['font']},{style['size']},{style['base_color']},{style['active_color']},{style['outline_color']},&H80000000,-1,0,0,0,100,100,0,0,1,{style['outline']},{style['shadow']},2,60,60,{style['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    base = override_color(style["base_color"])
    active = override_color(style["active_color"])
    pop = style["pop"]
    lines: List[str] = []

    for chunk in chunks:
        words = [ass_escape(w["text"]) for w in chunk]
        if style["uppercase"]:
            words = [w.upper() for w in words]

        if not karaoke:
            # satu baris utuh, tanpa highlight per kata
            start, end = chunk[0]["start"], chunk[-1]["end"]
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,"
                f"{{\\fad(60,60)}}{' '.join(words)}"
            )
            continue

        # satu event per kata: seluruh baris tampil, kata aktif diwarnai + membesar
        for i, w in enumerate(chunk):
            start = w["start"]
            end = chunk[i + 1]["start"] if i + 1 < len(chunk) else w["end"]
            if end <= start:
                end = start + 0.08
            parts = []
            for j, token in enumerate(words):
                if j == i:
                    parts.append(
                        f"{{\\c{active}\\fscx{pop}\\fscy{pop}}}{token}"
                        f"{{\\c{base}\\fscx100\\fscy100}}"
                    )
                else:
                    parts.append(token)
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,"
                + " ".join(parts)
            )

    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------- ffmpeg


def probe_size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def ffmpeg_escape(path: str) -> str:
    """Escape path untuk dipakai di dalam filtergraph (penting di Windows)."""
    p = str(Path(path).as_posix())
    p = p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return p


def burn(video: str, ass_path: str, out_path: str,
         scale_w: int, scale_h: int, nvenc: bool = False) -> str:
    vf = (
        f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=decrease,"
        f"pad={scale_w}:{scale_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"ass='{ffmpeg_escape(ass_path)}'"
    )
    if nvenc:
        venc = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "19"]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video, "-vf", vf, *venc,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Burn word-level karaoke captions ke short")
    ap.add_argument("result_json", nargs="?", help="result.json dari --output-json")
    ap.add_argument("--video", help="Mode standalone: burn satu file mp4 saja")
    ap.add_argument("--standalone", action="store_true",
                    help="Perlakukan --video sebagai sumber sendiri (offset 0)")
    ap.add_argument("--style", default="hormozi", choices=list(STYLES.keys()))
    ap.add_argument("--font", help="Override nama font")
    ap.add_argument("--size", type=int, help="Override ukuran font")
    ap.add_argument("--margin-v", type=int, help="Override jarak dari bawah (px @1920)")
    ap.add_argument("--no-karaoke", action="store_true",
                    help="Tampilkan baris utuh tanpa highlight per kata")
    ap.add_argument("--no-uppercase", action="store_true")
    ap.add_argument("--scale", default="1080x1920", help="Resolusi output (default 1080x1920)")
    ap.add_argument("--model", default=os.getenv("LOCAL_WHISPER_MODEL", "small"))
    ap.add_argument("--device", default=os.getenv("LOCAL_WHISPER_DEVICE", "cpu"))
    ap.add_argument("--language", default=None, help="Paksa bahasa, mis. 'id'")
    ap.add_argument("--out-dir", default=os.getenv("LOCAL_OUTPUT_DIR", "output"))
    ap.add_argument("--nvenc", action="store_true", help="Encode pakai GPU NVIDIA")
    ap.add_argument("--keep-ass", action="store_true", help="Jangan hapus file .ass")
    ap.add_argument("--refresh", action="store_true", help="Abaikan cache kata")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cpu"

    style = dict(STYLES[args.style])
    if args.font:
        style["font"] = args.font
    if args.size:
        style["size"] = args.size
    if args.margin_v is not None:
        style["margin_v"] = args.margin_v
    if args.no_uppercase:
        style["uppercase"] = False

    try:
        sw, sh = (int(x) for x in args.scale.lower().split("x"))
    except ValueError:
        sys.exit("--scale harus format WxH, mis. 1080x1920")

    # Kumpulkan daftar job: (video_klip, path_sumber, offset_detik, out_path)
    jobs = []
    if args.video:
        out = str(Path(args.video).with_name(Path(args.video).stem + "_cap.mp4"))
        jobs.append((args.video, args.video, 0.0, out))
    elif args.result_json:
        data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
        if data.get("mode") != "local":
            sys.exit("result.json bukan hasil --mode local (klip mode api ada di URL remote).")
        source = data["source_video_url"]
        if not os.path.exists(source):
            sys.exit(f"File sumber tidak ditemukan: {source}")
        for s in data.get("shorts", []):
            clip = s.get("clip_url")
            if not clip or not os.path.exists(clip):
                print(f"[skip] klip tidak ada: {clip}")
                continue
            out = str(Path(clip).with_name(Path(clip).stem + "_cap.mp4"))
            jobs.append((clip, source, float(s["start_time"]), out))
    else:
        sys.exit("Kasih result.json, atau pakai --video FILE.mp4")

    if not jobs:
        sys.exit("Tidak ada klip untuk diproses.")

    # Transkrip kata — sekali saja per file sumber
    word_cache: Dict[str, List[Dict]] = {}
    for _clip, source, _off, _out in jobs:
        if source not in word_cache:
            word_cache[source] = transcribe_words(
                source, args.out_dir, args.model, args.device,
                args.language, force=args.refresh,
            )

    for clip, source, offset, out_path in jobs:
        words_all = word_cache[source]
        clip_w, clip_h = probe_size(clip)
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip],
            capture_output=True, text=True, check=True,
        ).stdout.strip())

        # Ambil kata yang jatuh di dalam rentang klip, lalu geser ke waktu klip
        lo, hi = offset, offset + dur
        words = []
        for w in words_all:
            if w["end"] <= lo or w["start"] >= hi:
                continue
            words.append({
                "start": max(0.0, w["start"] - lo),
                "end": min(dur, w["end"] - lo),
                "text": w["text"],
            })

        if not words:
            print(f"[skip] tidak ada kata di rentang {clip}")
            continue

        chunks = group_words(words, style["max_words"], style["max_chars"])
        ass_text = build_ass(chunks, style, sw, sh, karaoke=not args.no_karaoke)
        ass_path = str(Path(out_path).with_suffix(".ass"))
        Path(ass_path).write_text(ass_text, encoding="utf-8")

        print(f"[burn] {clip} → {out_path}  ({len(words)} kata, {len(chunks)} baris)")
        burn(clip, ass_path, out_path, sw, sh, nvenc=args.nvenc)

        if not args.keep_ass:
            os.remove(ass_path)

    print("Selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
