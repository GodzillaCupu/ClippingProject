# AI YouTube Shorts Generator — Review & Setup Guide

Repo: https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator
Ditinjau: 10 September 2026 · commit terakhir: 8 September 2026

---

## TL;DR

| Pertanyaan | Jawaban singkat |
|---|---|
| Legit? | Ya. Kode nyata, jalan, tidak ada yang mencurigakan. Tapi ini juga **funnel marketing** untuk produk berbayar penulisnya (MuAPI). |
| Worth it? | Worth dicoba lewat **`--mode local`**. Mode `api` (default) = bayar ke MuAPI, tidak lebih murah dari kompetitor. |
| Gratis beneran? | Tidak 100%. Selalu butuh minimal 1 API key LLM (OpenAI/Gemini) atau MuAPI. Yang gratis: download, transkrip, crop, render. |
| Jalan di MacBook M-series? | **Ya, jalan.** Full ARM64-native. Tidak butuh CUDA/torch. Hanya lebih lambat karena Whisper jalan di CPU (tidak pakai GPU Metal). |
| Windows + RTX 3050 Ti? | **Ya, dan ini jalur tercepat.** Whisper pakai CUDA float16. Butuh install cuBLAS/cuDNN manual, dan VRAM 4 GB membatasi model maksimal di `medium`. Lihat bagian 5B. |
| Kekurangan terbesar | **Tidak ada subtitle/caption burn-in.** Fitur utama OpusClip. Ditambal sendiri lewat `burn_captions.py` — lihat bagian 9B. |

---

## 1. Apa Ini Sebenarnya

Tool CLI Python. Kasih URL YouTube → keluar N video vertikal 9:16 siap upload ke TikTok/Reels/Shorts.

Pipeline:

```
Download video → Transkrip (Whisper) → Klasifikasi tipe konten (LLM)
   → LLM ranking highlight (skor viral 0-100) → Dedupe overlap
   → Potong ffmpeg → Crop vertikal face-tracking → mp4
```

Tiap klip keluar dengan: `score`, `title`, `hook_sentence` (kalimat pembuka), dan alasan kenapa dianggap viral.

---

## 2. Legit atau Tidak

### Yang bikin percaya

- **4.919 star, 902 fork**, aktif sejak Juni 2024, commit terakhir 2 hari lalu (8 Sep 2026).
- Ada kontributor eksternal yang PR-nya di-merge (contoh: `LathissKhumar` — fix VAD & CUDA fallback).
- Kode bisa dibaca semua, kecil (~1,3 MB), tidak ada blob binary aneh.
- Tidak ada telemetri tersembunyi. Panggilan jaringan hanya ke: YouTube (yt-dlp), OpenAI/Gemini, dan MuAPI — semuanya sesuai fungsi.
- Penulis (Anil Matcha / SamurAI) adalah dev yang dikenal, punya banyak repo AI open-source populer.

### Yang perlu diwaspadai

1. **Konflik kepentingan.** README menjual MuAPI (produk penulis sendiri) di badge, di paragraf pertama, di CTA, dan mode `api` adalah **default**. Jalankan `python main.py <url>` tanpa flag → langsung nagih `MUAPI_API_KEY`.
2. **README overclaim.** Judulnya "free alternative to Opus Clip", tapi tanpa API key tool ini tidak jalan sama sekali. "Free" di sini artinya "tidak ada langganan bulanan", bukan "nol biaya".
3. **Klaim lisensi tidak akurat.** README bilang MIT, tapi **tidak ada file `LICENSE` di repo** dan GitHub API melaporkan `license: null`. Secara hukum, tanpa file lisensi = hak cipta penuh penulis. Kalau kamu mau pakai komersial, minta klarifikasi ke penulis dulu.
4. **Instruksi clone di README salah.** Menunjuk ke `SamurAIGPT/AI-Youtube-Shorts-Generator` (repo lama), bukan `Anil-matcha/...`. Pakai URL yang benar (lihat bagian setup).
5. **Repo dioptimasi untuk SEO/traffic.** Banyak badge, cross-link ke repo lain penulis, link tutorial YouTube. Bukan penipuan, tapi jelas ada motif marketing.

**Verdict: legit, tapi baca README-nya seperti membaca iklan.**

---

## 3. MacBook M-Series — Bisa Jalan?

**Bisa. Tidak ada blocker.** Rincian per komponen:

| Komponen | Status di Apple Silicon | Catatan |
|---|---|---|
| `requests`, `python-dotenv` | Aman | Pure Python |
| `yt-dlp` | Aman | Pure Python |
| `faster-whisper` (CTranslate2) | **Aman, ARM64 native** | Wheel arm64 macOS tersedia |
| `opencv-python` | **Aman, ARM64 native** | Wheel arm64 tersedia |
| `openai` / `google-genai` | Aman | HTTP client saja |
| `ffmpeg` | Aman | `brew install ffmpeg` |
| `torch` | **Tidak perlu** | Dikomentari di requirements, hanya untuk CUDA |

### Kenapa CUDA tidak jadi masalah

Kode di [`shorts_generator/local/transcriber.py`](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator/blob/main/shorts_generator/local/transcriber.py) melakukan:

```python
def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"
```

Di Mac: `import torch` gagal (tidak diinstal) → fallback ke `cpu` dengan `compute_type="int8"`. Aman, tidak error.

### Konsekuensi nyata di M-series

- **Tidak ada akselerasi GPU.** faster-whisper hanya support CPU dan CUDA. Metal/MPS tidak didukung. Whisper jalan di CPU M-series — masih cepat (M1/M2/M3 CPU kuat + int8 quantization), tapi bukan secepat GPU NVIDIA.
- Estimasi kasar transkrip video **60 menit** dengan model `base` di M-series CPU: **~5–12 menit**. Model `small`: ~2–3× lebih lama. Model `large-v3`: bisa 40+ menit dan makan RAM banyak.
- **Rekomendasi Mac**: pakai `LOCAL_WHISPER_MODEL=base` untuk video Inggris, atau `small` kalau akurasi kurang. Hindari `large-v3` kecuali RAM ≥ 32 GB dan kamu sabar.
- Face-tracking OpenCV pakai Haar cascade — ringan di CPU, bukan bottleneck.
- `ffmpeg` dari Homebrew sudah pakai VideoToolbox (hardware encode Apple), jadi tahap render cepat.

**Kesimpulan: M1/M2/M3/M4 semua jalan mulus. Yang M-series korbankan hanya kecepatan transkrip, bukan kompatibilitas.**

---

## 4. Requirement Lengkap

### Wajib

- **Python 3.10+**
- **API key**, minimal salah satu:
  - `MUAPI_API_KEY` → untuk mode `api` (default)
  - `OPENAI_API_KEY` atau `GEMINI_API_KEY` → untuk mode `local`

### Tambahan untuk mode `local` (yang direkomendasikan)

- **ffmpeg** di PATH → `brew install ffmpeg`
- Paket dari `requirements-local.txt`: `yt-dlp`, `faster-whisper`, `openai`, `google-genai`, `opencv-python`
- Disk kosong: video sumber + klip + cache `.srt` + model Whisper (~150 MB untuk `base`, ~3 GB untuk `large-v3`)
- RAM: 8 GB cukup untuk model `base`/`small`. 16 GB+ untuk `medium` ke atas.

### Tidak perlu

- GPU NVIDIA
- CUDA / cuDNN / cuBLAS
- PyTorch
- Docker

---

## 5. Setup di macOS (Apple Silicon)

### Langkah 1 — Prasyarat sistem

```bash
brew install ffmpeg python@3.11
```

### Langkah 2 — Clone

Catatan: URL di README salah. Pakai yang ini:

```bash
git clone https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator.git
```

### Langkah 3 — Virtual environment

```bash
cd AI-Youtube-Shorts-Generator && python3.11 -m venv venv && source venv/bin/activate
```

### Langkah 4 — Install dependency (mode local)

```bash
pip install -r requirements-local.txt
```

### Langkah 5 — Buat file `.env`

Isi file `.env` di root project (mode local, pakai Gemini karena paling murah):

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=isi_key_kamu_disini
GEMINI_MODEL=gemini-2.5-flash
LOCAL_WHISPER_MODEL=base
LOCAL_WHISPER_DEVICE=cpu
LOCAL_OUTPUT_DIR=output
```

Kalau lebih suka OpenAI, ganti jadi `LLM_PROVIDER=openai` + `OPENAI_API_KEY=...` + `OPENAI_MODEL=gpt-4o-mini`.

> `LOCAL_WHISPER_DEVICE=cpu` diset eksplisit supaya di Mac tidak buang waktu mencoba deteksi CUDA. Efeknya sama dengan `auto`, cuma lebih jelas.

### Langkah 6 — Test run

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --num-clips 3
```

Output mendarat di `./output/short_01.mp4`, `short_02.mp4`, dst.

**Penting: selalu tulis `--mode local`.** Tanpa flag itu, default-nya `api` dan akan minta `MUAPI_API_KEY`.

### Pakai file lokal (skip YouTube)

```bash
python main.py "/Users/macery/Movies/podcast.mp4" --mode local
```

### Batch dari daftar URL

```bash
xargs -a urls.txt -I{} python main.py "{}" --mode local
```

### Sebagai library Python

```python
from shorts_generator import generate_shorts

result = generate_shorts(
    "/Users/macery/Movies/podcast.mp4",
    num_clips=5,
    aspect_ratio="9:16",
    mode="local",
)
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

---

## 5B. Setup di Windows (NVIDIA RTX 3050 Ti)

Di Windows + GPU NVIDIA, tool ini justru jalan di jalur tercepatnya: `faster-whisper` bisa pakai CUDA `float16`, jauh lebih ngebut dari CPU Mac.

### 5B.1 Spesifikasi RTX 3050 Ti — yang perlu diperhatikan

RTX 3050 Ti itu **kartu laptop dengan VRAM 4 GB** (bukan 8 GB seperti RTX 3050 desktop). Ini menentukan model Whisper mana yang muat.

Kode di `transcriber.py` **hardcode** `compute_type="float16"` begitu device = `cuda`:

```python
device = _resolve_device()
compute_type = "float16" if device == "cuda" else "int8"
```

Jadi kebutuhan VRAM-nya:

| Model Whisper | VRAM float16 | RTX 3050 Ti (4 GB) | Estimasi transkrip 60 menit |
|---|---|---|---|
| `tiny` | ~0.5 GB | Aman | ~1 menit |
| `base` | ~0.7 GB | Aman | ~1–2 menit |
| `small` | ~1.2 GB | Aman | ~2–3 menit |
| `medium` | ~2.6 GB | Muat, tapi mepet | ~4–7 menit |
| `large-v3` | ~4.7 GB | **Tidak muat → OOM** | — |

Angka estimasi kasar, tergantung TGP laptop (3050 Ti mobile varian 35W–80W beda jauh) dan seberapa banyak VRAM dipakai browser/Windows.

**Rekomendasi: `small` untuk konten Inggris, `medium` untuk Bahasa Indonesia** (kalau tidak ada aplikasi berat lain yang buka). Tutup Chrome dulu kalau pakai `medium`.

Kalau tetap mau `large-v3`, edit `shorts_generator/local/transcriber.py` baris `compute_type` jadi:

```python
compute_type = "int8_float16" if device == "cuda" else "int8"
```

Itu memangkas VRAM `large-v3` ke ~2.5 GB dengan penurunan akurasi yang minim.

### 5B.2 Prasyarat

1. **Python 3.10 atau 3.11** — dari [python.org](https://www.python.org/downloads/windows/), centang **"Add Python to PATH"** saat install. Jangan pakai Python dari Microsoft Store (sering bermasalah dengan path & venv).

2. **Driver NVIDIA terbaru.** Cek dengan:

```
nvidia-smi
```

Kolom "CUDA Version" di pojok kanan atas harus **12.x**. Kalau masih 11.x, update driver dari GeForce Experience atau nvidia.com. Kamu **tidak perlu** install CUDA Toolkit terpisah — library-nya akan datang lewat pip.

3. **ffmpeg** — cara termudah:

```
winget install Gyan.FFmpeg
```

Tutup dan buka ulang terminal, lalu verifikasi:

```
ffmpeg -version
```

4. **Git** — `winget install Git.Git`

### 5B.3 Clone & virtual environment

Jalankan di PowerShell:

```
git clone https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator.git
```

```
cd AI-Youtube-Shorts-Generator; py -3.11 -m venv venv; .\venv\Scripts\Activate.ps1
```

> Kalau PowerShell menolak menjalankan script aktivasi, jalankan sekali:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 5B.4 Install dependency

```
pip install -r requirements-local.txt
```

### 5B.5 Install runtime CUDA (langkah kunci — jangan dilewat)

`faster-whisper` berjalan di atas CTranslate2, yang butuh **cuBLAS** dan **cuDNN 9**. Dua library ini tidak ikut di `requirements-local.txt`. Tanpa ini, kamu akan kena error `Library cublas64_12.dll is not found` atau proses langsung crash.

**Cara paling bersih** (hanya ~200 MB, tidak menarik PyTorch):

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

**Alternatif** kalau kamu memang butuh PyTorch untuk hal lain (~2.5 GB):

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 5B.6 File `.env`

Buat file `.env` di root project:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=isi_key_kamu_disini
GEMINI_MODEL=gemini-2.5-flash
LOCAL_WHISPER_MODEL=small
LOCAL_WHISPER_DEVICE=cuda
LOCAL_OUTPUT_DIR=output
```

**Kenapa `LOCAL_WHISPER_DEVICE=cuda` diset eksplisit, bukan `auto`:**

Fungsi `_resolve_device()` mendeteksi CUDA lewat `import torch`. Kalau kamu pilih jalur cuBLAS/cuDNN saja (tanpa PyTorch), `import torch` akan gagal → tool diam-diam fallback ke CPU dan kamu tidak sadar GPU-nya nganggur. Set `cuda` secara eksplisit membuat pengecekan torch dilewati sepenuhnya.

Verifikasi saat run — baris ini harus muncul di output:

```
[transcribe/local] faster-whisper model=small device=cuda
```

Kalau tertulis `device=cpu`, berarti `.env` belum kebaca atau nilainya masih `auto`.

### 5B.7 Test run

```
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --num-clips 3
```

Hasil mendarat di `output\short_01.mp4`, `short_02.mp4`, dst.

### 5B.8 Batch di Windows

Windows tidak punya `xargs`. Pakai PowerShell:

```
Get-Content urls.txt | ForEach-Object { python main.py $_ --mode local }
```

### 5B.9 Percepat tahap render dengan NVENC (opsional)

Tahap potong & crop pakai `ffmpeg` dengan encoder default `libx264` — itu murni CPU dan tidak menyentuh GPU sama sekali. RTX 3050 Ti punya encoder hardware NVENC yang menganggur.

Edit `shorts_generator/local/clipper.py`, cari dua blok `subprocess.run` yang memanggil ffmpeg, lalu tambahkan `"-c:v", "h264_nvenc", "-preset", "p4"` ke daftar argumennya. Tahap render bisa 3–5× lebih cepat.

Ini opsional — untuk video pendek bedanya tidak terasa. Baru signifikan saat batch banyak klip.

### 5B.10 Troubleshooting Windows

| Gejala | Penyebab | Solusi |
|---|---|---|
| `Library cublas64_12.dll is not found` | cuBLAS belum terinstal | Jalankan langkah 5B.5 |
| `Unable to load libcudnn_ops.so` / `cudnn64_9.dll` | cuDNN salah versi | `pip install nvidia-cudnn-cu12==9.*` (harus versi 9, bukan 8) |
| `CUDA out of memory` | Model terlalu besar untuk 4 GB | Turunkan ke `small`, atau ubah `compute_type` ke `int8_float16` (lihat 5B.1) |
| Log tetap bilang `device=cpu` | `.env` tidak kebaca / masih `auto` tanpa torch | Set `LOCAL_WHISPER_DEVICE=cuda` eksplisit |
| `ffmpeg is not recognized` | Belum di PATH | Restart terminal setelah `winget install Gyan.FFmpeg` |
| `Activate.ps1 cannot be loaded` | ExecutionPolicy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Download YouTube gagal | yt-dlp usang | `pip install -U yt-dlp` |
| Path error di file lokal | Backslash Windows | Pakai forward slash: `"C:/Users/nama/Videos/input.mp4"` |

### 5B.11 Windows RTX 3050 Ti vs MacBook M-series

| | Windows + RTX 3050 Ti | MacBook M-series |
|---|---|---|
| Device Whisper | `cuda` float16 | `cpu` int8 |
| Kecepatan transkrip 60 mnt (`small`) | ~2–3 menit | ~8–15 menit |
| Model maksimum realistis | `medium` (batas VRAM 4 GB) | `large-v3` (batas RAM, bukan VRAM) |
| Kerumitan setup | Sedang (cuBLAS/cuDNN manual) | Mudah (tidak ada langkah GPU) |
| Encode ffmpeg | Bisa NVENC (perlu edit kode) | VideoToolbox otomatis via Homebrew |
| Titik gagal umum | DLL CUDA hilang, VRAM habis | Tidak ada — cuma lambat |

**Ringkasnya:** Windows menang telak di kecepatan transkrip, kalah di kemudahan setup. Mac tidak pernah error tapi harus sabar. Untuk workload batch berat, mesin Windows jelas lebih cocok — dengan catatan model Whisper dibatasi di `medium` karena VRAM cuma 4 GB.

---

## 6. Flag CLI

| Flag | Default | Fungsi |
|---|---|---|
| `--mode` | `api` | `api` (MuAPI berbayar) atau `local` (mesin sendiri) — **selalu set `local`** |
| `--num-clips` | `3` | Jumlah short yang dirender |
| `--aspect-ratio` | `9:16` | `9:16` TikTok/Reels/Shorts, `1:1` square, bebas |
| `--format` | `720` | Resolusi download: `360`/`480`/`720`/`1080` |
| `--language` | auto | Paksa bahasa Whisper, misal `id` untuk Bahasa Indonesia |
| `--output-json` | — | Dump hasil lengkap (transkrip + semua kandidat) ke file JSON |

---

## 7. Cara Kerja Detail

### 7.1 Download
Mode local pakai `yt-dlp` dengan format selector sesuai `--format`, merge ke mp4. Hasil di-cache sebagai `output/source_<youtube_id>.mp4` — run kedua untuk video sama akan skip download.

### 7.2 Transkrip
`faster-whisper` (CTranslate2, int8 di CPU) menghasilkan segmen bertimestamp. Hasil di-cache sebagai `.srt` di `LOCAL_OUTPUT_DIR` pakai nama file sumber. Kalau cache lebih baru dari file sumber → reuse, tidak transkrip ulang. Ini penghematan waktu terbesar saat iterasi.

### 7.3 Deteksi tipe konten
LLM mengklasifikasi video (podcast / interview / tutorial / vlog / dst) dan densitas informasinya, lalu prompt highlight disesuaikan per gaya konten.

### 7.4 Ranking highlight
Ini inti nilai jual repo — file `shorts_generator/highlights.py` (~12 KB prompt engineering). LLM menilai kandidat berdasarkan:

- Hook (pembuka yang menahan scroll)
- Puncak emosi
- Opini kontroversial ("opinion bomb")
- Momen revelation / plot twist
- Konflik / perdebatan
- Kalimat quotable
- Puncak cerita
- Nilai praktis (actionable)

Tiap kandidat dapat skor 0–100 + hook sentence + alasan.

**Framework ini bisa kamu edit sendiri.** Ini keunggulan nyata dibanding OpusClip yang black-box — kalau kamu punya teori sendiri soal apa yang viral di niche kamu, tinggal ubah prompt-nya.

### 7.5 Chunking video panjang
Video >30 menit dipotong jadi chunk dengan overlap supaya tidak ada bagian yang terlewat dari context window LLM.

### 7.6 Dedupe
Highlight yang timestamp-nya overlap digabung, ambil yang skornya tertinggi. Mencegah dua klip nyaris kembar.

### 7.7 Potong & crop
- `ffmpeg -ss start -to end` → re-encode subclip, audio ikut.
- OpenCV `haarcascade_frontalface_default.xml` melacak wajah frame-per-frame, dengan motion smoothing supaya crop tidak goyang.
- Frame di-crop ke aspect ratio target, ditulis ulang, lalu audio di-mux balik pakai ffmpeg.

---

## 8. Analisis Biaya

| Mode | Yang dibayar | Estimasi per video 60 menit |
|---|---|---|
| `local` + Gemini Flash | Hanya token LLM | **~$0.01–0.05** |
| `local` + gpt-4o-mini | Hanya token LLM | **~$0.03–0.15** |
| `api` (MuAPI) | Download + Whisper + LLM + autocrop | Per-request, cek pricing MuAPI |
| OpusClip / Klap | Langganan | $20–300/bulan, ada cap menit |

Mode `local` benar-benar murah karena bagian mahal (transkrip + render) jalan di mesin kamu. Yang dikirim ke LLM cuma teks transkrip.

---

## 9. Batasan yang Harus Kamu Tahu

1. **Tidak ada subtitle burn-in.** Ini yang paling penting. Kode clipper hanya potong + crop — tidak ada rendering caption/karaoke text. Padahal caption adalah alasan utama orang bayar OpusClip/SubMagic. **Solusi lengkap ada di bagian 9B** — skrip `burn_captions.py` menambahkan caption karaoke setara Submagic.
2. **Face tracking pakai Haar cascade**, teknologi tahun 2001. Cukup untuk talking-head podcast satu-dua orang. Gagal di: wajah miring, pencahayaan buruk, banyak orang, konten tanpa wajah (gameplay, screencast, b-roll).
3. **Tidak ada B-roll, zoom dinamis, atau efek transisi.** Output-nya polos.
4. **Kualitas ranking = kualitas LLM.** Pakai `gpt-4o-mini`/`gemini-flash` untuk hemat, tapi model kecil kadang salah pilih momen. Untuk konten penting, coba upgrade `OPENAI_MODEL` ke model yang lebih kuat.
5. **yt-dlp rentan patah.** YouTube sering ubah proteksi. Kalau download gagal, `pip install -U yt-dlp` dulu sebelum lapor bug.
6. **Bahasa Indonesia**: Whisper `base` akurasinya menengah untuk Bahasa Indonesia. Pakai `LOCAL_WHISPER_MODEL=small` atau `medium` + `--language id` kalau kontennya Indonesia. Konsekuensinya lebih lambat di CPU M-series.
7. **Status lisensi belum jelas** (lihat bagian 2). Untuk penggunaan komersial, klarifikasi dulu.

---

## 9B. Burn-In Caption (Setara Social Clip Studio / Submagic / OpusClip)

Ini menutup kekurangan nomor 1 di bagian sebelumnya. Skrip siap pakai: [`burn_captions.py`](burn_captions.py) — taruh di root project generator.

### 9B.1 Kenapa tidak bisa pakai `.srt` bawaan repo

Tergoda pakai cache `.srt` yang sudah ada? Tidak bisa, karena tiga alasan:

1. **SRT hanya punya timestamp per kalimat, bukan per kata.** Efek karaoke (kata aktif menyala satu per satu) mustahil dibuat dari SRT. Itu justru ciri khas Submagic/Social Clip Studio.
2. **`transcribe_local()` tidak pernah minta word timestamps.** Cek `shorts_generator/local/transcriber.py` — parameter yang dikirim cuma `beam_size`, `condition_on_previous_text`, `vad_filter`. Tidak ada `word_timestamps=True`, jadi data per-kata tidak pernah dihitung.
3. **Timestamp SRT relatif ke video sumber, bukan ke klip.** `short_01.mp4` dipotong dari detik 1.240 misalnya. Kalau SRT sumber langsung ditempel ke klip, semua caption meleset sejauh 1.240 detik.

Jadi butuh pass transkrip terpisah dengan `word_timestamps=True`, lalu digeser per klip. Itu persis yang skrip ini lakukan.

### 9B.2 Alur kerja skrip

```
video sumber → faster-whisper (word_timestamps=True) → cache .words.json
   → potong kata sesuai rentang tiap klip → geser waktu (kurangi start_time klip)
   → kelompokkan jadi baris pendek (3 kata / 20 karakter)
   → tulis .ass karaoke (1 event per kata)
   → ffmpeg: scale 1080x1920 + filter ass → mp4 final
```

Transkrip kata dijalankan **sekali** per file sumber lalu di-cache ke `output/<nama>.words.json`. Klip ke-2 dan ke-3 tinggal pakai cache.

### 9B.3 Cara pakai

Salin `burn_captions.py` ke root project (sejajar dengan `main.py`), lalu:

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --num-clips 3 --output-json result.json
```

```bash
python burn_captions.py result.json --style hormozi --language id
```

Hasil: `output/short_01_cap.mp4`, `short_02_cap.mp4`, dst. File asli tanpa caption tetap ada.

Burn satu file saja:

```bash
python burn_captions.py --video output/short_01.mp4 --style clean
```

Windows + RTX 3050 Ti (GPU untuk transkrip **dan** encode):

```
python burn_captions.py result.json --style hormozi --device cuda --model small --nvenc
```

### 9B.4 Preset gaya

| Preset | Karakter | Cocok untuk |
|---|---|---|
| `hormozi` | Huruf kapital semua, Arial Black 92px, kata aktif kuning-oranye, outline hitam tebal, 3 kata per baris | Konten hype, motivasi, bisnis — gaya paling umum di TikTok |
| `clean` | Huruf normal, Arial 74px, kata aktif putih terang, 5 kata per baris | Edukasi, tutorial, konten yang perlu terbaca santai |
| `neon` | Kapital, hijau neon, shadow tebal, 3 kata per baris | Gaming, hype, reaction |

Override per-item tanpa mengedit skrip:

```bash
python burn_captions.py result.json --style clean --font "Montserrat ExtraBold" --size 84 --margin-v 340
```

| Flag | Fungsi |
|---|---|
| `--style` | `hormozi` / `clean` / `neon` |
| `--font`, `--size` | Ganti font dan ukuran (ukuran relatif kanvas 1080×1920) |
| `--margin-v` | Jarak caption dari bawah, px. Naikkan kalau ketutup UI TikTok |
| `--no-karaoke` | Baris utuh tampil sekaligus, tanpa highlight per kata |
| `--no-uppercase` | Matikan kapital paksa |
| `--scale` | Resolusi output, default `1080x1920` |
| `--model`, `--device`, `--language` | Setting Whisper (`--language id` untuk Bahasa Indonesia) |
| `--nvenc` | Encode pakai GPU NVIDIA |
| `--keep-ass` | Simpan file `.ass` untuk diedit manual di Aegisub |
| `--refresh` | Abaikan cache kata, transkrip ulang |

### 9B.5 Cara efek karaoke dibuat

Satu baris caption dipecah jadi beberapa event ASS — **satu event per kata**. Tiap event menampilkan seluruh baris, tapi kata yang sedang diucapkan diberi warna berbeda dan diperbesar 112%:

```
Dialogue: 0,0:00:00.00,0:00:00.40,Cap,,0,0,0,,{\c&H00E5FF&\fscx112\fscy112}INI{\c&HFFFFFF&\fscx100\fscy100} TEST CAPTION
Dialogue: 0,0:00:00.40,0:00:00.90,Cap,,0,0,0,,INI {\c&H00E5FF&\fscx112\fscy112}TEST{\c&HFFFFFF&\fscx100\fscy100} CAPTION
Dialogue: 0,0:00:00.90,0:00:01.50,Cap,,0,0,0,,INI TEST {\c&H00E5FF&\fscx112\fscy112}CAPTION{\c&HFFFFFF&\fscx100\fscy100}
```

Teknik ini lebih bisa dikontrol daripada tag karaoke `\k` bawaan ASS, karena warna, ukuran, dan efek pop bisa diatur bebas per kata.

> **Catatan warna ASS**: formatnya `&HAABBGGRR` — **BGR terbalik, bukan RGB**. Kuning (R=255,G=229,B=0) ditulis `&H0000E5FF`. Salah urutan = warna tertukar biru/merah. Ini jebakan paling sering saat bikin style sendiri.

Pemenggalan baris memutus di: batas 3 kata, batas 20 karakter, durasi >1,6 detik, jeda bicara >0,55 detik, atau setelah tanda baca. Hasilnya baris pendek yang ritmis, bukan blok teks panjang.

### 9B.6 Catatan font

Nama font harus **terinstal di sistem** — libass mencarinya lewat fontconfig, bukan lewat file.

| OS | Font aman bawaan | Cara pasang font kustom |
|---|---|---|
| macOS | `Arial Black`, `Impact`, `Helvetica` | Dobel-klik file `.ttf` → Font Book |
| Windows | `Arial Black`, `Impact`, `Segoe UI Black` | Klik kanan `.ttf` → Install for all users |

Font gratis yang cocok gaya TikTok: **Montserrat ExtraBold**, **Anton**, **Bebas Neue**, **Poppins Bold** (semua ada di Google Fonts). Install dulu, baru panggil namanya persis lewat `--font`.

Kalau font tidak ketemu, libass diam-diam pakai font pengganti — caption tetap muncul tapi tampilannya beda dari yang diharapkan. Itu gejala salah ketik nama font.

### 9B.7 Kualitas & performa

- Klip di-render ulang (`libx264 crf 19` atau NVENC `cq 20`) karena burn-in wajib re-encode. Audio disalin ulang ke AAC 160k.
- Video di-scale ke 1080×1920 dulu baru caption ditempel. Ini penting: output mentah repo resolusinya kecil (720p sumber → crop 405×720), kalau caption ditempel di situ lalu diperbesar, teksnya jadi buram.
- Padding hitam ditambahkan kalau rasio klip tidak persis 9:16, supaya tidak ada distorsi.
- Waktu proses per klip 30 detik: **~10–20 detik** di Mac CPU, **~3–5 detik** dengan `--nvenc` di RTX 3050 Ti.

### 9B.8 Batas yang tetap ada

Skrip ini menyamakan caption dengan Submagic, tapi tidak menyamakan segalanya:

1. **Tidak ada emoji otomatis.** Submagic menyisipkan emoji per kata kunci lewat LLM. Bisa ditambah: kirim transkrip ke LLM minta mapping kata→emoji, lalu sisipkan ke teks ASS. Perlu font emoji berwarna dan hasilnya tidak selalu konsisten di libass.
2. **Tidak ada highlight kata kunci semantik.** Sekarang yang diwarnai adalah kata yang sedang diucapkan. Submagic mewarnai kata *penting*. Bisa ditambah dengan minta LLM menandai kata kunci, lalu beri warna berbeda.
3. **Tidak ada animasi masuk per kata** (bounce/slide). Bisa pakai tag `\t()` di ASS untuk transisi, tapi tuning-nya makan waktu.
4. **Akurasi timestamp bergantung model Whisper.** Model `base` sering meleset 100–200 ms per kata — cukup terasa di gaya karaoke. Untuk hasil rapi pakai `small` minimal, idealnya `medium`. Alternatif lebih presisi: `pip install stable-ts` yang memperbaiki alignment kata.
5. **Bahasa Indonesia**: selalu set `--language id`. Tanpa itu Whisper kadang salah deteksi bahasa dan word timestamp jadi kacau.

### 9B.9 Jalur cepat tanpa skrip

Kalau cuma butuh caption biasa (per kalimat, tanpa karaoke) dan klip dipotong dari awal video, `.srt` bawaan bisa langsung dipakai — tapi ingat masalah offset di 9B.1, jadi ini hanya valid untuk `--video` yang memang video utuh:

```bash
ffmpeg -i input.mp4 -vf "subtitles=output/source_VIDEOID.srt:force_style='FontName=Arial Black,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=3,Alignment=2,MarginV=60'" -c:a copy out.mp4
```

Untuk hasil setara Social Clip Studio, pakai `burn_captions.py`.

---

## 10. Rekomendasi Akhir

**Worth it kalau:**
- Kamu punya banyak konten long-form (podcast, interview, webinar) dan mau otomatisasi tahap pertama pemilihan klip.
- Kamu nyaman dengan CLI dan tidak masalah menambah caption di tahap terpisah.
- Kamu mau kontrol penuh + biaya mendekati nol per klip.
- Kamu mau pipeline yang bisa di-embed ke workflow Python sendiri.

**Tidak worth it kalau:**
- Kamu butuh hasil siap-posting langsung dengan caption bergaya. Tool ini tidak sampai situ.
- Konten kamu bukan talking-head (face tracking akan mengecewakan).
- Kamu tidak mau urusan dependency dan API key.

**Cara pakai paling optimal:** jadikan ini **mesin pemilih highlight**, bukan editor final.
Jalankan `--mode local --output-json result.json`, ambil timestamp + skor + hook dari JSON, lalu edit final di CapCut/Premiere. Bagian yang paling melelahkan — nonton 2 jam video cari momen bagus — itu yang tergantikan, dan itu memang bagian yang paling berharga.

---

## Lampiran — Perintah Cepat

Setup lengkap dari nol (copy-paste satu blok):

```bash
brew install ffmpeg python@3.11 && git clone https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator.git && cd AI-Youtube-Shorts-Generator && python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements-local.txt
```

Tambah caption sendiri dari cache `.srt` yang dihasilkan tool:

```bash
ffmpeg -i output/short_01.mp4 -vf "subtitles=output/source_VIDEOID.srt:force_style='FontSize=18,Alignment=2,MarginV=60'" output/short_01_captioned.mp4
```
