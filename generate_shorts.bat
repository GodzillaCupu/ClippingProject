@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title AI YouTube Shorts Generator Auto-Runner

echo ========================================================================
echo        AI YOUTUBE SHORTS GENERATOR - ONE-CLICK AUTO-RUNNER
echo ========================================================================
echo.

REM 1. Tentukan direktori kerja
if exist "%~dp0AI-Youtube-Shorts-Generator\main.py" (
    cd /d "%~dp0AI-Youtube-Shorts-Generator"
) else if exist "%~dp0main.py" (
    cd /d "%~dp0"
) else (
    echo [ERROR] Tidak dapat menemukan folder AI-Youtube-Shorts-Generator atau main.py!
    echo Pastikan file bat ini berada di folder project.
    echo.
    pause
    exit /b 1
)

REM 2. Cek apakah virtual environment tersedia
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment Python tidak ditemukan di folder venv!
    echo Silakan pastikan venv sudah dibuat di folder AI-Youtube-Shorts-Generator.
    echo.
    pause
    exit /b 1
)

REM 3. Cek file .env
if not exist ".env" (
    echo [ERROR] File .env tidak ditemukan!
    echo Buat file .env dan masukkan GEMINI_API_KEY Anda terlebih dahulu.
    echo.
    pause
    exit /b 1
)

REM 4. Input URL YouTube
set "YOUTUBE_URL=%~1"
if "%YOUTUBE_URL%"=="" (
    echo Masukkan Link/URL YouTube yang ingin diproses:
    set /p "YOUTUBE_URL=URL: "
)

if "%YOUTUBE_URL%"=="" (
    echo.
    echo [ERROR] URL YouTube tidak boleh kosong!
    echo.
    pause
    exit /b 1
)

REM 5. Opsi konfigurasi
echo.
set /p "NUM_CLIPS=Jumlah klip shorts yang ingin dibuat [default: 3]: "
if "%NUM_CLIPS%"=="" set "NUM_CLIPS=3"

echo.
echo Pilih gaya tampilan subtitle karaoke:
echo   [1] hormozi  (Kapital tebal, kata aktif kuning/oranye - Gaya Populer TikTok/Shorts)
echo   [2] clean    (Huruf rapi, putih bersih - Gaya Edukasi/Podcast Santai)
echo   [3] neon     (Kapital tebal, hijau neon menyala - Gaya Gaming/Hype)
set /p "STYLE_OPT=Pilihan [1/2/3, default: 1]: "

set "SUB_STYLE=hormozi"
if "%STYLE_OPT%"=="2" set "SUB_STYLE=clean"
if "%STYLE_OPT%"=="3" set "SUB_STYLE=neon"

echo.
echo ========================================================================
echo Target URL    : !YOUTUBE_URL!
echo Jumlah Klip   : !NUM_CLIPS!
echo Gaya Subtitle : !SUB_STYLE!
echo Device CUDA   : GPU NVIDIA RTX (NVENC)
echo ========================================================================
echo.

REM 6. Langkah 1: Ekstraksi Highlight dan Reframing 9:16
echo ------------------------------------------------------------------------
echo [LANGKAH 1/2] Mengunduh, Transkrip CUDA, LLM Highlight dan Reframing 9:16...
echo ------------------------------------------------------------------------
call .\venv\Scripts\python.exe main.py "!YOUTUBE_URL!" --mode local --num-clips !NUM_CLIPS! --output-json result.json
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Terjadi kesalahan saat menjalankan main.py!
    echo Silakan periksa koneksi internet, API key di .env, atau driver GPU Anda.
    echo.
    pause
    exit /b !ERRORLEVEL!
)

REM 7. Langkah 2: Membakar Subtitle Karaoke
echo.
echo ------------------------------------------------------------------------
echo [LANGKAH 2/2] Membakar Subtitle Karaoke Kata-per-Kata (Gaya: !SUB_STYLE!)...
echo ------------------------------------------------------------------------
call .\venv\Scripts\python.exe burn_captions.py result.json --style !SUB_STYLE! --device cuda --model small --nvenc
if !ERRORLEVEL! neq 0 (
    echo [INFO] NVENC gagal atau tidak tersedia, mencoba mode default CPU/CUDA...
    call .\venv\Scripts\python.exe burn_captions.py result.json --style !SUB_STYLE! --device cuda --model small
)

echo.
echo ========================================================================
echo [SELESAI] Seluruh video shorts dan subtitle berhasil diproses!
echo Video siap unggah tersimpan di folder:
echo   %CD%\output\*_cap.mp4
echo ========================================================================
echo.

if exist "output" (
    explorer output
)

pause