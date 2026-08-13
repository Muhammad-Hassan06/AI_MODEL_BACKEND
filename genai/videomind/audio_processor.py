import os
import tempfile
import subprocess
import wave
from yt_dlp import YoutubeDL
import imageio_ffmpeg

try:
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"

DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "videomind_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

CLIENT_STRATEGIES = [
    ['ios'],
    ['tv', 'mweb'],
    ['android_creator', 'android'],
    ['web_embedded', 'mweb']
]

def convert_to_mono_16k_wav(input_path, output_path):
    """
    Directly convert any audio/video file to 16kHz Mono WAV using imageio_ffmpeg executable.
    Bypasses pydub and ffprobe entirely.
    """
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        output_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err_msg = res.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"Audio conversion failed: {err_msg[:200]}")

def process_input(source):
    """
    Process YouTube URL or local audio/video file.
    Converts audio to Mono 16kHz WAV and enforces a 10-minute public portfolio limit.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://") or "youtube.com" in source or "youtu.be" in source):
        file_path = None

        for clients in CLIENT_STRATEGIES:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(DOWNLOAD_DIR, '%(id)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'extractor_args': {
                        'youtube': {
                            'player_client': clients
                        }
                    },
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                        'Accept-Language': 'en-US,en;q=0.9',
                    },
                    'nocheckcertificate': True,
                    'quiet': True,
                    'no_warnings': True
                }
                if FFMPEG_PATH:
                    ydl_opts['ffmpeg_location'] = FFMPEG_PATH

                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=True)
                    file_id = info.get('id', 'audio')
                    file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}.wav")
                    if os.path.exists(file_path):
                        break
            except Exception:
                continue

        if not file_path or not os.path.exists(file_path):
            raise RuntimeError(
                "YouTube restricted automated downloading for this specific video on cloud servers. "
                "Please upload the audio/video file directly using the 'Upload File' tab or try a different YouTube link."
            )
    else:
        file_path = source
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Convert to 16kHz Mono WAV directly with FFmpeg (no ffprobe needed!)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    converted_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_16k.wav")
    convert_to_mono_16k_wav(file_path, converted_path)

    # Calculate duration using standard library `wave`
    with wave.open(converted_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration_sec = frames / float(rate)

    # Portfolio Guardrail: Cap public audio processing at 10 mins (600 sec)
    max_allowed_sec = 600
    if duration_sec > max_allowed_sec:
        duration_mins = round(duration_sec / 60.0, 1)
        raise ValueError(
            f"Audio duration ({duration_mins} mins) exceeds the 10-minute public limit. "
            "Please use a video or audio clip under 10 minutes."
        )

    chunks = [converted_path]
    return chunks
