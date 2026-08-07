import os
import pydub
from pydub import AudioSegment
from yt_dlp import YoutubeDL

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    pydub.AudioSegment.converter = FFMPEG_PATH
    pydub.AudioSegment.ffmpeg = FFMPEG_PATH
    pydub.AudioSegment.ffprobe = FFMPEG_PATH
except Exception:
    FFMPEG_PATH = None

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def process_input(source):
    """
    Process YouTube URL or local audio/video file.
    Converts audio to Mono 16kHz WAV and enforces a 10-minute public portfolio limit.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://") or "youtube.com" in source or "youtu.be" in source):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{DOWNLOAD_DIR}/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'ios', 'web']
                }
            },
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
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
    else:
        file_path = source
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Convert to Mono and 16kHz (Whisper Sweet Spot)
    audio = AudioSegment.from_file(file_path)

    # -----------------------------------------------------------
    # Portfolio Guardrail: Cap public audio processing at 10 mins
    # -----------------------------------------------------------
    max_allowed_ms = 10 * 60 * 1000  # 10 minutes in ms
    if len(audio) > max_allowed_ms:
        duration_mins = round(len(audio) / 60000.0, 1)
        raise ValueError(
            f"Audio duration ({duration_mins} mins) exceeds the 10-minute public limit. "
            "Please use a video or audio clip under 10 minutes."
        )

    audio = audio.set_channels(1).set_frame_rate(16000)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    converted_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_converted.wav")
    audio.export(converted_path, format="wav")
    
    # Chunk audio (10 minutes per chunk)
    chunk_ms = 10 * 60 * 1000
    chunks = []
    
    if len(audio) <= chunk_ms:
        chunks.append(converted_path)
    else:
        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start:start + chunk_ms]
            chunk_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        
    return chunks
