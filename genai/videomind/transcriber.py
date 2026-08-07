import os
from dotenv import load_dotenv

load_dotenv()

def transcribe_all(chunks, language="auto") -> str:
    """
    Transcribes audio chunks using Groq Whisper API (whisper-large-v3) with fallback to local Whisper.
    Fast, reliable, and supports multilingual audio (English, Urdu, Hindi, Spanish, etc.).
    """
    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY")
    
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            transcripts = []
            
            for idx, chunk in enumerate(chunks):
                print(f"Transcribing chunk {idx + 1}/{len(chunks)} via Groq Whisper API...")
                with open(chunk, "rb") as file_obj:
                    kwargs = {
                        "file": (os.path.basename(chunk), file_obj.read()),
                        "model": "whisper-large-v3"
                    }
                    if language and language.lower() not in ["auto", "default"]:
                        kwargs["language"] = language.lower()
                    
                    res = client.audio.transcriptions.create(**kwargs)
                    transcripts.append(res.text.strip())
            return " ".join(transcripts)
        except Exception as e:
            print(f"Groq Whisper API notice: {e}. Attempting local Whisper ASR...")

    # Fallback to local Whisper model if installed
    try:
        import whisper
        model_name = os.getenv("WHISPER_MODEL", "small")
        model = whisper.load_model(model_name)
        transcripts = []
        for chunk in chunks:
            options = {}
            if language and language.lower() not in ["auto", "default"]:
                options["language"] = language.lower()
            res = model.transcribe(chunk, **options)
            transcripts.append(res.get("text", "").strip())
        return " ".join(transcripts)
    except Exception as err:
        return f"Audio processed ({len(chunks)} chunks). Transcription engine notice: {str(err)}"
