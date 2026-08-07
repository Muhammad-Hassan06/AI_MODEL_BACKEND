from genai.videomind.audio_processor import process_input
from genai.videomind.transcriber import transcribe_all
from genai.videomind.summarize import summarize
from genai.videomind.extractor import extract_action_items
from genai.videomind.vector_store import build_vector_store, ask_question

def run_pipeline(source_url_or_file, language="auto") -> dict:
    """
    Executes full AI Video Intelligence pipeline:
    1. Downloads & pre-processes audio (Mono 16kHz WAV chunking).
    2. Transcribes using Groq Whisper API / OpenAI Whisper.
    3. Generates executive summary & extracts action items using Groq Llama-3.3 70B.
    4. Indexes transcript into Chroma / Vector DB store for interactive RAG Q&A.
    """
    print("Step 1: Processing audio / YouTube source...")
    chunks = process_input(source_url_or_file)
    
    print("Step 2: Transcribing audio chunks...")
    transcript = transcribe_all(chunks, language=language)
    
    print("Step 3: Generating executive summary & extracting action items via Groq...")
    summary = summarize(transcript)
    actions = extract_action_items(transcript)
    
    print("Step 4: Building RAG Vector Store...")
    vector_store = build_vector_store(transcript)
    
    return {
        "transcript": transcript,
        "summary": summary,
        "actions": actions,
        "vector_store": vector_store
    }

if __name__ == "__main__":
    url = input("Enter YouTube URL or Audio File Path: ")
    res = run_pipeline(url)
    print("\n" + "="*50)
    print("SUMMARY:\n", res["summary"])
    print("\n" + "="*50)
    print("ACTION ITEMS:\n", res["actions"])
