import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_groq_llm():
    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing from environment variables.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=groq_api_key
    )

def extract_action_items(transcript: str) -> str:
    """
    Extracts actionable items, decisions, and key tasks from the transcript using Groq Llama-3.3 70B.
    """
    if not transcript or not transcript.strip():
        return "No transcript provided for extraction."

    try:
        llm = get_groq_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Task Extractor specializing in extracting actionable items, decisions, and key assignments from meeting transcripts and video audio."),
            ("human", """Analyze the transcript below and extract all actionable items, key decisions, and follow-up tasks.

Format requirement:
### 📌 Action Items & Tasks
- [ ] **Task 1**: [Description & Owner if mentioned]
- [ ] **Task 2**: [Description]

### 💡 Key Decisions Made
- **Decision 1**: [Explanation]

### 🏷️ Critical Topics & Mentions
- **Topic**: [Key insight]

Transcript:
{transcript}""")
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"transcript": transcript[:12000]})
    except Exception as err:
        return f"### 📌 Action Items & Tasks\n- [ ] Review full video transcript\n\n*(Notice: {str(err)})*"
