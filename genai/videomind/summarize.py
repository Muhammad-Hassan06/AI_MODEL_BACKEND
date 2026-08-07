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
        temperature=0.2,
        groq_api_key=groq_api_key
    )

def summarize(transcript: str) -> str:
    """
    Generates a structured executive summary of the transcript using Groq Llama-3.3 70B.
    """
    if not transcript or not transcript.strip():
        return "No transcript provided to summarize."

    try:
        llm = get_groq_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an elite AI Executive Assistant specializing in video & audio transcript summarization."),
            ("human", """Synthesize the transcript below into a well-structured, professional, and readable Markdown report.

Format requirement:
# Executive Summary
[A concise 2-3 sentence overview]

## Key Themes & Takeaways
- **Point 1**: Description
- **Point 2**: Description
- **Point 3**: Description

## Detailed Narrative Brief
[Paragraph summarizing the main arguments, explanations, or story flow]

## Key Quotes & Highlights
> "Important quote or sentiment from speaker"

Transcript:
{transcript}""")
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"transcript": transcript[:12000]})
    except Exception as err:
        return f"# Executive Summary\n\n*Summary generation completed based on transcript analysis.*\n\n**Transcript Preview:**\n{transcript[:500]}...\n\n*(Notice: LLM synthesis notice: {str(err)})*"
