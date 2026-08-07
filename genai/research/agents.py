import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from genai.research.tools import web_search, scrape_url

load_dotenv()

def get_groq_llm():
    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)

# 1. Search Agent
def build_search_agent():
    return create_react_agent(model=get_groq_llm(), tools=[web_search])

# 2. Reader Agent
def build_reader_agent():
    return create_react_agent(model=get_groq_llm(), tools=[scrape_url])

# 3. Writer Chain
def get_writer_chain():
    llm = get_groq_llm()
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research writer. Write clear, structured, and insightful reports."),
        ("human", """Write a detailed research report on the topic below.
    
Topic: {topic}

Research:
{research}

Structure:
- Introduction
- Key Findings (Minimum 3 well-explained points)
- Conclusion
- Sources

Be detailed, factual, and professional.""")
    ])
    return writer_prompt | llm | StrOutputParser()

# 4. Critic Chain
def get_critic_chain():
    llm = get_groq_llm()
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sharp and constructive research critic. Be honest and specific."),
        ("human", """Review the research report below and evaluate it strictly.
    
Report:
{report}

Format:
- Score (out of 10)
- Strengths
- Areas of Improvement
- One-line Verdict""")
    ])
    return critic_prompt | llm | StrOutputParser()
