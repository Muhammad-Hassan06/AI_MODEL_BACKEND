import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs, and snippets."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Error: TAVILY_API_KEY is not configured in environment variables."
    try:
        tavily = TavilyClient(api_key=tavily_key)
        results = tavily.search(query=query, max_results=5)
        out = []
        for r in results.get("results", []):
            out.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')[:300]}")
        return "\n----\n".join(out)
    except Exception as e:
        return f"Tavily search error: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unnecessary tags
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
            
        text = soup.get_text(separator=" ", strip=True)
        return text[:3000] # Return only first 3000 chars to save tokens
    except Exception as e:
        return f"Failed to scrape URL: {str(e)}"
