import os
from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        Chroma = None

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

class SimpleVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def as_retriever(self, search_kwargs=None):
        class SimpleRetriever:
            def __init__(self, docs):
                self.docs = docs
            def invoke(self, question):
                q_words = set(question.lower().split())
                def score(doc):
                    words = set(doc.page_content.lower().split())
                    return len(q_words.intersection(words))
                sorted_docs = sorted(self.docs, key=score, reverse=True)
                return sorted_docs[:4]
        return SimpleRetriever(self.docs)

def build_vector_store(transcript: str, collection_name: str = "video_transcript"):
    """
    Splits transcript into chunks and builds Chroma or Simple vector store for RAG Q&A.
    """
    if not transcript or not transcript.strip():
        transcript = "Empty transcript provided."

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]
    
    if Chroma is not None:
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
            embeddings = FastEmbedEmbeddings()
            return Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=collection_name)
        except Exception:
            pass

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=collection_name)
        except Exception:
            pass

    return SimpleVectorStore(docs)

def ask_question(vector_store, question: str) -> str:
    """
    Answers questions grounded in the video transcript context using RAG and Groq Llama-3.3 70B.
    """
    if not vector_store or not question or not question.strip():
        return "Please process a video first and provide a valid question."

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    llm = get_groq_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent Video Q&A Assistant. Answer the user's question accurately based ONLY on the video transcript context below. If the answer is not mentioned in the context, state that clearly."),
        ("human", """Context from Video Transcript:
{context}

Question: {question}

Answer with detailed explanation:""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
