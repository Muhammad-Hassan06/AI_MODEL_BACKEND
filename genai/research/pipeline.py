import json
from genai.research.agents import build_search_agent, build_reader_agent, get_writer_chain, get_critic_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}
    
    # Step 1: Search Agent
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable, and detailed information about the topic: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    
    # Step 2: Reader Agent
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user", f"Based on the following research results about '{topic}', pick the most relevant URLs and scrape them for deeper content.\n\n{state['search_results'][-800:]}")]
    })
    state["scraped_content"] = reader_result["messages"][-1].content
    
    # Step 3: Writer Chain
    research_combined = f"Search Results:\n{state['search_results']}\n\nDetailed Scraped Content:\n{state['scraped_content']}"
    writer_chain = get_writer_chain()
    report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    state["report"] = report
    
    # Step 4: Critic Chain
    critic_chain = get_critic_chain()
    feedback = critic_chain.invoke({
        "report": state["report"]
    })
    state["feedback"] = feedback
    
    return state

def run_research_pipeline_stream(topic: str):
    state = {}
    
    # Step 1: Search Agent
    yield json.dumps({"step": 1, "status": "searching", "message": "Search Agent is querying web sources via Tavily..."})
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable, and detailed information about the topic: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    yield json.dumps({"step": 1, "status": "search_complete", "message": "Search completed.", "data": state["search_results"]})
    
    # Step 2: Reader Agent
    yield json.dumps({"step": 2, "status": "scraping", "message": "Reader Agent is parsing webpage content..."})
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user", f"Based on the following research results about '{topic}', pick the most relevant URLs and scrape them for deeper content.\n\n{state['search_results'][-800:]}")]
    })
    state["scraped_content"] = reader_result["messages"][-1].content
    yield json.dumps({"step": 2, "status": "reader_complete", "message": "Scraping completed.", "data": state["scraped_content"]})
    
    # Step 3: Writer Chain
    yield json.dumps({"step": 3, "status": "writing", "message": "Writer Agent is synthesizing research into a structured report..."})
    research_combined = f"Search Results:\n{state['search_results']}\n\nDetailed Scraped Content:\n{state['scraped_content']}"
    writer_chain = get_writer_chain()
    report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    state["report"] = report
    yield json.dumps({"step": 3, "status": "writer_complete", "message": "Report writing completed.", "data": state["report"]})
    
    # Step 4: Critic Chain
    yield json.dumps({"step": 4, "status": "critiquing", "message": "Critic Agent is reviewing the report..."})
    critic_chain = get_critic_chain()
    feedback = critic_chain.invoke({
        "report": state["report"]
    })
    state["feedback"] = feedback
    yield json.dumps({"step": 4, "status": "finished", "message": "Research pipeline finished!", "data": state["feedback"], "full_state": state})
