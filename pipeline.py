from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def _extract_message_text(result):
    if isinstance(result, dict):
        messages = result.get("messages") or result.get("message")
        if messages:
            last = messages[-1]
            return getattr(last, "content", str(last))
        return str(result)
    return getattr(result, "content", str(result))


def run_research_pipeline(topic: str) -> dict:
    state = {}

    print("\n=" * 50)
    print("step 1 - search agent is working...")

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    state["search_results"] = _extract_message_text(search_result)
    print("\n search result ", state['search_results'])

    print("\n=" * 50)
    print("step 2 - scrape agent is working...")
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about {topic}, "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
            )]
    })

    state['scraped_content'] = _extract_message_text(reader_result)
    print("\n scarped content\n", state['scraped_content'])

    print("\n=" * 50)
    print("step 3 - Writer agent is working...")
    print("\n=" * 50)
    researh_combined = (
        f"SEARCH_RESULT:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": researh_combined
    })
    print("\n Final Report\n", state['report'])

    state['feedback'] = critic_chain.invoke({
        "topic": topic,
        "research": state['report']
    })
    print("\n critic report\n", state['feedback'])
    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic :")
    run_research_pipeline(topic)

if __name__=="__main__":
    topic=input("\n Enter a research topic :")
    run_research_pipeline(topic)