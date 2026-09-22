from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_groq import ChatGroq

from agent.tools import web_search

SYSTEM_PROMPT = """
    You are a medical imaging research assistant.

    Your job is to research medical imaging topics using the available
    web search tool.

    When answering a research question:

    1. Determine what information is needed.
    2. Use the web search tool to find relevant sources.
    3. Prefer reliable sources such as universities, hospitals,
       government organizations, and peer-reviewed research.
    4. If the first search does not provide enough information,
       perform another search with a more specific query.
    5. Summarize the information found in the sources.
    6. Do not invent information or citations.
    7. Do not provide a medical diagnosis.

    Return a concise research summary based on the retrieved sources.
    """


research_agent = create_agent(
    model=ChatGroq(model="openai/gpt-oss-120b"),
    tools=[web_search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
    system_prompt=SYSTEM_PROMPT,
)
