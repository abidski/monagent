from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_groq import ChatGroq

from agent.tools import analyze_medical_image, web_search


llm = ChatGroq(model="openai/gpt-oss-120b")

scanning_agent = create_agent(
    model=ChatGroq(model="openai/gpt-oss-120b"),
    tools=[analyze_medical_image],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
    system_prompt="""You orchestrate medical imaging tools for research/
    educational purposes. Report technical output only. Never provide a
    diagnosis. Always note that findings require professional review.""",
)
