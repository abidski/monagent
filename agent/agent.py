import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools import analyze_medical_image
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware


agent = create_agent(
    model=ChatGroq(model="llama-3.1-8b-instant"),
    tools=[analyze_medical_image],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)


if __name__ == "__main__":
    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
    if not os.environ["GROQ_API_KEY"]:
        os.environ["GROQ_API_KEY"] = input("Enter your Groq API key: ")
