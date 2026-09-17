import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from tools import analyze_medical_image
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware


# --- Agent ---
os.environ["GROQ_API_KEY"] = input("Enter Groq API key: ")
llm = ChatGroq(model="openai/gpt-oss-120b")

agent = create_agent(
    model=ChatGroq(model="llama-3.1-8b-instant"),
    tools=[analyze_medical_image],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
    system_prompt="""You orchestrate medical imaging tools for research/
    educational purposes. Report technical output only. Never provide a
    diagnosis. Always note that findings require professional review.""",
)

# --- Test ---
if __name__ == "__main__":
    image_path = input("Enter image path: ")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"What kind of scan is this: {image_path}",
                }
            ]
        }
    )
    print(f"\n{result['messages'][-1].content}\n")
