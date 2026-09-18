from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_groq import ChatGroq

from agent.tools import analyze_medical_image, web_search

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
    model="openai/gpt-oss-120b",
    tools=[web_search],
    system_prompt=SYSTEM_PROMPT,
)


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


INTERPRETATION_PROMPT = """
You are a medical imaging interpretation assistant.

Your task is to interpret the output of a medical image
classification model together with information retrieved
from reliable sources.

You will receive:
1. The classification produced by a MONAI model.
2. The model's confidence and class probabilities.
3. Research findings retrieved from the web.

Your responsibilities are to:
- Explain what the model's classification means.
- Relate the classification to the research findings.
- Identify important limitations or uncertainty.
- Clearly distinguish model output from information found in
  external sources.
- Do not invent information.
- Do not diagnose a patient.
- Do not claim that the classification represents a clinical diagnosis.

Provide a clear, concise interpretation.
"""


interpretation_agent = create_agent(
    model="openai/gpt-oss-20b",
    tools=[],
    system_prompt=INTERPRETATION_PROMPT,
)


REPORT_PROMPT = """
You are a medical imaging research report writer.

Your job is to create a clear research report using the
information provided by the other components of the system.

The information may include:
- The MONAI image classification result
- Model confidence and class probabilities
- Research findings retrieved from external sources
- An interpretation of the model's output

Your report should contain the following sections:

1. Image Classification
2. Model Results
3. Research Context
4. Interpretation
5. Limitations

Important rules:
- Do not invent information.
- Do not introduce information that was not provided.
- Clearly distinguish the model's classification from medical
  conclusions.
- Do not present the model output as a medical diagnosis.
- Mention relevant uncertainty and limitations.
- Write in a professional and objective style.
"""


report_agent = create_agent(
    model="qwen/qwen3.8-27b",
    tools=[],
    system_prompt=REPORT_PROMPT,
)
