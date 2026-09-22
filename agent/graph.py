from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from models.classifier_model import predict_image
from agent.research_agent import research_agent
from agent.interpretation_agent import interpretation_agent
from agent.report_agent import report_agent

DISCLAIMER = (
    "\n\nThis is a research tool output, not a medical diagnosis. "
    "Findings require review by a qualified professional."
)


class AgentState(TypedDict, total=False):
    image_path: str
    image_analysis: dict
    research: str
    interpretation: str
    report: str


def ask(agent, content: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    return result["messages"][-1].content


def scan_node(state: AgentState) -> dict:
    return {"image_analysis": predict_image(state["image_path"])}


def research_node(state: AgentState) -> dict:
    label = state["image_analysis"]["label"]
    return {
        "research": ask(
            research_agent,
            f"Find general educational information about {label} imaging: "
            f"what the modality is and what it is typically used for.",
        )
    }


def interpret_node(state: AgentState) -> dict:
    return {
        "interpretation": ask(
            interpretation_agent,
            f"Model output: {state['image_analysis']}\n\n"
            f"Research findings: {state['research']}",
        )
    }


def report_node(state: AgentState) -> dict:
    report = ask(
        report_agent,
        f"Model output: {state['image_analysis']}\n\n"
        f"Research findings: {state['research']}\n\n"
        f"Interpretation: {state['interpretation']}",
    )
    return {"report": report + DISCLAIMER}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("scan", scan_node)
    graph.add_node("research", research_node)
    graph.add_node("interpret", interpret_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "scan")
    graph.add_edge("scan", "research")
    graph.add_edge("research", "interpret")
    graph.add_edge("interpret", "report")
    graph.add_edge("report", END)
    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"image_path": input("Image path: ")})
    print(result["report"])
