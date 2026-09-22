from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image

from models.classifier_model import predict_image

CONFIDENCE_THRESHOLD = 0.90
DISCLAIMER = (
    "\n\nThis is a research tool output, not a medical diagnosis. "
    "Findings require review by a qualified professional."
)


class PipelineState(TypedDict, total=False):
    image_path: str
    is_valid: bool
    validation_error: str
    prediction: dict
    needs_review: bool
    report: str


def validate_node(state: PipelineState) -> dict:
    path = Path(state["image_path"])
    if not path.exists():
        return {"is_valid": False, "validation_error": f"File not found: {path}"}
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return {
            "is_valid": False,
            "validation_error": f"Unsupported format: {path.suffix}",
        }
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        return {"is_valid": False, "validation_error": f"Unreadable image: {e}"}
    return {"is_valid": True}


def classify_node(state: PipelineState) -> dict:
    return {"prediction": predict_image(state["image_path"])}


def review_node(state: PipelineState) -> dict:
    confidence = state["prediction"]["confidence"]
    return {"needs_review": confidence < CONFIDENCE_THRESHOLD}


def build_report_node(llm):
    def report_node(state: PipelineState) -> dict:
        pred = state["prediction"]
        response = llm.invoke(
            [
                (
                    "system",
                    "You write short technical summaries of image classification "
                    "results. Never give a diagnosis or clinical interpretation.",
                ),
                ("human", f"Summarize this classification result: {pred}"),
            ]
        )
        return {"report": response.content + DISCLAIMER}

    return report_node


def flag_node(state: PipelineState) -> dict:
    pred = state["prediction"]
    return {
        "report": (
            f"LOW CONFIDENCE - flagged for review.\n"
            f"Predicted: {pred['label']} ({pred['confidence']:.2%})" + DISCLAIMER
        )
    }


def invalid_node(state: PipelineState) -> dict:
    return {"report": f"Image rejected before analysis: {state['validation_error']}"}


def build_graph(llm):
    graph = StateGraph(PipelineState)
    graph.add_node("validate", validate_node)
    graph.add_node("classify", classify_node)
    graph.add_node("review", review_node)
    graph.add_node("report", build_report_node(llm))
    graph.add_node("flag", flag_node)
    graph.add_node("invalid", invalid_node)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate",
        lambda s: "classify" if s["is_valid"] else "invalid",
        {"classify": "classify", "invalid": "invalid"},
    )
    graph.add_edge("classify", "review")
    graph.add_conditional_edges(
        "review",
        lambda s: "flag" if s["needs_review"] else "report",
        {"report": "report", "flag": "flag"},
    )
    graph.add_edge("report", END)
    graph.add_edge("flag", END)
    graph.add_edge("invalid", END)
    return graph.compile()
