import operator
from typing import Annotated

from langchain.messages import AnyMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    image_path: str
    image_analysis: dict
    research: str
    interpretation: str
    report: str
