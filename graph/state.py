from typing import TypedDict, List, Optional, Literal, Dict, Any

class AgentState(TypedDict, total=False):
    question: str
    route: Literal["qa","repo_eval","search"]
    context: str
    refs: List[Dict[str, Any]]
    answer: str
    repo: Optional[str]
    search_query: Optional[str]
