from typing import TypedDict, List, Optional, Literal, Dict, Any

class AgentState(TypedDict, total=False):
    question: str
    route: Literal["qa","repo_eval","search","agent","reset"]
    context: str
    refs: List[Dict[str, Any]]
    answer: str
    repo: Optional[str]
    search_query: Optional[str]
    memory: List[Dict[str, str]]  # e.g. [{"q":"...", "a":"...", "refs":"..."}]
    verified: bool  # set true when reflection/verification passed
    # Agent
    plan: Optional[str]           # natural language or structured plan
    actions: Optional[List[Dict[str, Any]]]  # parsed actions/tools to call
    tool_output: Optional[Any]    # result from action_executor
    
    # Loop control
    reflect_retries: Dict[str, int]       # e.g. {"reflect": 0, "planner": 0}
    plan_retries: Dict[str, int]  
