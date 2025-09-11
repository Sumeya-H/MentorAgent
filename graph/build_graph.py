import networkx as nx
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import router_node, retrieve_node, generate_node, reflect_node, reset_node, repo_eval_node, search_node, planner_node, action_executor_node

def compile_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("reflect", reflect_node)
    g.add_node("reset", reset_node)
    g.add_node("repo_eval", repo_eval_node)
    g.add_node("search", search_node)
    g.add_node("planner", planner_node)
    g.add_node("action_executor", action_executor_node)

    g.set_entry_point("router")

    # Edges based on router route
    g.add_conditional_edges(
        "router",
        lambda s: s.get("route","qa"),
        {
            "qa": "retrieve",
            "repo_eval": "repo_eval",
            "search": "search",
            "planner": "planner"
        }
    )

    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "reflect")   
    #g.add_edge("reflect", END)

    # Reflect conditional edges
    def reflect_route(state):
        if not state.get("verified", True):
            retries = state.get("reflect_retries", 0)
            if retries < 1:   # allow one retry
                return "retry"
        return "end"

    g.add_conditional_edges(
        "reflect",
        reflect_route,
        {
            "retry": "router",  # go back and retry
            "end": END
        }
    )
    g.add_edge("reset", END)
    g.add_edge("repo_eval", END)
    g.add_edge("search", END)
    g.add_edge("planner", "action_executor")
    #g.add_edge("action_executor", "reflect")

    def action_route(state):
        if not state.get("verified", True):  
            retries = state.get("plan_retries", 0)
            if retries < 1:   # allow one re-plan
                return "retry"
        return "end"

    g.add_conditional_edges(
        "action_executor",
        action_route,
        {
            "retry": "planner",  # re-plan if failed
            "end": "reflect"     # successful → reflect
        }
    )

    return g.compile()


# ---- generate workflow PNG ----
#graph = compile_graph()

# Convert to networkx DiGraph
#nx_graph = graph.get_graph()

# Draw using pygraphviz
#nx.drawing.nx_agraph.write_dot(nx_graph, "workflow.dot")

#import pygraphviz
#A = pygraphviz.AGraph("workflow.dot")
#A.layout("dot")
#A.draw("workflow.png")