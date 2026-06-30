import os
import operator
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ── Shared State ──
class TeamState(TypedDict):
    task: str
    research_notes: str
    draft: str
    review_feedback: str
    final_output: str
    iteration: int
    next_agent: str

# ── Agent 1: Researcher ──
def researcher_agent(state: TeamState) -> TeamState:
    print(f"\n🔍 RESEARCHER working...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a research specialist. 
Your job is to gather key facts and points about a topic.
Provide 4-5 concise bullet points of factual information.
Do not write prose, just research notes."""),
        HumanMessage(content=f"Research this topic: {state['task']}")
    ])
    
    print(f"📝 Research notes:\n{response.content}\n")
    
    return {
        "research_notes": response.content,
        "next_agent": "writer"
    }

# ── Agent 2: Writer ──
def writer_agent(state: TeamState) -> TeamState:
    print(f"\n✍️ WRITER working...")
    
    feedback_context = ""
    if state.get("review_feedback"):
        feedback_context = f"\n\nPrevious feedback to address:\n{state['review_feedback']}"
    
    response = llm.invoke([
        SystemMessage(content="""You are a content writer.
Write a clear, engaging paragraph (4-6 sentences) using the research notes provided.
Make it flow naturally, not just a list of facts."""),
        HumanMessage(content=f"""Topic: {state['task']}

Research notes:
{state['research_notes']}{feedback_context}

Write the content:""")
    ])
    
    print(f"📄 Draft:\n{response.content}\n")
    
    return {
        "draft": response.content,
        "next_agent": "reviewer"
    }

# ── Agent 3: Reviewer ──
def reviewer_agent(state: TeamState) -> TeamState:
    print(f"\n🔎 REVIEWER working...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a STRICT quality reviewer.
Reject drafts that don't include specific numbers, statistics, or concrete examples.
If it's good, respond with exactly: "APPROVED"
If it needs work, respond with exactly: "REVISE: " followed by specific feedback."""),
        HumanMessage(content=f"""Topic: {state['task']}
Draft: {state['draft']}

Your review:""")
    ])
    
    feedback = response.content.strip()
    print(f"📋 Review: {feedback}\n")
    
    iteration = state.get("iteration", 0) + 1
    
    if "APPROVED" in feedback or iteration >= 3:
        return {
            "final_output": state["draft"],
            "next_agent": "done",
            "iteration": iteration
        }
    else:
        return {
            "review_feedback": feedback,
            "next_agent": "writer",
            "iteration": iteration
        }

# ── Routing ──
def route_next(state: TeamState) -> str:
    next_agent = state.get("next_agent", "researcher")
    print(f"🚦 Routing to: {next_agent}")
    
    if next_agent == "done":
        return END
    return next_agent

# ── Build Graph ──
graph = StateGraph(TeamState)

graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)

graph.set_entry_point("researcher")

graph.add_conditional_edges(
    "researcher",
    route_next,
    {"writer": "writer"}
)
graph.add_conditional_edges(
    "writer",
    route_next,
    {"reviewer": "reviewer"}
)
graph.add_conditional_edges(
    "reviewer",
    route_next,
    {"writer": "writer", END: END}
)

app = graph.compile()

# ── Run it ──
def run_team(task: str):
    print(f"\n{'='*60}")
    print(f"🎯 TEAM TASK: {task}")
    print(f"{'='*60}")
    
    result = app.invoke({
        "task": task,
        "research_notes": "",
        "draft": "",
        "review_feedback": "",
        "final_output": "",
        "iteration": 0,
        "next_agent": "researcher"
    })
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL OUTPUT (after {result['iteration']} review cycle(s)):")
    print(f"{'='*60}")
    print(result["final_output"])
    
    return result

# ── Tests ──
run_team("Benefits of using FastAPI over Django for AI backends")
run_team("Why RAG systems reduce LLM hallucinations")