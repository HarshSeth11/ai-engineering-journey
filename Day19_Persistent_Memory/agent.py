import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import TypedDict, Annotated
import operator

load_dotenv()

# ── Step 1: Define tools using LangGraph's @tool decorator ──
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "delhi": {"temp": 45, "condition": "Sunny", "humidity": 45},
        "mumbai": {"temp": 32, "condition": "Humid", "humidity": 85},
        "bangalore": {"temp": 26, "condition": "Cloudy", "humidity": 65},
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"{city}: {data['temp']}°C, {data['condition']}, {data['humidity']}% humidity"
    return f"{city}: Weather data not available"

@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations. Input should be a valid math expression like '2 + 2' or '15 * 8'."""
    try:
        expression = expression.replace('%', '/100')
        allowed = set('0123456789+-*/()., ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return f"Invalid expression: {expression}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search company HR policies and knowledge base."""
    kb = {
        "leave policy": "Employees get 20 days annual leave per year.",
        "work from home": "WFH allowed up to 3 days per week with manager approval.",
        "dress code": "Business casual Monday-Thursday, casual on Fridays.",
        "insurance": "Health, dental and vision covered from day one.",
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return "No information found for that query."

tools = [get_weather, calculate, search_knowledge_base]

# ------ Step 2: Define State --------
class AgentState(TypedDict):
    messages: Annotated[list, operator.add] # messages accumulate

# ------- Step 3: Defind Nodes -----------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
).bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    """LLM decides what to do next"""
    print(f"\n Agent thinking... ({len(state['messages'])} messages in context)")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Decide whether to use a tool or end"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔧 Tools to call: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"
    print("✅ Agent finished")
    return END

tool_node = ToolNode(tools)

# ---- Ste 4: Build the Graph ----
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent") # after tools, always go back to agent

# ── Persistent Memory via SQLite ──
conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)

#  ── Chat function with thread_id ──
def chat(message : str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )

    answer = result["messages"][-1].content
    print(f"\n[Thread: {thread_id}]")
    print(f"User: {message}")
    print(f"Agent: {answer}")
    return answer


# ── Test 1: Basic memory ──
print("\n" + "="*60)
print("TEST 1: Basic Memory")
print("="*60)
chat("Hi! My name is Harsh and I'm a Django developer.", "user_harsh")
chat("I prefer Python over JavaScript.", "user_harsh")
chat("What's my name and what do I prefer?", "user_harsh")

# ── Test 2: Different user — no memory bleed ──
print("\n" + "="*60)
print("TEST 2: Different User")
print("="*60)
chat("What's my name?", "user_unknown")

# ── Test 3: Memory + tools together ──
print("\n" + "="*60)
print("TEST 3: Memory + Tools")
print("="*60)
chat("I'm planning to move to Bangalore.", "user_harsh")
chat("What's the weather like in the city I'm moving to?", "user_harsh")

# ── Test 4: Resume same thread after "restart" ──
print("\n" + "="*60)
print("TEST 4: Resume After Restart")
print("="*60)
# Simulate restart by creating new app instance with same DB
conn2 = sqlite3.connect("agent_memory.db", check_same_thread=False)
memory2 = SqliteSaver(conn2)
app2 = graph.compile(checkpointer=memory2)

def chat2(message: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app2.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    answer = result["messages"][-1].content
    print(f"\n[Thread: {thread_id}] [NEW APP INSTANCE]")
    print(f"User: {message}")
    print(f"Agent: {answer}")
    return answer

chat2("Do you remember my name?", "user_harsh")
chat2("And what city am I moving to?", "user_harsh")