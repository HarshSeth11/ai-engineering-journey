import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
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

# ---- Ste 4: Build the Graph ----

tool_node = ToolNode(tools)

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

# Compile
app = graph.compile()

# ---- Step : Run it ----
def run_agent(task: str):
    print(f"\n{'='*60}")
    print(f"🎯 Task: {task}")
    print(f"{'='*60}")

    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant. Use tools to answer questions accurately. Trust tool results completely."),
            HumanMessage(content=task)
        ]
    }

    final_state = app.invoke(initial_state)
    final_answer = final_state["messages"][-1].content

    print(f"\n🤖 Final Answer: {final_answer}")
    return final_answer

# ── Tests ──
# run_agent("What's the weather in Bangalore?")
# run_agent("What's the weather in Delhi and Mumbai? Which city is hotter?")
# run_agent("What is 1547 * 89 + 234?")
# run_agent("What is our work from home policy?")
# run_agent("""
# I'm planning a team lunch for 12 people with Rs 500 per person budget.
# What's the total budget? Also check Mumbai weather to see if we can eat outside.
# """)

run_agent("""
Step 1: Check weather in Delhi.
Step 2: If temperature is above 35 degrees, calculate how much ice cream budget 
we need for 10 people at Rs 50 per person per degree above 35.
Step 3: Also check Mumbai weather and do the same calculation.
Step 4: Tell me which city needs more ice cream budget and the total combined budget.
""")