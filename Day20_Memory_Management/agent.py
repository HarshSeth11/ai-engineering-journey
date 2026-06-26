import os
import json
from groq import Groq
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

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
).bind_tools(tools)

# ------- Step 2: Defind Nodes -----------

# Memory Strategies 
def sliding_window(messages: list, window_size: int = 6) -> list:
    """Keep only the last k messages"""
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) > window_size:
        print(f"  🪟 Sliding window: keeping last {window_size} of {len(non_system)} messages")
        non_system = non_system[-window_size:]
    
    return system_messages+non_system

def summarize_old_messages(messages: list, keep_last: int = 4) -> list:
    """Summarize old messages, keep recent ones fresh"""
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) < 10:
        return messages
    
    old_messages = non_system[0:-keep_last]
    recent_messages = non_system[-keep_last:]

    # Built text to summerize
    conversation_text = ""

    for msg in old_messages:
        if isinstance(msg, HumanMessage):
            role = "User"

        elif isinstance(msg, AIMessage):
            if not msg.content:      # Skip tool-call AI messages
                continue
            role = "Assistant"

        else:
            continue                 # Skip ToolMessage/SystemMessage

        conversation_text += f"{role}: {msg.content}\n"

    messages_prompt=[{
    "role": "system",
    "content": """
You summarize conversations.

Always produce a summary.

Never say:
- There is no conversation.
- There is nothing to summarize.
- The conversation just started.

Return only the important long-term facts.

Keep:
- name
- occupation
- preferences
- goals

Ignore greetings and assistant questions.
"""
},
{
    "role": "user",
    "content": conversation_text
}]

    if not conversation_text.strip():
        return system_messages+recent_messages

    summary_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        temperature=0,
        messages=messages_prompt
    )

    summary = summary_response.choices[0].message.content.strip()
    print(f"  📝 Summarized {len(old_messages)} old messages into: '{summary[:80]}...'")
    
    # Replace old messages with summary
    summary_message = SystemMessage(content=f"Previous conversation summary: {summary}")
    return system_messages + [summary_message] + recent_messages

def extract_key_facts(messages: list) -> str:
    """Extract important user facts from conversation"""
    conversation_text = ""
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                conversation_text += f"{role}: {content}\n"
    
    if not conversation_text.strip():
        return ""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Extract key facts about the user from this conversation.
Format as bullet points. Only include concrete facts (name, location, preferences, plans).
If no facts found, return "No key facts found."

Conversation:
{conversation_text}

Key facts:"""
        }]
    )
    
    return response.choices[0].message.content.strip()

# ------ Step 3: Define State --------
class AgentState(TypedDict):
    messages: Annotated[list, operator.add] # messages accumulate
    memory_strategy : str


def agent_node(state: AgentState) -> AgentState:
    """LLM decides what to do next"""
    strategy = state.get("memory_strategy", "sliding_window")
    messages = state["messages"]

    # Apply memory strategy before LLM call
    if strategy == "sliding_window":
        messages = sliding_window(messages, window_size=6)
    elif strategy == "summarization":
        messages = summarize_old_messages(messages, keep_last=4)

    print(f"\n Agent thinking... ({len(state['messages'])} messages in context)")
    response = llm.invoke(messages)

    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Decide whether to use a tool or end"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
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
conn = sqlite3.connect("memory_strategies.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)

#  ── Chat function with thread_id ──
def chat(message : str, thread_id: str = "default", strategy: str = "sliding_window"):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}

    state = app.get_state(config)

    messages = []

    # First interaction of this thread
    if not state.values:
        messages.append(
            SystemMessage(
                content="""
You are a helpful assistant.

Do NOT use tools when the user is simply sharing information.

Examples:

User: My name is Harsh.
→ Reply normally.

User: I work at TCS.
→ Reply normally.

User: I like Python.
→ Reply normally.

Only use search_knowledge_base when the user explicitly asks about company policies or HR information.

Only use calculate when the user asks for a calculation.

Only use get_weather when the user asks about weather.

If a tool returns "No information found", do not retry with different queries. Inform the user that the requested information isn't available.
"""
            )
        )

    messages.append(HumanMessage(content=message))

    result = app.invoke(
        {"messages": messages,
        "memory_strategy": strategy
        },
        config=config
    )

    answer = result["messages"][-1].content
    print(f"User: {message}")
    print(f"Agent: {answer}")
    return answer

# ── Test Strategy 1: Sliding Window ──
# print("\n" + "="*60)
# print("STRATEGY 1: Sliding Window (keep last 6 messages)")
# print("="*60)

# thread = "test_sliding"
# chat("My name is Harsh.", thread, "sliding_window")
# chat("I work at TCS.", thread, "sliding_window")
# chat("I prefer Python.", thread, "sliding_window")
# chat("I'm learning AI engineering.", thread, "sliding_window")
# chat("My target companies are Zepto and Razorpay.", thread, "sliding_window")
# chat("What's the weather in Delhi?", thread, "sliding_window")
# chat("What do you know about me?", thread, "sliding_window")  # will some info be lost?

# ── Test Strategy 2: Summarization ──
print("\n" + "="*60)
print("STRATEGY 2: Summarization (compress old messages)")
print("="*60)

thread2 = "test_summary"
chat("My name is Harsh.", thread2, "summarization")
chat("I work at TCS as an AI engineer.", thread2, "summarization")
chat("I prefer FastAPI over Django now.", thread2, "summarization")
chat("I'm targeting Zepto and Meesho.", thread2, "summarization")
chat("What's 500 * 12?", thread2, "summarization")
chat("What do you know about me?", thread2, "summarization")

# ── Key Fact Extraction ──
print("\n" + "="*60)
print("KEY FACT EXTRACTION")
print("="*60)

from langgraph.checkpoint.sqlite import SqliteSaver
conn3 = sqlite3.connect("memory_strategies.db", check_same_thread=False)
memory3 = SqliteSaver(conn3)
app3 = graph.compile(checkpointer=memory3)

config = {"configurable": {"thread_id": "test_summary"}}
state = app3.get_state(config)
if state and state.values:
    messages = state.values.get("messages", [])
    facts = extract_key_facts(messages)
    print(f"Extracted facts from conversation:\n{facts}")