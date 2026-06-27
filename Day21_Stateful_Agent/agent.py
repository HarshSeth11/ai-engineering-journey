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

# ── User Profile Manager ──
class UserProfileManger:
    """Manages persistent user profiles in SQLite"""

    def __init__(self, db_path: str = "user_profiles.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                          user_id TEXT PRIMARY KEY,
                          profile JSON,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
""")
        self.conn.commit()

    def get_profile(self, user_id: str) -> dict:
        cursor = self.conn.execute("SELECT profile FROM user_profiles WHERE user_id=?", (user_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return {}
    
    def update_profile(self, user_id: str, updates: dict):
        current = self.get_profile(user_id)

        for key, value in updates.items():

            # ignore empty values
            if value in ("", None, [], {}):
                continue

            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key].update(value)
            else:
                current[key] = value

        self.conn.execute(
            """
            INSERT OR REPLACE INTO user_profiles
            (user_id, profile, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (user_id, json.dumps(current))
        )

        self.conn.commit()
        return current
    
    def extract_and_save_facts(self, user_id: str, messages: list):
        """Extract key facts from conversation and save to profile"""
        conversation_text=""
        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                conversation_text += f"User: {msg.content}\n"

        if not conversation_text.strip():
            return
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=300,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Extract structured information about the user.

                Return ONLY valid JSON.

                Possible keys:
                - name
                - job
                - company
                - location
                - preferences
                - goals
                - skills

                IMPORTANT RULES:

                1. Only include information explicitly stated by the user.
                2. Never guess.
                3. Never infer.
                4. Never output null.
                5. Never output empty strings.
                6. If a field is not mentioned, omit it completely.
                7. If nothing new is learned, return {{}}.

                Conversation:
                {conversation_text}"""
            }]
        )

        try:
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            facts = json.loads(raw)
            if facts:
                self.update_profile(user_id, facts)
                print(f"  📋 Profile updated: {facts}")
        except:
            pass

profile_manager = UserProfileManger()

# ------- Step 2: Defind Nodes -----------

def manage_memory(messages: list, keep_last: int = 4) -> list:
    """Summarize old messages, keep recent ones fresh"""
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= keep_last:
        return messages
    
    old_messages = non_system[0:-keep_last]
    recent_messages = non_system[-keep_last:]

    # Built text to summerize
    conversation_text=""
    for msg in old_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                conversation_text += f"{role}: {content}\n"

    if not conversation_text.strip():
        return system_messages + recent_messages

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

    • identity
    • occupation
    • employer
    • preferences
    • goals
    • long-term projects
    • anything the assistant should remember

    Ignore:
    • greetings
    • small talk
    • assistant questions

    Ignore greetings and assistant questions.
    """
    },
    {
        "role": "user",
        "content": conversation_text
    }]


    summary_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        temperature=0,
        messages=messages_prompt
    )

    summary = summary_response.choices[0].message.content.strip()
    print(f"  📝 Summarized {len(old_messages)} old messages into: '{summary[:80]}...'")
    
    # Replace old messages with summary
    summary_message = SystemMessage(content=f"Earlier conversation summary: {summary}")
    return system_messages + [summary_message] + recent_messages


# ------ Step 3: Define State --------
class AgentState(TypedDict):
    messages: Annotated[list, operator.add] # messages accumulate
    user_id : str


def agent_node(state: AgentState) -> AgentState:
    """LLM decides what to do next"""
    user_id = state.get("user_id", "anonymous")

    # Get user profile
    profile = profile_manager.get_profile(user_id)
    
    # Build system message with profile content
    profile_context = ""
    if profile:
        profile_context += f"\n\nUser Profile: {json.dumps(profile)}"

    system_msg = SystemMessage(
        content=f"You are a helpful AI assistant with memory.{profile_context}\nUse this profile to personalize responses."
    )

    # Manage memory
    all_messages = state["messages"]
    messages_to_send = ""
    if len(all_messages) > 20:
        managed = manage_memory(all_messages, keep_last=6)

        # Ensure system message is first
        non_system = [m for m in managed if not isinstance(m, SystemMessage)]
        messages_to_send = [system_msg] + non_system
    else:
        messages_to_send = [system_msg] + all_messages


    print(f"\n Agent thinking... ({len(state['messages'])} messages in context)")

    response = llm.invoke(messages_to_send)

    # Extract and save messages after every 3 messages
    if isinstance(all_messages[-1], HumanMessage):
        profile_manager.extract_and_save_facts(user_id, all_messages)

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
conn = sqlite3.connect("stateful_agent.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)



# ── Chat Interface ──
def chat(message: str, user_id: str):
    config = {"configurable": {"thread_id": f"thread_{user_id}"}, "recursion_limit": 10}

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
        {
            "messages": messages,
            "user_id": user_id
        },
        config=config
    )
    
    answer = result["messages"][-1].content
    print(f"  User: {message}")
    print(f"  Agent: {answer}\n")
    return answer

def show_profile(user_id: str):
    profile = profile_manager.get_profile(user_id)
    print(f"\n📋 Profile for {user_id}:")
    print(json.dumps(profile, indent=2))

# ── Tests ──
print("\n" + "="*60)
print("TEST 1: Building User Profile")
print("="*60)
chat("Hi! I'm Harsh, a Django developer at TCS.", "harsh")
chat("I'm learning AI engineering and targeting Zepto and Razorpay.", "harsh")
chat("I prefer Python and FastAPI.", "harsh")
show_profile("harsh")

print("\n" + "="*60)
print("TEST 2: Personalized Responses")
print("="*60)
chat("What framework should I use for my AI backend?", "harsh")
chat("What's the weather in Delhi?", "harsh")

print("\n" + "="*60)
print("TEST 3: Profile Persists Across Sessions")
print("="*60)
# New app instance — simulates server restart
conn2 = sqlite3.connect("stateful_agent.db", check_same_thread=False)
memory2 = SqliteSaver(conn2)
app2 = graph.compile(checkpointer=memory2)

def chat2(message: str, user_id: str):
    config = {"configurable": {"thread_id": f"thread_{user_id}"}}
    result = app2.invoke(
        {"messages": [HumanMessage(content=message)], "user_id": user_id},
        config=config
    )
    answer = result["messages"][-1].content
    print(f"  [NEW SESSION] User: {message}")
    print(f"  Agent: {answer}\n")
    return answer

chat2("Do you remember what I do for work?", "harsh")
chat2("And what companies am I targeting?", "harsh")
show_profile("harsh")