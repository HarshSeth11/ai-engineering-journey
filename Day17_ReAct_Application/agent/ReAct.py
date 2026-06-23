import json
import os
from groq import Groq
from dotenv import load_dotenv
from tools.tool_dispatcher import dispatch_tool


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Tools from Day 16 ──
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current time and date",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone e.g IST"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search company HR policies",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to search"}
                },
                "required": ["query"]
            }
        }
    }
]

# ── ReAct Agent Loop ──
def react_agent(user_task: str, max_iterations: int = 5):
    print(f"\n{'='*60}")
    print(f"🎯 Task: {user_task}")
    print(f"{'='*60}")

    messages = [
        {
            "role": "system",
            "content": """You are a ReAct agent. For every task:
1. THINK about what you need to do
2. ACT by calling the appropriate tool
3. OBSERVE the result
4. Repeat until you have enough information
5. Give a final comprehensive answer

Always use tools to get real information. Never guess.
Trust tool results completely."""
        },
        {"role": "user", "content": user_task}
    ]

    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=500,
            tools=tools,
            tool_choice="auto",
            messages=messages
        )

        message = response.choices[0].message
        stop_reason = response.choices[0].finish_reason

        print(f"💭 Stop reason: {stop_reason}")

        # ── Agent decided to use tools ──
        if message.tool_calls:
            # Add assistant message with all tool calls
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })

            # Execute each tool
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"🔧 ACT: {tool_name}({tool_args})")

                tool_result = dispatch_tool(tool_name, tool_args)
                print(f"👁️ OBSERVE: {tool_result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # ── Agent decided it's done ──
        elif stop_reason == "stop":
            print(f"\n✅ Agent finished after {iteration} iterations")
            print(f"\n🤖 Final Answer:\n{message.content}")
            return message.content

        else:
            print(f"⚠️ Unexpected stop reason: {stop_reason}")
            break

    print(f"\n⚠️ Max iterations ({max_iterations}) reached")
    return messages[-1].get("content", "Could not complete task")

# ── Test with increasingly complex tasks ──

# Simple — one tool
react_agent("What's the weather in Bangalore?")

# Medium — needs two tools in sequence
react_agent("What's the weather in Delhi and Mumbai? Which city is hotter?")

# Complex — multi-step reasoning
react_agent("""
I'm planning a team lunch. We have 12 people. 
Budget is Rs 500 per person. 
What's the total budget? 
Also what's the weather in Mumbai today so I know if we can eat outside?
""")

# Reasoning chain — needs to think between steps
react_agent("""
If it's hotter than 35 degrees in Delhi, 
we need to add Rs 200 per person for cold drinks.
We have 8 people. What's the total cost?
""")