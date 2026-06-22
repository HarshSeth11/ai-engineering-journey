from groq import Groq
import os
from dotenv import load_dotenv
# from agent.tools_discription import tools
import json
from tools.tool_dispatcher import dispatch_tool

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name e.g. Delhi, Mumbai"
                    }
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
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression e.g. '2 + 2' or '15 * 8'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time and date",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone e.g. IST, UTC"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search company HR policies and knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def agent(user_message: str):
    print(f"\n{'='*60}")
    print(f"User: {user_message}")

    messages = [
    {
        "role": "system",
        "content": """You are a helpful assistant with access to tools. Always use the appropriate tool to answer questions.
        Always trust tool results over your own calculations. 
        If a tool returns "No information found", say exactly that. Do not make up information."""
    },
    {"role": "user", "content": user_message}
]

    # First LLM Call - Decide which tool to use
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=500,
        tools=tools,
        tool_choice="auto",
        messages=messages
    )

    message = response.choices[0].message

    # Did the LLM want to use a tool?
    if message.tool_calls:
        # Add the full assistant message first
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls 
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 Tool called: {tool_name}")
            print(f"📥 Arguments: {tool_args}")

            # Execute the tool
            tool_result = dispatch_tool(tool_name, tool_args)
            print(f"📤 Result: {tool_result}")

            # Add tool result to conversation
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # Second LLM call - generate final answer using tool results
        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=500,
            messages=messages
        )

        answer = final_response.choices[0].message.content
    else:
        answer = message.content

    print(f"🤖 Answer: {answer}")
    return answer

# ── Test it ──
agent("What's the weather like in Delhi?")
agent("What is 1547 * 89 + 234?")
agent("What time is it right now?")
agent("What is our company's work from home policy?")
agent("What's the weather in Mumbai and what is 15% of 8500?")  # needs 2 tools