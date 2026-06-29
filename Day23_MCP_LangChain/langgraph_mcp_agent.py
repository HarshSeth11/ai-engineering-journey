import asyncio
import json
import os
import operator
import sqlite3
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
import functools

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Step 1: Discover tools from MCP server ──
async def get_mcp_tools(server_script: str) -> tuple[list, dict]:
    """Connect to MCP server and get available tools"""
    server_params = StdioServerParameters(
        command="python",
        args=[server_script]
    )
    
    tools_list = []
    tool_schemas = {}
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            mcp_tools = await session.list_tools()
            
            for mcp_tool in mcp_tools.tools:
                tool_schemas[mcp_tool.name] = mcp_tool.inputSchema
                tools_list.append({
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description,
                        "parameters": mcp_tool.inputSchema
                    }
                })
                print(f"  ✅ Discovered tool: {mcp_tool.name}")
    
    return tools_list, tool_schemas

# ── Step 3: State ──
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# ── Single persistent MCP session ──
async def build_and_run_agent(server_script: str, tasks: list[str]):
    
    server_params = StdioServerParameters(
        command="python",
        args=[server_script]
    )
    
    # One connection for everything
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            
            # Discover tools
            print("\n🔍 Discovering tools from MCP server...")
            mcp_tools = await mcp_session.list_tools()
            tools_schemas = []
            for t in mcp_tools.tools:
                tools_schemas.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                })
                print(f"  ✅ Discovered: {t.name}")
            print(f"✅ Found {len(tools_schemas)} tools\n")

            # ── Agent Node ──
            async def agent_node(state: AgentState) -> AgentState:
                messages = state["messages"]
                print(f"\n💭 Agent thinking... ({len(messages)} messages)")
                
                formatted_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        formatted_messages.append({"role": "system", "content": msg.content})
                    elif isinstance(msg, HumanMessage):
                        formatted_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            formatted_messages.append({
                                "role": "assistant",
                                "content": msg.content or "",
                                "tool_calls": [{
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["args"])
                                    }
                                } for tc in msg.tool_calls]
                            })
                        else:
                            formatted_messages.append({"role": "assistant", "content": msg.content})
                    elif isinstance(msg, ToolMessage):
                        formatted_messages.append({
                            "role": "tool",
                            "tool_call_id": msg.tool_call_id,
                            "content": msg.content
                        })
                
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=500,
                    temperature=0,
                    tools=tools_schemas,
                    tool_choice="auto",
                    messages=formatted_messages
                )
                
                msg = response.choices[0].message
                
                if msg.tool_calls:
                    ai_msg = AIMessage(
                        content=msg.content or "",
                        tool_calls=[{
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments)
                        } for tc in msg.tool_calls]
                    )
                else:
                    ai_msg = AIMessage(content=msg.content or "")
                
                return {"messages": [ai_msg]}

            # ── Tool Node — uses persistent session ──
            async def tool_node(state: AgentState) -> AgentState:
                last_message = state["messages"][-1]
                tool_messages = []
                
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    print(f"🔧 MCP Tool: {tool_name}({tool_args})")
                    
                    # Use persistent session — no new connection
                    result = await mcp_session.call_tool(tool_name, tool_args)
                    result_text = result.content[0].text
                    print(f"👁️ Result: {result_text}")
                    
                    tool_messages.append(ToolMessage(
                        content=result_text,
                        tool_call_id=tool_call["id"]
                    ))
                
                return {"messages": tool_messages}

            # ── Routing ──
            def should_continue(state: AgentState) -> str:
                last_message = state["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    return "tools"
                return END

            # ── Graph ──
            graph = StateGraph(AgentState)
            graph.add_node("agent", agent_node)
            graph.add_node("tools", tool_node)
            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
            graph.add_edge("tools", "agent")

            async with AsyncSqliteSaver.from_conn_string("mcp_agent.db") as memory:
                app = graph.compile(checkpointer=memory)
                
                for i, task in enumerate(tasks):
                    print(f"\n{'='*60}")
                    print(f"🎯 Task: {task}")
                    print(f"{'='*60}")
                    
                    config = {"configurable": {"thread_id": f"task_{i}"}}
                    
                    result = await app.ainvoke(
                        {
                            "messages": [
                                SystemMessage(content="You are a helpful assistant. Use MCP tools to answer questions. After getting tool results, provide the final answer directly without calling tools again."),
                                HumanMessage(content=task)
                            ]
                        },
                        config=config
                    )
                    
                    print(f"\n🤖 Answer: {result['messages'][-1].content}")
# ── Main ──
async def main():
    tasks = [
        "What's the weather in Mumbai?",
        "What is 15% of 85000?",
        "What is our work from home policy?",
        "Check weather in Delhi and Mumbai. Which is hotter and by how many degrees?",
    ]
    
    await build_and_run_agent("mcp_server.py", tasks)

if __name__ == "__main__":
    asyncio.run(main())