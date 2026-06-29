import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_mcp_client():
    """Connect to MCP server and use its tools"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize connection
            await session.initialize()
            print("✅ Connected to MCP server")
            
            # List available tools
            tools = await session.list_tools()
            print(f"\n📦 Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # List available resources
            resources = await session.list_resources()
            print(f"\n📂 Available resources:")
            for resource in resources.resources:
                print(f"  - {resource.name}: {resource.uri}")
            
            # Call tools
            print(f"\n🔧 Testing tools:")
            
            weather = await session.call_tool("get_weather", {"city": "Delhi"})
            print(f"  Weather: {weather.content[0].text}")
            
            calc = await session.call_tool("calculate", {"expression": "1547 * 89 + 234"})
            print(f"  Calculate: {calc.content[0].text}")
            
            policy = await session.call_tool("search_hr_policy", {"query": "work from home"})
            print(f"  HR Policy: {policy.content[0].text}")
            
            # Read resources
            print(f"\n📖 Reading resources:")
            
            policies = await session.read_resource("company://policies/summary")
            print(f"  Policies:\n{policies.contents[0].text}")
            
            team = await session.read_resource("company://team/info")
            print(f"  Team: {team.contents[0].text}")

if __name__ == "__main__":
    asyncio.run(run_mcp_client())