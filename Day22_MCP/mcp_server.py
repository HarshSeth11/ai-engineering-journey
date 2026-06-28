from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import json
import asyncio

# Create MCP server
server = Server("company-assistant")

# ── Define Tools ──
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Tell clients what tools are available"""
    return [
        types.Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name e.g. Delhi, Mumbai"
                    }
                },
                "required": ["city"]
            }
        ),
        types.Tool(
            name="calculate",
            description="Perform mathematical calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            }
        ),
        types.Tool(
            name="search_hr_policy",
            description="Search company HR policies",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Policy topic to search"
                    }
                },
                "required": ["query"]
            }
        )
    ]

# ── Implement Tools ──
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool when called by the LLM"""
    
    if name == "get_weather":
        city = arguments.get("city", "")
        weather_data = {
            "delhi": {"temp": 45, "condition": "Sunny", "humidity": 45},
            "mumbai": {"temp": 32, "condition": "Humid", "humidity": 85},
            "bangalore": {"temp": 26, "condition": "Cloudy", "humidity": 65},
        }
        data = weather_data.get(city.lower())
        if data:
            result = f"{city}: {data['temp']}°C, {data['condition']}, {data['humidity']}% humidity"
        else:
            result = f"{city}: Weather data not available"
    
    elif name == "calculate":
        expression = arguments.get("expression", "")
        try:
            expression = expression.replace('%', '/100')
            allowed = set('0123456789+-*/()., ')
            if all(c in allowed for c in expression):
                result = f"{expression} = {eval(expression)}"
            else:
                result = "Invalid expression"
        except Exception as e:
            result = f"Error: {str(e)}"
    
    elif name == "search_hr_policy":
        query = arguments.get("query", "")
        kb = {
            "leave policy": "Employees get 20 days annual leave per year.",
            "work from home": "WFH allowed up to 3 days per week with manager approval.",
            "dress code": "Business casual Monday-Thursday, casual on Fridays.",
            "insurance": "Health, dental and vision covered from day one.",
        }
        result = "No information found."
        for key, value in kb.items():
            if key in query.lower():
                result = value
                break
    
    else:
        result = f"Unknown tool: {name}"
    
    return [types.TextContent(type="text", text=result)]

# ── Define Resources ──
@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Expose data resources the LLM can read"""
    return [
        types.Resource(
            uri="company://policies/summary",
            name="Company Policies Summary",
            description="Summary of all company policies",
            mimeType="text/plain"
        ),
        types.Resource(
            uri="company://team/info",
            name="Team Information",
            description="Current team size and structure",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Return resource content when requested"""
    if str(uri) == "company://policies/summary":
        return """Company Policy Summary:
- Leave: 20 days annual, 10 days sick
- WFH: Up to 3 days/week with approval
- Insurance: Health, dental, vision from day 1
- Reviews: Twice yearly in June and December
- Travel: Economy under 6hrs, Business over 6hrs"""
    
    elif str(uri) == "company://team/info":
        return json.dumps({
            "total_employees": 150,
            "engineering": 60,
            "product": 20,
            "design": 15,
            "operations": 55
        })
    
    return "Resource not found"

# ── Run the server ──
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())