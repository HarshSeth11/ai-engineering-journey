from tools.weather_tool import get_weather
from tools.calculate_tool import calculate
from tools.current_time_tool import get_current_time
from tools.knowledge_base_tool import search_knowledge_base
import json


def dispatch_tool(tool_name: str, tool_args: dict) ->str:
    """Execute the function the LLM chose"""
    tool_map = {
        "get_weather": get_weather,
        "calculate": calculate,
        "get_current_time": get_current_time,
        "search_knowledge_base": search_knowledge_base
    }

    if tool_name in tool_map:
        result = tool_map[tool_name](**tool_args)
        return json.dumps(result)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})