"""
Common Tool Definitions and Implementations

This is like a "ToolFactory" + implementations in Java.
Contains:
- Tool creators (factory methods returning Tool objects)
- Actual function implementations (the real logic)
- Validators (pre-execution validation)
"""

import math
from typing import Optional

from models import Tool, ToolParameter


# ─────────────────────────────────────────────────────────────
# TOOL CREATORS (Factory methods)
# ─────────────────────────────────────────────────────────────


def create_weather_tool() -> Tool:
    """
    create a weather lookup tool

    Example LLM usage:
    User: "What's the weather in Tokyo?"
    LLM: calls get_weather(location="Tokyo", unit="celsius")
    """
    return Tool(
        name="get_weather",
        description="Get the current weather for a location. Use this when the user asks about weather, temperature, or climate conditions.",
        parameters=[
            ToolParameter(
                name="location",
                type="string",
                description="The city and optionally country, e.g., 'Tokyo', 'London, UK', 'New York, NY'",
                required=True,
            ),
            ToolParameter(
                name="unit",
                type="string",
                description="Temperature unit",
                required=False,
                enum=["celsius", "fahrenheit"],
                default="celsius",
            ),
        ],
    )


def create_calculator_tool() -> Tool:
    """
    create a calculator tool

    Example LLM usage:
    User: "What is 15% of 250?"
    LLM: calls calculate(expression="250 * 0.15")
    """
    return Tool(
        name="calculate",
        description="Perform mathematical calculations. Use this for any math operations like addition, subtraction, multiplication, division, percentages, etc.",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="The mathematical expression to evaluate, e.g., '2 + 2', '15 * 7', '100 / 4', 'sqrt(16)'",
                required=True,
            ),
        ],
    )


def create_search_tool() -> Tool:
    """
    create a web search tool

    Example LLM usage:
    User: "Find information about Python 3.12 features"
    LLM: calls web_search(query="Python 3.12 new features")
    """
    return Tool(
        name="web_search",
        description="Search the web for current information. Use this when you need up-to-date information, facts you're unsure about, or recent events.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
            ToolParameter(
                name="num_results",
                type="number",
                description="Number of results to return (1-10)",
                required=False,
                default=5,
            ),
        ],
    )


def create_database_query_tool() -> Tool:
    """
    create a database query tool

    Example LLM usage:
    User: "How many users signed up last month?"
    LLM: calls query_database(query="SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'")
    """
    return Tool(
        name="query_database",
        description="Execute a read-only SQL query against the database. Use this to retrieve data, counts, or statistics.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The SQL SELECT query to execute. Only SELECT queries are allowed.",
                required=True,
            ),
        ],
    )


def create_send_email_tool() -> Tool:
    """
    create an email sending tool

    Example LLM usage:
    User: "Send an email to john@example.com about the meeting"
    LLM: calls send_email(to="john@example.com", subject="Meeting", body="...")
    """
    return Tool(
        name="send_email",
        description="Send an email to a recipient. Use this when the user explicitly asks to send an email.",
        parameters=[
            ToolParameter(
                name="to",
                type="string",
                description="Recipient email address",
                required=True,
            ),
            ToolParameter(
                name="subject",
                type="string",
                description="Email subject line",
                required=True,
            ),
            ToolParameter(
                name="body",
                type="string",
                description="Email body content",
                required=True,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────
# IMPLEMENTATIONS (The actual functions)
# ─────────────────────────────────────────────────────────────


def get_weather(location: str, unit: str = "celsius") -> dict:
    """
    mock weather function (in production, call real weather API)

    Returns:
        dict with temperature, condition, humidity
    """
    # mock data based on location
    mock_data = {
        "tokyo": {"temp": 22, "condition": "cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "rainy", "humidity": 80},
        "new york": {"temp": 18, "condition": "sunny", "humidity": 55},
        "paris": {"temp": 17, "condition": "partly cloudy", "humidity": 60},
    }

    location_lower = location.lower()
    for city, data in mock_data.items():
        if city in location_lower:
            temp = data["temp"]
            if unit == "fahrenheit":
                temp = round(temp * 9 / 5 + 32)
            return {
                "location": location,
                "temperature": temp,
                "unit": unit,
                "condition": data["condition"],
                "humidity": data["humidity"],
            }

    # default for unknown locations
    return {
        "location": location,
        "temperature": 20 if unit == "celsius" else 68,
        "unit": unit,
        "condition": "unknown",
        "humidity": 50,
    }


def calculate(expression: str) -> dict:
    """
    safe calculator function

    Supports basic math operations and some functions like sqrt, abs, etc.
    """
    # whitelist of allowed names for safety
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # evaluate expression safely
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {
            "expression": expression,
            "result": result,
        }
    except Exception as e:
        raise ValueError(f"cannot evaluate '{expression}': {e}")


def web_search(query: str, num_results: int = 5) -> dict:
    """
    mock web search (in production, call real search API like Tavily)
    """
    # mock results
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a mock search result about {query}...",
                "url": f"https://example.com/result{i+1}",
            }
            for i in range(min(num_results, 5))
        ],
    }


# ─────────────────────────────────────────────────────────────
# VALIDATORS (Pre-execution checks)
# ─────────────────────────────────────────────────────────────


def validate_calculate_args(args: dict) -> Optional[str]:
    """validator for calculator - check for dangerous operations"""
    expression = args.get("expression", "")

    # block dangerous operations
    dangerous = ["import", "exec", "eval", "open", "file", "__"]
    for d in dangerous:
        if d in expression.lower():
            return f"expression contains forbidden term: {d}"

    return None  # no error
