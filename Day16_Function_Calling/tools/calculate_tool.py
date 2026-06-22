def calculate(expression: str) -> dict:
    """Safe calculator"""
    try:
        allowed = set('0123456789+-*/()., ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return {"expression": expression, "result": result}
        return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}