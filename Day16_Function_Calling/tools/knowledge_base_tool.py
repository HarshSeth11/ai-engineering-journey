

def search_knowledge_base(query: str) -> dict:
    """Search company knowledge base"""
    kb = {
        "leave policy": "Employees get 20 days annual leave per year.",
        "work from home": "WFH allowed up to 3 days per week with manager approval.",
        "dress code": "Business casual Monday-Thursday, casual on Fridays.",
        "insurance": "Health, dental and vision covered from day one.",
    }

    for key, value in kb.items():
        if key in query.lower():
            return {"query": query, "result": value}
    return {"query": query, "result": "No information found."}