import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .ai_engine import answer_question

@csrf_exempt
@require_http_methods(["POST"])
def ask(request):
    try:
        body = json.loads(request.body)
        query = body.get("question", "").strip()
        
        if not query:
            return JsonResponse({"error": "question is required"}, status=400)
        
        result = answer_question(query)
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "ok", "message": "AI Q&A API is running"})