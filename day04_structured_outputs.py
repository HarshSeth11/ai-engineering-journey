from groq import Groq
import os
from dotenv import load_dotenv
import json

# Load variables from the .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def analyze_review(review):
    prompt = f"""Analyze this product review and return a JSON object only.
No explanation, no markdown, no code blocks. Raw JSON only.

Return exactly this structure:
{{
  "product_sentiment": "positive/negative/neutral",
  "shipping_sentiment": "positive/negative/neutral", 
  "overall_sentiment": "positive/negative/neutral",
  "key_issues": ["list", "of", "problems"],
  "key_positives": ["list", "of", "good", "things"],
  "recommend_follow_up": true/false
}}

Set recommend_follow_up to true if the review mentions any problem that needs attention.

Review: {review}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Parse JSON safely
    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        return {"error": "Failed to parse", "raw": raw}

# Test it
reviews = [
    "Best purchase I've made this year, highly recommend!",
    "Waste of money. Looks nothing like the pictures.",
    "Great product but shipping took 3 weeks and the box was damaged, though the item itself works fine."
]

# for review in reviews:
#     print(f"\nReview: {review[:50]}...")
#     result = analyze_review(review)
#     print(json.dumps(result, indent=2))

def process_reviews(reviews):
    results = []
    
    for review in reviews:
        data = analyze_review(review)
        
        if "error" in data:
            print(f"⚠ Parse failed: {data['raw']}")
            continue
            
        results.append(data)
        
        # Real business logic driven by AI output
        if data.get("recommend_follow_up"):
            print(f"🚨 Follow-up needed: {data['key_issues']}")
        
        if data.get("overall_sentiment") == "negative":
            print(f"👎 Negative review flagged for review team")
            
    # Aggregate stats
    sentiments = [r["overall_sentiment"] for r in results]
    print(f"\n📊 Summary:")
    print(f"  Positive: {sentiments.count('positive')}")
    print(f"  Neutral:  {sentiments.count('neutral')}")
    print(f"  Negative: {sentiments.count('negative')}")
    print(f"  Follow-ups needed: {sum(1 for r in results if r.get('recommend_follow_up'))}")

process_reviews(reviews)