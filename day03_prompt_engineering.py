from groq import Groq
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

reviews = [
    # "Absolutely love this product, works perfectly!",
    # "Terrible. Broke after 2 days and support ignored me.",
    # "It's okay, nothing special but gets the job done.",
    "Best purchase I've made this year, highly recommend!",
    "Waste of money. Looks nothing like the pictures.",
    "Great product but shipping took 3 weeks and the box was damaged, though the item itself works fine."
]

def classify(prompt_style, review):
    if prompt_style == "zero_shot":
        prompt = f"Classify this review as positive, negative, or neutral for product and shipping. Use this exact format: Product: [positive/negative/neutral], Shipping: [positive/negative/neutral], Overall: [positive/negative/neutral] \n\nReview: {review}"
        token_needed = 100

    elif prompt_style == "few_shot":
        prompt = f"""Classify the LAST review only. Use this exact format:
    Product: [positive/negative/neutral], Shipping: [positive/negative/neutral], Overall: [positive/negative/neutral]

    Examples:
    "Amazing quality, love it!" → Product: positive, Shipping: neutral, Overall: positive
    "Stopped working after a week" → Product: negative, Shipping: neutral, Overall: negative
    "It's fine, nothing special" → Product: neutral, Shipping: neutral, Overall: neutral
    "Great product but shipping took 3 weeks and box damaged" → Product: positive, Shipping: negative, Overall: neutral

    Classify this review:
    "{review}" →"""
        token_needed = 200

    elif prompt_style == "chain_of_thought":
        prompt = f"""Classify this review as positive, negative, or neutral for product and shipping.
Think through it step by step, then end your response with:
Use this exact format:
    Product: [positive/negative/neutral], Shipping: [positive/negative/neutral], Overall: [positive/negative/neutral]

Review: {review}"""
        token_needed = 300

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=token_needed,
        temperature=0,  # consistency matters for classification
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Run all three techniques on every review
for review in reviews:
    print(f"\nReview: {review[:50]}...")
    print(f"  Zero-shot : {classify('zero_shot', review)}")
    print(f"  Few-shot  : {classify('few_shot', review)}")
    print(f"  CoT       : {classify('chain_of_thought', review)}")