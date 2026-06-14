from groq import Groq
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Experiment 1 — basic call
# response = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     max_tokens=300,
#     messages=[{"role": "user", "content": "Explain black holes in 3 sentences."}]
# )
# print(response.choices[0].message.content)
# print("--- STOP REASON:", response.choices[0].finish_reason)
# print("--- TOKENS USED:", response.usage)

# # Experiment 2 — system prompt
# response = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     max_tokens=300,
#     messages=[
#         {"role": "system", "content": "You are a pirate. Respond only in pirate speak."},
#         {"role": "user", "content": "Explain black holes in 3 sentences."}
#     ]
# )
# print(response.choices[0].message.content)


#Temperature controls the randomness of the output. A temperature of 0 makes the model deterministic, while higher values (e.g., 1) make it more creative and diverse. Let's see how the output changes with different temperature settings.

print("--- temperature=0 (deterministic) ---")
for i in range(3):
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=60,
        temperature=0,
        messages=[{"role": "user", "content": "Give me a one-line fun fact about space."}]
    )
    print(f"{i+1}:", r.choices[0].message.content)

print("\n--- temperature=1 (creative) ---")
for i in range(3):
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=60,
        temperature=1,
        messages=[{"role": "user", "content": "Give me a one-line fun fact about space."}]
    )
    print(f"{i+1}:", r.choices[0].message.content)