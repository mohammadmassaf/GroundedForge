"""
Throwaway smoke test — verifies your Groq key works.
Run once, then you can delete this file.

Usage:
    python smoke_test.py
"""
import os
from dotenv import load_dotenv
import groq

load_dotenv()

key = os.getenv("GROQ_API_KEY")
if not key:
    raise SystemExit("GROQ_API_KEY not found in .env — copy .env.example to .env and add your key")

client = groq.Groq(api_key=key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Reply with exactly: Grounded Forge smoke test OK"}],
)

print(response.choices[0].message.content)
