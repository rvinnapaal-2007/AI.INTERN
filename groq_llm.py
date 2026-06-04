# AI.INTERN
from groq import Groq

# Inbuilt API Key
client = Groq(
    api_key="your_api_key_here"
)
# Take user question
user_prompt = input("Ask something to the LLM:\n ")

# Send request to Groq model
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)

# Print response
print("\nLLM Response:\n")
print(response.choices[0].message.content)
