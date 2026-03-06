import os
import json
from dotenv import load_dotenv

from groq import Groq

load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="openai/gpt-oss-20b",
)

# print(chat_completion.choices[0].message.content)
# print(type(chat_completion))
# print(chat_completion)

# Turn chat completion into a dictionary and dump as json
chat_completion_dict = chat_completion.model_dump()
with open("gpt_oss_20b.json", "w") as f:
    json.dump(chat_completion_dict, f, indent=4)
