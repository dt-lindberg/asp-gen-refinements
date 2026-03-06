"""Test the Gemini API response structure"""

import os
import json

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Test the structure of the response
def test_response_structure():
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="What is the capital of France?")],
            )
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
            temperature=0.0,
            max_output_tokens=1500,
        ),
    )

    print(response)
    print("-" * 80)
    print(response.model_dump_json(indent=2))
    # print(json.dumps(response, indent=4))


if __name__ == "__main__":
    test_response_structure()
