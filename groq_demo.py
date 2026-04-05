#
# Title: Groq Demo
# Description: This script demonstrates how to use the Groq API to generate a chat completion.
# Author: @adityapatange_
# Date: 2026-04-05
# Version: 1.0.0
#

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


def load_environment_variables():
    load_dotenv(dotenv_path=".env")


def get_groq_client():
    return Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def generate_chat_completion(messages):
    client = get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )
    return chat_completion


def main():
    load_environment_variables()
    chat_completion = generate_chat_completion(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
