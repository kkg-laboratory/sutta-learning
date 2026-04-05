#
# Title: Sutta Creator.
# Description: This script creates a new Sutta using the Groq API.
# Author: @adityapatange_
# Date: 2026-04-05
# Version: 1.0.0
#
#

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


def load_environment_variables():
    """
    Loads the environment variables from the .env file.

    Returns:
        None.
    """
    load_dotenv(dotenv_path=".env")


def get_sutta_source():
    """
    Gets the Sutta source from the user.

    Returns:
        The Sutta source from the user.
    """
    sutta_source = input("Enter a Buddhist friendly comment to generate a new Sutta: ")
    return sutta_source


def get_groq_client():
    """
    Gets the Groq client.

    Returns:
        The Groq client.
    """
    return Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def generate_sutta(sutta_source):
    """
    Generates a new Sutta based on the Sutta source.

    Args:
        sutta_source: The Sutta source to generate a new Sutta from.

    Returns:
        The chat completion from the Groq API.

    Raises:
        Exception: If the chat completion is not successful.
    """
    client = get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Create a new Sutta based on the following source: {sutta_source}. The Sutta should be in the following format: <sutta_name> <sutta_content>",
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion


def main():
    """
    Main function to run the Sutta Creator.
    """
    load_environment_variables()
    sutta_source = get_sutta_source()
    chat_completion = generate_sutta(sutta_source=sutta_source)
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
