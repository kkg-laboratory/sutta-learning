#!/usr/bin/env python3
"""
Sutta Creator.

Generate a new Buddhist-style sutta using the Groq API, print it beautifully
in the terminal, and save it to a markdown file under suttas/<difficulty>/.

Example:
    python sutta_creator.py \
        --source "Speak kindly, act carefully, and let go of grasping." \
        --output metta-sutta.md \
        --difficulty beginner

Environment:
    GROQ_API_KEY must be available in the environment or in a .env file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

DEFAULT_MODEL: Final[str] = "llama-3.3-70b-versatile"
BASE_OUTPUT_DIR: Final[Path] = Path("suttas")
DIFFICULTY_LEVELS: Final[tuple[str, ...]] = (
    "beginner",
    "novice",
    "advanced",
    "master",
)

console = Console()


def load_environment_variables(dotenv_path: str = ".env") -> None:
    """
    Load environment variables from a .env file.

    Args:
        dotenv_path: Path to the .env file.
    """
    load_dotenv(dotenv_path=dotenv_path)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a new sutta using the Groq API."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source line or Buddhist-friendly comment used to generate the sutta.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output markdown filename. This will be created inside "
            "suttas/<difficulty>/. Example: my-sutta.md"
        ),
    )
    parser.add_argument(
        "--difficulty",
        required=False,
        default="beginner",
        choices=DIFFICULTY_LEVELS,
        help="Difficulty level for the sutta and output folder.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Groq model to use. Default: {DEFAULT_MODEL}",
    )
    return parser.parse_args()


def validate_api_key() -> str:
    """
    Validate and return the Groq API key from the environment.

    Returns:
        The Groq API key.

    Raises:
        RuntimeError: If GROQ_API_KEY is missing.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file."
        )
    return api_key


def get_groq_client() -> Groq:
    """
    Create and return a Groq client.

    Returns:
        An initialized Groq client.
    """
    api_key = validate_api_key()
    return Groq(api_key=api_key)


def get_output_directory(difficulty: str) -> Path:
    """
    Return the output directory for a given difficulty level.

    Args:
        difficulty: Difficulty level.

    Returns:
        Output directory path.
    """
    return BASE_OUTPUT_DIR / difficulty


def normalize_output_path(output_name: str, difficulty: str) -> Path:
    """
    Normalize the output filename so it is always written inside suttas/<difficulty>/.

    Args:
        output_name: User-provided output filename.
        difficulty: Difficulty level.

    Returns:
        Resolved output path inside suttas/<difficulty>/.
    """
    filename = Path(output_name).name
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    return get_output_directory(difficulty) / filename


def get_difficulty_prompt_rules(difficulty: str) -> str:
    """
    Return prompt rules tailored to the requested difficulty.

    Args:
        difficulty: Difficulty level.

    Returns:
        Difficulty-specific prompt guidance.
    """
    rules = {
        "beginner": """
- Use very simple plain English.
- Keep the teaching gentle, welcoming, and easy to understand.
- Avoid heavy doctrinal vocabulary unless briefly explained.
- Include practical examples from daily life.
- Keep sections short and clear.
""".strip(),
        "novice": """
- Use accessible English with a little more Buddhist vocabulary.
- Explain key terms when used.
- Include a balance of clarity, reflection, and light doctrine.
- Encourage steady practice and moral discipline.
- Keep the structure readable and moderately detailed.
""".strip(),
        "advanced": """
- Use more depth in doctrine, contemplation, and philosophical framing.
- You may include terms such as impermanence, non-self, dependent origination,
  mindfulness, concentration, and liberation when relevant.
- Assume the reader already has some familiarity with practice.
- Include deeper reflections and more nuanced lessons.
""".strip(),
        "master": """
- Write in a profound, subtle, contemplative, and spiritually refined manner.
- Use deeper Dhamma framing, careful paradox, insight language, and advanced reflection.
- Assume the reader is highly mature in practice and can handle dense contemplative material.
- Emphasize insight, release, emptiness, non-clinging, wisdom, and liberation.
- Keep the language beautiful and precise rather than merely complex.
""".strip(),
    }
    return rules[difficulty]


def build_prompt(sutta_source: str, difficulty: str) -> str:
    """
    Build the prompt sent to the Groq model.

    Args:
        sutta_source: Source line for the sutta.
        difficulty: Difficulty level.

    Returns:
        Prompt text.
    """
    difficulty_rules = get_difficulty_prompt_rules(difficulty)

    return f"""
Create a new original Buddhist-inspired sutta based on the following source:

{sutta_source}

Difficulty level: {difficulty}

Difficulty guidance:
{difficulty_rules}

Requirements:
1. Return valid markdown only.
2. Start with a clear title as a level-1 markdown heading.
3. Include these sections:
   - Meaning
   - Reflection
   - Practice
   - Lessons
4. Make it poetic, clear, readable, and spiritually grounded.
5. Do not mention that it was generated by AI.
6. Include the source and the origin of the sutta in plain English.
7. Align the tone, vocabulary, and conceptual depth with the requested difficulty level.
""".strip()


def generate_sutta(sutta_source: str, difficulty: str, model: str) -> str:
    """
    Generate a new sutta from the provided source using the Groq API.

    Args:
        sutta_source: Source line for the sutta.
        difficulty: Difficulty level.
        model: Groq model name.

    Returns:
        Generated sutta content as markdown text.

    Raises:
        RuntimeError: If no content is returned.
    """
    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a wise Buddhist monk who knows the Noble Way and the Dhamma deeply. "
                    "You write gentle, clear, original suttas in beautiful markdown. "
                    "You carefully adapt your style to the requested difficulty level."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(sutta_source, difficulty),
            },
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()


def ensure_output_directory(path: Path) -> None:
    """
    Ensure the parent directory for the output file exists.

    Args:
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def write_markdown_file(output_path: Path, content: str) -> None:
    """
    Write markdown content to the specified file.

    Args:
        output_path: Path to the markdown file.
        content: Markdown content to write.
    """
    ensure_output_directory(output_path)
    output_path.write_text(content, encoding="utf-8")


def print_sutta_to_terminal(content: str, output_path: Path, difficulty: str) -> None:
    """
    Print the sutta to the terminal with aesthetic markdown rendering.

    Args:
        content: Markdown content to print.
        output_path: Output path where the file was saved.
        difficulty: Difficulty level used.
    """
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Sutta generated successfully[/bold green]\n"
            f"[cyan]Difficulty:[/cyan] {difficulty}\n"
            f"[cyan]Saved to:[/cyan] {output_path}",
            border_style="bright_blue",
            title="Sutta Creator",
        )
    )
    console.print()
    console.print(Markdown(content))
    console.print()


def main() -> int:
    """
    Run the Sutta Creator CLI.

    Returns:
        Exit status code.
    """
    try:
        load_environment_variables()
        args = parse_args()

        output_path = normalize_output_path(args.output, args.difficulty)
        sutta_markdown = generate_sutta(
            sutta_source=args.source,
            difficulty=args.difficulty,
            model=args.model,
        )

        write_markdown_file(output_path, sutta_markdown)
        print_sutta_to_terminal(sutta_markdown, output_path, args.difficulty)

        return 0

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        return 130
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
