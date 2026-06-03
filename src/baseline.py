"""
baseline.py

the baseline system answers questions using only the system prompt and the
user's question. it does not retrieve any shakespeare passages, so the model
must rely entirely on what it already knows from training.

we compare this against the rag system to show that retrieval improves
the quality and grounding of the answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# add src/ to path so imports work when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LM_MODEL_NAME,
    LM_MAX_NEW_TOKENS,
    LM_TEMPERATURE,
    PROMPT_DIR,
)


# load the same system prompt the rag system uses so the two are comparable
def load_system_prompt() -> str:
    """
    read the system prompt text file from the prompts folder.

    outputs:
      the prompt as a string, or a hardcoded fallback if the file is missing.

    we use the same prompt as the rag system so the only difference between
    baseline and rag is whether retrieved context passages are included.
    """
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    if not prompt_path.exists():
        # fallback so the baseline still works even if the prompts folder is missing
        return (
            "You are a Shakespeare-aware assistant. "
            "Answer the question in a beginner-friendly way. "
            "Do not invent unsupported details."
        )
    return prompt_path.read_text(encoding="utf-8").strip()


# answer a question using only the language model with no retrieved context
def baseline_answer(query: str) -> str:
    """
    generate an answer with no retrieval - prompt only.

    inputs:
      query - the user's question as a plain string

    outputs:
      the model's answer as a string.

    the model sees only the system prompt and the question. there is no
    "retrieved context" section, which is the key difference from the rag system.
    that absence is the whole point of the baseline because it shows us what
    the model knows without any help from the shakespeare passages.
    """
    import ollama

    system_prompt = load_system_prompt()

    # note: there is no "retrieved context" section here, unlike the rag prompt.
    # that absence is intentional so we can measure how much retrieval actually helps.
    prompt = (
        f"{system_prompt}\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )

    response = ollama.chat(
        model=LM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": LM_MAX_NEW_TOKENS,
            "temperature": LM_TEMPERATURE,
        },
    )

    # dig into the nested response dict to get the answer text
    return response["message"]["content"].strip()


if __name__ == "__main__":
    question = "Why does Macbeth kill Duncan?"
    print("=" * 60)
    print("baseline system (prompt-only, no retrieval)")
    print("=" * 60)
    print(f"\nquestion: {question}\n")
    print("generating answer (this can take up to a minute on cpu)...\n")
    print("-" * 60)
    print("answer:")
    print("-" * 60)
    print(baseline_answer(question))
    print()
