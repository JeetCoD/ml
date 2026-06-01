"""
baseline.py

the baseline system for comparison with the rag system.

design choice: prompt-only generation, no retrieval.
the baseline uses the SAME language model (qwen2.5:1.5b via ollama)
and the SAME system prompt as the rag system, but it receives NO
retrieved shakespeare passages. it answers only from the model's
own general knowledge.

why this is a fair baseline:
  the only difference between the baseline and the rag system is the
  retrieved context. so any difference in answer quality between the
  two can be attributed to retrieval, not to the underlying model.
  this isolates the contribution of rag.

usage:
  python3 src/baseline.py
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


def load_system_prompt() -> str:
    """
    read the same system prompt the rag system uses, so the two
    systems differ only in whether they get retrieved context.
    """
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    if not prompt_path.exists():
        return (
            "You are a Shakespeare-aware assistant. "
            "Answer the question in a beginner-friendly way. "
            "Do not invent unsupported details."
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def baseline_answer(query: str) -> str:
    """
    generate an answer with no retrieval.
    the model sees only the system prompt and the question -
    no shakespeare passages are provided.
    """
    import ollama

    system_prompt = load_system_prompt()

    # note: there is NO "retrieved context" section here, unlike the
    # rag prompt. that absence is the whole point of the baseline.
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