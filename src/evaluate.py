"""
evaluate.py

this script runs the same evaluation questions through two systems:
  1. baseline system: prompt-only, no retrieved shakespeare context
  2. rag system: retrieval + generated answer + displayed evidence

it writes one row per (question, system) pair to:
  results/evaluation_results.csv

after the script finishes, you should score each row manually
using the rubric in results/scoring_rubric.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# allow running from either project root or src/ so imports work either way
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_TOP_K, RESULTS_DIR
from baseline import baseline_answer
from rag_chatbot import (
    build_rag_prompt,
    build_stylised_prompt,
    generate_answer,
    setup_retriever,
)
from chunking import format_chunk_for_display


# paths to the input questions file and the two output files we write
QUESTIONS_PATH = RESULTS_DIR / "evaluation_questions.json"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary_template.csv"


# column names for the detailed results csv, one row per (question, system) pair
FIELDNAMES = [
    "question_id",
    "source",
    "question_type",
    "play",
    "question",
    "expected_focus",
    "system",
    "retrieved_passages",
    "generated_response",
    "correctness_score",
    "grounding_score",
    "retrieval_relevance_score",
    "usefulness_score",
    "style_quality_score",
    "comments",
]


# read the evaluation questions from the json file
def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, str]]:
    """
    load the evaluation question list from the json file on disk.

    inputs:
      path - path to the json file (defaults to results/evaluation_questions.json)

    outputs:
      a list of question dicts, each with fields like question_id, question, play, type, etc.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation question file not found: {path}\n"
            "Expected file: results/evaluation_questions.json"
        )
    # load the json file so we can read the questions inside it
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # the file can be either a plain list or a dict with a "questions" key, so handle both
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    return data


# check if a question is asking for a creative stylised response
def is_stylised_question(question: Dict[str, str]) -> bool:
    """
    return true if this question expects a creative shakespearean-style answer.

    inputs:
      question - a question dict with at least "type" and "question" fields

    outputs:
      true if the type or question text contains the word "stylised", false otherwise.
    """
    q_type = question.get("type", "").lower()
    q_text = question.get("question", "").lower()
    # check both the type field and the question text so we catch all stylised questions
    return "stylised" in q_type or "shakespearean-style" in q_text


# turn the list of retrieved chunks into a readable string for the csv
def serialise_evidence(retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
    """
    convert a list of (chunk, score) pairs into a formatted string for storing in the csv.

    inputs:
      retrieved - list of (chunk dict, cosine similarity score) pairs

    outputs:
      a multi-line string where each chunk is a numbered block separated by dashes.
    """
    blocks: List[str] = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        blocks.append(
            f"Rank {rank} | cosine similarity={score:.4f}\n"
            f"{format_chunk_for_display(chunk)}"
        )
    # join blocks with a visible separator so each passage is easy to read in the csv
    return "\n\n---\n\n".join(blocks)


# run the baseline (no retrieval) for one question
def run_baseline(question: Dict[str, str], mock: bool = False) -> str:
    """
    generate a baseline answer for one question using no retrieved context.

    inputs:
      question - the question dict with a "question" field
      mock     - if true, return a placeholder string instead of calling the model

    outputs:
      the generated answer as a string, or a placeholder in mock mode.
    """
    q = question["question"]
    if mock:
        return "[MOCK BASELINE OUTPUT: run without --mock to generate with the selected language model.]"

    if is_stylised_question(question):
        # for stylised questions, ask the baseline to produce creative output too
        q = (
            "Write a short creative Shakespearean-style response under 150 words. "
            "Clearly treat it as creative output, not factual evidence. Task: " + q
        )
    return baseline_answer(q)


# run the rag system for one question and return both the answer and the evidence
def run_rag(question: Dict[str, str], retriever: Any, mock: bool = False) -> Tuple[str, str]:
    """
    retrieve passages and generate a rag answer for one question.

    inputs:
      question  - the question dict with a "question" field
      retriever - an embeddingretriever that is already loaded and ready
      mock      - if true, skip the model call but still run retrieval

    outputs:
      a tuple of (generated answer string, serialised evidence string).
    """
    q = question["question"]
    # retrieve the top-k most relevant chunks for this question
    retrieved = retriever.retrieve(q, top_k=DEFAULT_TOP_K)
    # format the retrieved passages into a readable string for the csv
    evidence = serialise_evidence(retrieved)

    if mock:
        return "[MOCK RAG OUTPUT: run without --mock to generate with the selected language model.]", evidence

    if is_stylised_question(question):
        prompt = build_stylised_prompt(q, retrieved)
        # cap stylised responses at 180 tokens so they stay short and creative
        answer = generate_answer(prompt, max_tokens=180)
    else:
        prompt = build_rag_prompt(q, retrieved)
        answer = generate_answer(prompt)
    return answer, evidence


# return a dict with blank score fields ready for manual filling
def empty_score_fields() -> Dict[str, str]:
    """
    return a dict of scoring fields all set to empty strings.

    these fields are left blank on purpose so a human can fill them in
    manually using the scoring rubric after the csv is generated.
    """
    return {
        "correctness_score": "",
        "grounding_score": "",
        "retrieval_relevance_score": "",
        "usefulness_score": "",
        "style_quality_score": "",
        "comments": "",
    }


# run all questions through both systems and save the results to csv
def evaluate(mock: bool = False) -> None:
    """
    run every evaluation question through the baseline and the rag system,
    then write all results to a csv file for manual scoring.

    inputs:
      mock - if true, skips the language model calls and writes placeholder outputs.
             useful for quickly checking the csv structure without waiting for the llm.
    """
    questions = load_questions()
    rows: List[Dict[str, str]] = []

    retriever = None
    if not mock:
        # load or build the embedding index before we start so retrieval is ready
        retriever = setup_retriever()
    else:
        # even in mock mode we try to set up retrieval so we can inspect the retrieved passages
        try:
            retriever = setup_retriever()
        except Exception as error:
            print(f"Warning: retrieval unavailable in mock mode: {error}")

    for question in questions:
        # build the baseline row for this question
        base_row = {
            "question_id": question.get("question_id", question.get("id", "")),
            "source": question.get("source", ""),
            "question_type": question.get("type", ""),
            "play": question.get("play", ""),
            "question": question.get("question", ""),
            "expected_focus": question.get("expected_focus", ""),
            "system": "baseline",
            "retrieved_passages": "N/A - baseline uses no retrieval",
            "generated_response": "",
            **empty_score_fields(),
        }
        try:
            base_row["generated_response"] = run_baseline(question, mock=mock)
        except Exception as error:
            base_row["generated_response"] = f"ERROR while running baseline: {error}"
        rows.append(base_row)

        # build the rag row for the same question
        rag_row = {
            "question_id": question.get("question_id", question.get("id", "")),
            "source": question.get("source", ""),
            "question_type": question.get("type", ""),
            "play": question.get("play", ""),
            "question": question.get("question", ""),
            "expected_focus": question.get("expected_focus", ""),
            "system": "rag",
            "retrieved_passages": "",
            "generated_response": "",
            **empty_score_fields(),
        }
        try:
            if retriever is None:
                raise RuntimeError("Retriever is not available.")
            answer, evidence = run_rag(question, retriever, mock=mock)
            rag_row["retrieved_passages"] = evidence
            rag_row["generated_response"] = answer
        except Exception as error:
            rag_row["generated_response"] = f"ERROR while running RAG: {error}"
        rows.append(rag_row)

    # make sure the results folder exists before writing the output files
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    write_summary_template()
    print(f"Wrote detailed evaluation results to: {OUTPUT_PATH}")
    print(f"Wrote compact summary template to: {SUMMARY_PATH}")
    print("Next step: score each output manually using results/scoring_rubric.md.")


# write the blank summary table that will be filled in after manual scoring
def write_summary_template() -> None:
    """
    create a blank csv template for the per-system average scores.

    this produces a two-row table (one row for baseline, one for rag) with
    empty score columns. after manual scoring is done, those averages can
    be filled in here to produce the summary table for the report.
    """
    summary_fields = [
        "system",
        "avg_correctness",
        "avg_grounding",
        "avg_retrieval_relevance",
        "avg_usefulness",
        "avg_style_quality_stylised_only",
        "overall_comment",
    ]
    rows = [
        {"system": "baseline", "avg_correctness": "", "avg_grounding": "", "avg_retrieval_relevance": "", "avg_usefulness": "", "avg_style_quality_stylised_only": "", "overall_comment": ""},
        {"system": "rag", "avg_correctness": "", "avg_grounding": "", "avg_retrieval_relevance": "", "avg_usefulness": "", "avg_style_quality_stylised_only": "", "overall_comment": ""},
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline and RAG evaluation.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Create output rows without calling the language model. Useful for checking CSV structure.",
    )
    args = parser.parse_args()
    evaluate(mock=args.mock)
