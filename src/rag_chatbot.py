"""
rag_chatbot.py

this is the main chatbot script that uses retrieval-augmented generation (rag).
it works in three steps for every question:

  step 1 - retrieve:  find the most relevant scene chunks from shakespeare
  step 2 - prompt:    build a prompt that includes those chunks as context
  step 3 - generate:  send the prompt to a language model to get an answer

the language model we use is google/flan-t5-base from huggingface transformers.
it is a small (~250mb) instruction-following model that runs on cpu.
it does not need a gpu, so it works fine on a standard laptop.

usage:
  python src/rag_chatbot.py

type 'stylised' before your question to get a creative shakespearean response.
type 'quit' or 'exit' to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_CACHE_PATH,
    CHUNKS_CACHE_PATH,
    LM_MODEL_NAME,
    LM_MAX_NEW_TOKENS,
    PROMPT_DIR,
)
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


# type alias for a chunk dictionary
Chunk = Dict[str, Any]


# -----------------------------------------------------------------
# language model setup
# -----------------------------------------------------------------

def load_language_model():
    """
    load the flan-t5-base model and its tokenizer from huggingface.
    on first run the model is downloaded (~250mb) and cached locally.
    after the first run it loads from the local cache in a few seconds.
    """
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError as error:
        raise ImportError(
            "transformers is not installed. "
            "install with: pip install transformers"
        ) from error

    print(f"loading language model: {LM_MODEL_NAME}")
    print("(first run downloads ~250mb, then loads from cache)")

    tokenizer = T5Tokenizer.from_pretrained(LM_MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(LM_MODEL_NAME)
    model.eval()   # set to evaluation mode so we do not accidentally update weights

    print("language model loaded.\n")
    return tokenizer, model


def generate_answer(prompt: str, tokenizer, model) -> str:
    """
    send the prompt to flan-t5-base and return the generated answer.
    the model reads the prompt and produces a response conditioned on it.
    """
    import torch

    # tokenize the prompt (convert text to token ids the model understands)
    # truncation=True cuts very long prompts to fit the model's input limit
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # generate a response with no gradient tracking (faster, less memory)
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=LM_MAX_NEW_TOKENS,
            num_beams=2,          # beam search for slightly better quality
            early_stopping=True
        )

    # decode the output token ids back to a human-readable string
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()


# -----------------------------------------------------------------
# prompt building
# -----------------------------------------------------------------

def load_system_prompt() -> str:
    """
    read the system prompt from the prompts folder.
    this tells the model how to behave as a shakespeare assistant.
    """
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    if not prompt_path.exists():
        # fallback prompt if the file is missing
        return (
            "You are a Shakespeare-aware assistant. "
            "Use the retrieved context to answer the question. "
            "Be beginner-friendly and do not invent unsupported details."
        )
    return prompt_path.read_text(encoding="utf-8").strip()


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    assemble the full prompt for the rag system.
    the prompt contains:
      1. the system instruction (how to behave)
      2. the retrieved context passages
      3. the user's question
    """
    system_prompt = load_system_prompt()

    # format each retrieved chunk as a numbered context block
    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        block = (
            f"[Context {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )
        context_blocks.append(block)

    combined_context = "\n\n".join(context_blocks)

    # put it all together in a clear structure flan-t5 can follow
    prompt = (
        f"{system_prompt}\n\n"
        f"Retrieved context:\n{combined_context}\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )
    return prompt


def build_stylised_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    build a prompt for stylised shakespearean generation.
    loads the stylised generation prompt template.
    """
    template_path = PROMPT_DIR / "stylised_generation_prompt.txt"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = (
            "Write a short creative response in Shakespearean English "
            "(thee, thou, dost) based on the context below. "
            "Label it as creative output. Keep it under 150 words.\n\n"
            "Context: {context}\n\nPrompt: {query}\n\n"
            "[CREATIVE OUTPUT - not a direct quote]:"
        )

    # format the context passages for the template
    context_text = "\n\n".join(
        format_chunk_for_display(chunk) for chunk, _ in retrieved
    )

    prompt = template.replace("{context}", context_text).replace("{query}", query)
    return prompt


# -----------------------------------------------------------------
# retriever setup
# -----------------------------------------------------------------

def setup_retriever() -> EmbeddingRetriever:
    """
    set up the retriever by either loading cached embeddings from disk
    or building the index fresh from the play files.
    loading from cache is much faster so we prefer it.
    """
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)

    # try to load pre-built embeddings from disk first
    loaded = retriever.load_from_disk(EMBEDDINGS_CACHE_PATH, CHUNKS_CACHE_PATH)

    if not loaded:
        # if no cache exists, build the index from scratch
        print("no cached index found. building from scratch...")
        records = load_all_plays()
        chunks  = create_chunks(records)
        retriever.build_index(chunks)
        retriever.save_to_disk(EMBEDDINGS_CACHE_PATH, CHUNKS_CACHE_PATH)

    return retriever


# -----------------------------------------------------------------
# main chatbot loop
# -----------------------------------------------------------------

def run_chatbot(retriever: EmbeddingRetriever, tokenizer, model) -> None:
    """
    the interactive question-answering loop.
    the user types a question and sees both the retrieved evidence
    and the generated answer.
    """
    print("\n" + "=" * 60)
    print("shakespeare-aware rag chatbot")
    print("=" * 60)
    print("covers: hamlet, macbeth, romeo and juliet")
    print("type 'stylised: <question>' for a creative shakespearean response")
    print("type 'quit' or 'exit' to stop\n")

    while True:
        # read the user's question
        raw_input = input("your question: ").strip()

        # check for exit commands
        if raw_input.lower() in {"quit", "exit", "q"}:
            print("goodbye.")
            break

        # check if this is a stylised generation request
        is_stylised = raw_input.lower().startswith("stylised:")
        if is_stylised:
            query = raw_input[len("stylised:"):].strip()
        else:
            query = raw_input

        if not query:
            print("please enter a question.\n")
            continue

        # step 1: retrieve the most relevant scene chunks
        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

        # step 2: build the right kind of prompt
        if is_stylised:
            prompt = build_stylised_prompt(query, retrieved)
        else:
            prompt = build_rag_prompt(query, retrieved)

        # step 3: generate an answer from the language model
        print("\ngenerating answer...\n")
        answer = generate_answer(prompt, tokenizer, model)

        # --- display the retrieved evidence ---
        print("-" * 60)
        print("retrieved evidence:")
        print("-" * 60)
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print(f"\nrank {rank} | cosine similarity: {score:.4f}")
            print(format_chunk_for_display(chunk))

        # --- display the generated answer ---
        print("\n" + "-" * 60)
        if is_stylised:
            print("[creative shakespearean output - not factual evidence]:")
        else:
            print("answer:")
        print("-" * 60)
        print(answer)
        print("\n")


# -----------------------------------------------------------------
# entry point
# -----------------------------------------------------------------

def main() -> None:
    """
    set up the retriever and language model then start the chatbot.
    """
    retriever            = setup_retriever()
    tokenizer, lm_model  = load_language_model()
    run_chatbot(retriever, tokenizer, lm_model)


if __name__ == "__main__":
    main()
