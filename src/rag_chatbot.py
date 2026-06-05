"""
this is the main chatbot script that uses retrieval-augmented generation (rag).
it works in three steps for every question:

  step 1 - retrieve:  find the most relevant scene chunks from shakespeare
  step 2 - prompt:    build a prompt that includes those chunks as context
  step 3 - generate:  send the prompt to the language model to get an answer

the language model we use is qwen2.5:1.5b, a 1.5 billion parameter small
language model served locally by ollama. it runs fully on cpu with no api key
and no internet connection needed at inference time.

usage:
  python src/rag_chatbot.py

type 'stylised: <question>' before your question to get a creative shakespearean response.
type 'quit' or 'exit' to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# add src/ to path so imports work whether we run from root or from src/
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


# type alias for a chunk dictionary so function signatures read clearly
Chunk = Dict[str, Any]


# -----------------------------------------------------------------
# language model setup
# -----------------------------------------------------------------

# check that ollama is running and the model is available before starting the chat loop
def check_language_model() -> None:
    """
    verify that ollama is running and the chosen model can respond.

    ollama runs as a background service outside python, so we do not load
    the model into memory. instead we send a tiny test message to confirm
    we can reach the service and the model is downloaded.
    """
    try:
        import ollama
    except ImportError as error:
        raise ImportError(
            "ollama python client is not installed. "
            "install with: pip3 install ollama"
        ) from error

    print(f"checking language model: {LM_MODEL_NAME}")

    try:
        # send a tiny test message so we know ollama is reachable and the model is loaded
        ollama.chat(
            model=LM_MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 5},
        )
    except Exception as error:
        raise RuntimeError(
            f"could not reach ollama or load model {LM_MODEL_NAME}. "
            f"make sure the ollama app is running and you have pulled "
            f"the model with: ollama pull {LM_MODEL_NAME}.\n"
            f"original error: {error}"
        ) from error

    print("language model is ready.\n")


# send the assembled prompt to the language model and return its response
def generate_answer(prompt: str, max_tokens: int = None) -> str:
    """
    send a prompt to qwen2.5:1.5b via ollama and return the generated answer.

    inputs:
      prompt     - the full prompt string including context and the user question
      max_tokens - optional override for how many tokens to generate

    we use a low temperature so the model stays closer to the retrieved context
    and is less likely to make up details that are not in the passages.
    """
    import ollama
    from config import LM_TEMPERATURE

    response = ollama.chat(
        model=LM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            # use the passed max_tokens if given, otherwise fall back to the config default
            "num_predict": max_tokens if max_tokens else LM_MAX_NEW_TOKENS,
            "temperature": LM_TEMPERATURE,
        },
    )

    # the response is a nested dict, so we dig into it to get the text string
    answer = response["message"]["content"].strip()
    return answer


# -----------------------------------------------------------------
# prompt building
# -----------------------------------------------------------------

# load the system prompt that tells the model how to behave
def load_system_prompt() -> str:
    """
    read the system prompt file from the prompts folder.

    outputs:
      the prompt text as a string, or a hardcoded fallback if the file is missing.

    the system prompt tells the model to act as a shakespeare assistant and
    to base its answers on the retrieved context rather than inventing things.
    """
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    if not prompt_path.exists():
        # use a safe fallback so the chatbot still works even without the file
        return (
            "You are a Shakespeare-aware assistant. "
            "Use the retrieved context to answer the question. "
            "Be beginner-friendly and do not invent unsupported details."
        )
    return prompt_path.read_text(encoding="utf-8").strip()


# assemble the full rag prompt from the system instruction, context, and question
def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    build the complete prompt for the rag system.

    inputs:
      query     - the user's question as a plain string
      retrieved - list of (chunk, score) pairs returned by the retriever

    outputs:
      a single prompt string that contains the system instruction, the retrieved
      context passages, and the user question so the model can answer grounded in evidence.
    """
    system_prompt = load_system_prompt()

    # format each retrieved chunk as a numbered block so the model knows which passage is which
    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        block = (
            f"[Context {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )
        context_blocks.append(block)

    # join all context blocks with a blank line between them for readability
    combined_context = "\n\n".join(context_blocks)

    # put it all together in a clear structure the model can follow
    prompt = (
        f"{system_prompt}\n\n"
        f"Retrieved context:\n{combined_context}\n\n"
        f"User question: {query}\n\n"
        f"Answer:"
    )
    return prompt


# build a prompt for creative shakespearean-style generation
def build_stylised_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    build a prompt that asks the model for a creative shakespearean-style response.

    inputs:
      query     - the user's prompt or question
      retrieved - list of (chunk, score) pairs used as inspiration context

    loads the stylised generation template from the prompts folder if available,
    otherwise uses a hardcoded fallback. the {context} and {query} placeholders
    in the template are replaced with the actual values.
    """
    template_path = PROMPT_DIR / "stylised_generation_prompt.txt"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        # fallback template if the file is missing
        template = (
            "Write a short creative response in Shakespearean English "
            "(thee, thou, dost) based on the context below. "
            "Label it as creative output. Keep it under 150 words.\n\n"
            "Context: {context}\n\nPrompt: {query}\n\n"
            "[CREATIVE OUTPUT - not a direct quote]:"
        )

    # join all retrieved chunk displays into one block to use as context
    context_text = "\n\n".join(
        format_chunk_for_display(chunk) for chunk, _ in retrieved
    )

    # replace the placeholders in the template with the real values
    prompt = template.replace("{context}", context_text).replace("{query}", query)
    return prompt


# -----------------------------------------------------------------
# retriever setup
# -----------------------------------------------------------------

# load or build the embedding index so we can do retrieval
def setup_retriever() -> EmbeddingRetriever:
    """
    prepare the retriever by loading cached embeddings if they exist,
    or building the index from scratch if they do not.

    outputs:
      an embeddingretriever that is ready to answer queries.

    loading from cache is much faster (seconds vs minutes), so we always
    try that first and only rebuild if the cache files are missing.
    """
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)

    # try to load the pre-built embeddings from disk first to save time
    loaded = retriever.load_from_disk(EMBEDDINGS_CACHE_PATH, CHUNKS_CACHE_PATH)

    if not loaded:
        # no cache found, so we build the index from the raw play files
        print("no cached index found. building from scratch...")
        records = load_all_plays()
        chunks  = create_chunks(records)
        retriever.build_index(chunks)
        # save to disk so future startups can skip this slow step
        retriever.save_to_disk(EMBEDDINGS_CACHE_PATH, CHUNKS_CACHE_PATH)

    return retriever


# -----------------------------------------------------------------
# main chatbot loop
# -----------------------------------------------------------------

# run the interactive question-answering loop until the user quits
def run_chatbot(retriever: EmbeddingRetriever) -> None:
    """
    start the interactive chat loop where the user can ask questions.

    the loop reads a question, retrieves relevant passages, builds a prompt,
    sends it to the language model, then displays the evidence and the answer.
    it keeps running until the user types 'quit' or 'exit'.
    """
    print("\n" + "=" * 60)
    print("shakespeare-aware rag chatbot")
    print("=" * 60)
    print("covers: hamlet, macbeth, romeo and juliet")
    print("type 'stylised: <question>' for a creative shakespearean response")
    print("type 'quit' or 'exit' to stop\n")

    while True:
        # read whatever the user typed
        raw_input = input("your question: ").strip()

        # check for exit commands first so we can break out cleanly
        if raw_input.lower() in {"quit", "exit", "q"}:
            print("goodbye.")
            break

        # check if the user wants a creative stylised response instead of a factual one
        is_stylised = raw_input.lower().startswith("stylised:")
        if is_stylised:
            # strip the "stylised:" prefix to get the actual question
            query = raw_input[len("stylised:"):].strip()
        else:
            query = raw_input

        if not query:
            print("please enter a question.\n")
            continue

        # step 1: retrieve the most relevant scene chunks for this query
        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

        # step 2: build the appropriate prompt depending on whether it is stylised or factual
        if is_stylised:
            prompt = build_stylised_prompt(query, retrieved)
        else:
            prompt = build_rag_prompt(query, retrieved)

        # step 3: send the prompt to the language model and get the answer back
        print("\ngenerating answer...\n")
        # stylised responses are capped at 180 tokens so they stay short and poetic
        answer = generate_answer(prompt, max_tokens=180 if is_stylised else None)

        # show the retrieved evidence so the user can see what the answer is based on
        print("-" * 60)
        print("retrieved evidence:")
        print("-" * 60)
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print(f"\nrank {rank} | cosine similarity: {score:.4f}")
            print(format_chunk_for_display(chunk))

        # then show the generated answer below the evidence
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

# set up all components and start the chatbot
def main() -> None:
    """
    load the retriever and check the language model, then start the chat loop.
    """
    retriever = setup_retriever()
    check_language_model()
    run_chatbot(retriever)


if __name__ == "__main__":
    main()
