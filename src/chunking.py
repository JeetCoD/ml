"""
chunking.py

converts scene-level records into retrieval chunks.

the real dataset structure (after data_loader.py processes it):
  each scene has:
    - play, act, scene, scene_id, location
    - scene_summary   -> plain english summary of the scene
    - keywords        -> list of topic words
    - text            -> full scene text as one string
    - utterances      -> list of { speaker, text, utterance_id, ... }

chunking strategy chosen: scene-level chunking
-----------------------------------------------
we use the full scene as one retrieval chunk.
the scene_summary and keywords are prepended to the dialogue so the
embedding model captures the topic even when the question uses modern
words that do not appear in the elizabethan dialogue.

the chunk text is structured like this:
  [summary: guards and horatio see the ghost and decide to tell hamlet.]
  [keywords: ghost, horatio]
  francisco: you come most carefully upon your hour.
  bernardo: 'tis now struck twelve. get thee to bed, francisco.
  ...

why scene-level and not utterance-level?
  - a single utterance like "to be, or not to be" has almost no
    context on its own. the retriever cannot match it to a question
    like "what is hamlet thinking about in his soliloquy?"
  - a full scene chunk gives the retriever narrative context, character
    names, and topic words all in one vector.
  - the scene_summary in plain english also helps bridge the gap between
    a modern question and old elizabethan vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict, List


# type aliases so function signatures read more clearly
Chunk = Dict[str, Any]
Record = Dict[str, Any]


# convert a list of scene records into one chunk per scene
def create_chunks(scenes: List[Record]) -> List[Chunk]:
    """
    turn a list of scene records into retrieval chunks, one chunk per scene.

    inputs:
      scenes - list of scene dicts from data_loader.py

    outputs:
      list of chunk dicts, each combining the summary, keywords, and dialogue into one text field
    """
    chunks: List[Chunk] = []

    for scene in scenes:
        # build the combined text for this chunk by gathering parts in order
        parts = []

        # combine summary and keywords first because they are in plain english
        # and help the embedding model understand the topic before reading elizabethan dialogue
        summary = scene.get("scene_summary", "").strip()
        if summary:
            parts.append(f"[summary: {summary}]")

        # prepend keywords for the same reason - they act as topic anchors for the embedding
        keywords = scene.get("keywords", [])
        if keywords:
            # join keyword list into a comma-separated string so it is readable in the chunk text
            kw_string = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
            parts.append(f"[keywords: {kw_string}]")

        # add each line of dialogue as "speaker: text" so the model sees who says what
        for utt in scene.get("utterances", []):
            speaker  = utt.get("speaker", "").strip()
            utt_text = utt.get("text", "").strip()
            if utt_text:
                if speaker and speaker != "STAGE_DIRECTION":
                    parts.append(f"{speaker}: {utt_text}")
                elif speaker == "STAGE_DIRECTION":
                    # include stage directions in brackets so the context is clear
                    parts.append(f"[stage direction: {utt_text}]")
                else:
                    parts.append(utt_text)

        # join all parts with newlines to form the final chunk text
        combined_text = "\n".join(parts).strip()

        # skip any scene that ended up with no usable text, since empty chunks are useless
        if not combined_text:
            continue

        # use scene_id as the stable identifier, or build one from act/scene if it is missing
        chunk_id = scene.get("scene_id") or (
            f"{scene.get('play_key','unknown')}_"
            f"{scene.get('act','?')}_"
            f"{scene.get('scene','?')}"
        ).lower().replace(" ", "_")

        chunk: Chunk = {
            "chunk_id":      chunk_id,
            "play":          scene.get("play", scene.get("play_key", "unknown")),
            "act":           scene.get("act", "?"),
            "scene":         scene.get("scene", "?"),
            "scene_id":      scene.get("scene_id", chunk_id),
            "location":      scene.get("location", ""),
            "scene_summary": summary,
            "keywords":      keywords,
            "speaker":       None,            # scene chunks span many speakers so we leave this as none
            "text":          combined_text,
            "num_utterances": len(scene.get("utterances", [])),
        }
        chunks.append(chunk)

    return chunks


# alternative: one utterance = one chunk (used only for comparison, not the main pipeline)
def create_utterance_chunks(scenes: List[Record]) -> List[Chunk]:
    """
    split scenes into individual utterances and return one chunk per line of dialogue.

    this is an alternative chunking strategy used only for the scene-vs-utterance
    comparison in the report. it is not used in the main rag pipeline.

    inputs:
      scenes - list of scene dicts from data_loader.py

    outputs:
      list of chunk dicts, each containing one utterance and its scene metadata
    """
    chunks: List[Chunk] = []

    for scene in scenes:
        for utt in scene.get("utterances", []):
            text = utt.get("text", "").strip()
            # skip empty utterances since they carry no information
            if not text:
                continue

            chunk: Chunk = {
                "chunk_id":      utt.get("utterance_id", utt.get("source_id", "")),
                "play":          scene.get("play", scene.get("play_key", "unknown")),
                "act":           scene.get("act", "?"),
                "scene":         scene.get("scene", "?"),
                "scene_id":      scene.get("scene_id", ""),
                "location":      scene.get("location", ""),
                "scene_summary": scene.get("scene_summary", ""),
                "keywords":      scene.get("keywords", []),
                "speaker":       utt.get("speaker", ""),
                "text":          text,
                "num_utterances": 1,   # each utterance chunk contains exactly one line
            }
            chunks.append(chunk)

    return chunks


# format a single retrieved chunk into a readable string for display
def format_chunk_for_display(chunk: Chunk) -> str:
    """
    turn a chunk dict into a human-readable string showing the source and a text preview.

    inputs:
      chunk - a chunk dict with play, act, scene, speaker, and text fields

    outputs:
      a formatted string like "[hamlet, act 1, scene 1]\n<preview of text>"
    """
    play    = chunk.get("play", "unknown play")
    act     = chunk.get("act", "?")
    scene   = chunk.get("scene", "?")
    speaker = chunk.get("speaker", "")

    # build the header line that shows where this chunk comes from
    header = f"{play}, {act}, {scene}"
    if speaker:
        header += f", Speaker: {speaker}"

    # show up to 500 chars so the display does not become too long
    text    = chunk.get("text", "")
    preview = text[:500] + "..." if len(text) > 500 else text

    return f"[{header}]\n{preview}"


# -----------------------------------------------------------------
# quick test: run directly to verify chunking works
# -----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_all_scenes

    print("loading scenes...")
    scenes = load_all_scenes()

    print("\ncreating scene-level chunks...")
    chunks = create_chunks(scenes)
    print(f"created {len(chunks)} scene-level chunks from {len(scenes)} scenes\n")

    print("creating utterance-level chunks (for comparison)...")
    utt_chunks = create_utterance_chunks(scenes)
    print(f"created {len(utt_chunks)} utterance-level chunks\n")

    print("--- comparison ---")
    print(f"scene chunks:     {len(chunks):4d}  (one chunk per scene)")
    print(f"utterance chunks: {len(utt_chunks):4d}  (one chunk per line of dialogue)")
    print(f"\nscene chunks are {len(utt_chunks)//max(len(chunks),1)}x fewer but each is much richer")
    print(f"\nfirst scene chunk preview:")
    print(format_chunk_for_display(chunks[0]))
