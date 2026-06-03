"""
data_loader.py

loads the three shakespeare play json files and returns structured data.

the real json format is:
{
  "metadata": { ... },
  "scenes": [
    {
      "play": "hamlet",
      "act": 1,               <- integer
      "scene": 1,             <- integer
      "scene_id": "hamlet_1_1",
      "location": "",
      "scene_summary": "guards and horatio see the ghost...",
      "keywords": ["ghost", "horatio"],
      "text": "full scene text as one string",
      "utterances": [
        {
          "speaker": "francisco",
          "text": "you come most carefully upon your hour.",
          "utterance_id": "hamlet_1_1_0001",
          "source_id": "hamlet_1_1_0001",
          ...
        },
        ...
      ]
    },
    ...
  ]
}

we support two loading modes:
  - load_all_scenes()     -> one record per scene (used for scene-level chunking)
  - load_all_utterances() -> one record per utterance (used for utterance comparison)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import PLAY_FILES


# type alias so the code reads like "a list of records" rather than "a list of dicts"
Record = Dict[str, Any]


# open one play json file and return the raw parsed object
def _load_raw(path: Path) -> dict:
    """
    read a single play json file from disk and return the parsed contents.
    raises a clear error if the file is missing so the user knows what to fix.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"dataset file not found: {path}\n"
            "please place the three play json files in data/processed/\n"
            "or update PLAY_FILES in config.py to point to shakespeare_slm_dataset/"
        )
    # load the json file from disk so we can read the scenes inside it
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# pull out the list of scene objects from the raw parsed json
def _extract_scenes(raw: Any, play_key: str) -> List[Record]:
    """
    extract the list of scene dicts from the raw json object.

    inputs:
      raw      - the parsed json, either a dict with a "scenes" key or a plain list
      play_key - short name for the play (e.g. "hamlet") so we can stamp each scene

    outputs:
      a list of scene dicts, each one having fields like play, act, scene, utterances, etc.
    """
    # handle the expected { "metadata": ..., "scenes": [...] } format
    if isinstance(raw, dict) and "scenes" in raw:
        scenes = raw["scenes"]
        if isinstance(scenes, list):
            for s in scenes:
                # stamp the play_key onto each scene so we can always tell which play it came from
                s.setdefault("play_key", play_key)

                # the json stores act and scene as integers (e.g. 1, 2) but we want
                # strings like "act 1" and "scene 2" so they display nicely
                if isinstance(s.get("act"), int):
                    s["act_num"] = s["act"]           # keep the raw integer too, in case we need it
                    s["act"] = f"Act {s['act']}"
                if isinstance(s.get("scene"), int):
                    s["scene_num"] = s["scene"]       # keep the raw integer too
                    s["scene"] = f"Scene {s['scene']}"
            return scenes

    # fallback: if the top level is already a list, treat each item as a scene
    if isinstance(raw, list):
        for s in raw:
            s.setdefault("play_key", play_key)
        return raw

    raise ValueError(
        f"could not extract scenes from {play_key} json. "
        "expected a dict with a 'scenes' key containing a list."
    )


# load every scene from all three plays and return them as one flat list
def load_all_scenes() -> List[Record]:
    """
    load all three plays and return a flat list of scene-level records.

    outputs:
      a list where each item is one scene dict with fields:
        play, act, scene, scene_id, location, scene_summary, keywords,
        text, utterances, play_key
    """
    all_scenes: List[Record] = []

    for play_key, file_path in PLAY_FILES.items():
        raw = _load_raw(file_path)
        scenes = _extract_scenes(raw, play_key)
        # add all scenes from this play into the combined list
        all_scenes.extend(scenes)
        print(f"  loaded {len(scenes)} scenes from {play_key}")

    print(f"\ntotal scenes across all plays: {len(all_scenes)}")
    return all_scenes


# load every individual line of dialogue from all three plays
def load_all_utterances() -> List[Record]:
    """
    load all three plays and return a flat list of utterance-level records.

    each record is one line of dialogue with fields:
      speaker, text, utterance_id, source_id, play, act, scene,
      scene_summary, keywords, play_key

    this is used for the utterance-vs-scene chunking comparison in the report.
    """
    all_utterances: List[Record] = []

    for play_key, file_path in PLAY_FILES.items():
        raw = _load_raw(file_path)
        scenes = _extract_scenes(raw, play_key)

        # each scene has a list of utterances, so we loop through them one by one
        for scene in scenes:
            for utt in scene.get("utterances", []):
                # copy scene-level metadata onto each utterance so every record is self-contained
                utt.setdefault("play_key", play_key)
                utt.setdefault("scene_summary", scene.get("scene_summary", ""))
                utt.setdefault("keywords", scene.get("keywords", []))
                utt.setdefault("act", scene.get("act", "?"))
                utt.setdefault("scene", scene.get("scene", "?"))
                all_utterances.append(utt)

    print(f"total utterances across all plays: {len(all_utterances)}")
    return all_utterances


# keep load_all_plays as an alias so build_index.py and other files still work
def load_all_plays() -> List[Record]:
    """
    alias for load_all_scenes().
    called by build_index.py and dataset_stats.py so those files do not need changing.
    """
    return load_all_scenes()


# -----------------------------------------------------------------
# quick test: run directly to verify loading works
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("loading scene-level records...\n")
    scenes = load_all_scenes()

    print("\nfirst scene preview:")
    s = scenes[0]
    print(f"  play:          {s.get('play')}")
    print(f"  act:           {s.get('act')}")
    print(f"  scene:         {s.get('scene')}")
    print(f"  scene_id:      {s.get('scene_id')}")
    print(f"  scene_summary: {s.get('scene_summary')}")
    print(f"  keywords:      {s.get('keywords')}")
    print(f"  utterances:    {len(s.get('utterances', []))} lines")
    print(f"  text length:   {len(s.get('text', ''))} chars")
