"""
data_loader.py

loads the three shakespeare play json files and returns structured data.

the real json format is:
{
  "metadata": { ... },
  "scenes": [
    {
      "play": "Hamlet",
      "act": 1,               <- integer
      "scene": 1,             <- integer
      "scene_id": "hamlet_1_1",
      "location": "",
      "scene_summary": "Guards and Horatio see the ghost...",
      "keywords": ["ghost", "Horatio"],
      "text": "full scene text as one string",
      "utterances": [
        {
          "speaker": "FRANCISCO",
          "text": "You come most carefully upon your hour.",
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


Record = Dict[str, Any]


def _load_raw(path: Path) -> dict:
    """
    open one play json file and return the raw parsed object.
    raises a clear error if the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"dataset file not found: {path}\n"
            "please place the three play json files in data/processed/\n"
            "or update PLAY_FILES in config.py to point to shakespeare_slm_dataset/"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_scenes(raw: Any, play_key: str) -> List[Record]:
    """
    pull out the list of scene objects from the raw json.

    the json structure is: { "metadata": {...}, "scenes": [...] }
    each element in scenes is one scene dict with fields:
      play, act, scene, scene_id, location, scene_summary, keywords,
      text, utterances
    """
    # handle the expected { "metadata": ..., "scenes": [...] } format
    if isinstance(raw, dict) and "scenes" in raw:
        scenes = raw["scenes"]
        if isinstance(scenes, list):
            # stamp the play_key onto each scene so we can always identify the play
            for s in scenes:
                s.setdefault("play_key", play_key)
                # normalise act and scene to "Act N" / "Scene N" string format
                # because the json stores them as integers
                if isinstance(s.get("act"), int):
                    s["act_num"] = s["act"]
                    s["act"] = f"Act {s['act']}"
                if isinstance(s.get("scene"), int):
                    s["scene_num"] = s["scene"]
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


def load_all_scenes() -> List[Record]:
    """
    load all three plays and return a flat list of scene-level records.
    each record represents one scene and has fields:
      play, act, scene, scene_id, location, scene_summary, keywords,
      text, utterances, play_key
    """
    all_scenes: List[Record] = []

    for play_key, file_path in PLAY_FILES.items():
        raw = _load_raw(file_path)
        scenes = _extract_scenes(raw, play_key)
        all_scenes.extend(scenes)
        print(f"  loaded {len(scenes)} scenes from {play_key}")

    print(f"\ntotal scenes across all plays: {len(all_scenes)}")
    return all_scenes


def load_all_utterances() -> List[Record]:
    """
    load all three plays and return a flat list of utterance-level records.
    each record is one line of dialogue with fields:
      speaker, text, utterance_id, source_id, play, act, scene,
      scene_summary, keywords, play_key
    this is used for the utterance-vs-scene chunking comparison.
    """
    all_utterances: List[Record] = []

    for play_key, file_path in PLAY_FILES.items():
        raw = _load_raw(file_path)
        scenes = _extract_scenes(raw, play_key)

        for scene in scenes:
            for utt in scene.get("utterances", []):
                # stamp scene-level metadata onto each utterance
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
    called by build_index.py and dataset_stats.py.
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
