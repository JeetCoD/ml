"""
dataset_stats.py

computes statistics about the shakespeare dataset and saves them to disk.
produces a csv and a bar chart for section ii of the report.

outputs:
  results/dataset_statistics.csv
  results/chunk_counts_chart.png
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib
# use the non-interactive agg backend so matplotlib does not try to open a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PLAY_FILES, RESULTS_DIR
from data_loader import load_all_scenes
from chunking import create_chunks, create_utterance_chunks


# compute all statistics for a single play given its scene records
def compute_stats_for_play(play_key: str, scenes: list) -> dict:
    """
    calculate counts and averages for one play from its list of scene records.

    inputs:
      play_key - short name for the play (e.g. "hamlet") used to filter the scenes
      scenes   - the full list of scenes from all plays (we filter to this play inside)

    outputs:
      a dict with fields like total_scenes, total_utterances, scene_chunks, avg_chunk_tokens, etc.
      returns an empty dict if no scenes are found for this play.
    """
    # normalise helper so "romeo and juliet" and "romeo_and_juliet" both match
    def normalise(name: str) -> str:
        return name.lower().replace(" ", "_")

    # keep only the scenes that belong to this play
    play_scenes = [
        s for s in scenes
        if normalise(s.get("play", s.get("play_key", ""))) == normalise(play_key)
    ]

    if not play_scenes:
        return {}

    # count unique acts and scenes so we know the play structure
    unique_acts   = len(set(s.get("act", "?") for s in play_scenes))
    unique_scenes = len(play_scenes)  # one record per scene, so len gives us the count

    # count unique speakers and total utterances across all scenes in this play
    all_speakers = set()       # set so each speaker is only counted once
    total_utterances = 0
    for scene in play_scenes:
        for utt in scene.get("utterances", []):
            spk = utt.get("speaker", "").strip()
            # skip stage directions because they are not real speakers
            if spk and spk not in ("STAGE_DIRECTION", ""):
                all_speakers.add(spk)
        total_utterances += len(scene.get("utterances", []))

    # build scene-level chunks for this play and measure their text lengths
    scene_chunks = create_chunks(play_scenes)
    chunk_lengths = [len(c.get("text", "")) for c in scene_chunks]
    avg_chars  = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    # rough token estimate: divide characters by 4 since most tokens are about 4 chars
    avg_tokens = avg_chars / 4

    # also build utterance-level chunks just to report the comparison count
    utt_chunks = create_utterance_chunks(play_scenes)

    # collect all keywords across scenes and find the 5 most common ones
    all_keywords = []
    for scene in play_scenes:
        kws = scene.get("keywords", [])
        if isinstance(kws, list):
            all_keywords.extend(kws)
    # counter.most_common(5) returns [(word, count), ...] so we take just the words
    top_kws = [kw for kw, _ in Counter(all_keywords).most_common(5)]

    return {
        "play":                play_key,
        "total_scenes":        unique_scenes,
        "total_acts":          unique_acts,
        "total_utterances":    total_utterances,
        "unique_speakers":     len(all_speakers),
        "scene_chunks":        len(scene_chunks),
        "utterance_chunks":    len(utt_chunks),
        "avg_chunk_chars":     round(avg_chars, 1),
        "avg_chunk_tokens":    round(avg_tokens, 1),
        "top_keywords":        ", ".join(top_kws),
    }


# save the list of stats dicts to a csv file using pandas
def save_stats_csv(stats_list: list, output_path: Path) -> None:
    """
    write the statistics list to a csv file on disk.

    inputs:
      stats_list  - list of stat dicts, one per play
      output_path - path where the csv file should be written

    we use pandas here because it handles column alignment and csv encoding cleanly.
    """
    df = pd.DataFrame(stats_list)
    # create the parent folder if it does not exist yet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"saved statistics csv to: {output_path}")


# draw and save a grouped bar chart comparing scene vs utterance chunk counts
def save_chunk_count_chart(stats_list: list, output_path: Path) -> None:
    """
    produce a bar chart comparing scene-level and utterance-level chunk counts for each play.

    inputs:
      stats_list  - list of stat dicts, each must have "play", "scene_chunks", "utterance_chunks"
      output_path - path where the png file should be written

    the chart shows side-by-side bars for each play so it is easy to compare
    how many chunks each strategy produces.
    """
    # replace underscores with newlines in play names so they fit on the x-axis
    labels        = [s["play"].replace("_", "\n") for s in stats_list]
    scene_counts  = [s["scene_chunks"] for s in stats_list]
    utt_counts    = [s["utterance_chunks"] for s in stats_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    x   = range(len(labels))
    w   = 0.35   # width of each bar so both bars fit side by side

    # draw utterance bars on the left and scene bars on the right of each group
    b1 = ax.bar([p - w/2 for p in x], utt_counts,  width=w, label="utterance chunks", color="#4C72B0", alpha=0.85)
    b2 = ax.bar([p + w/2 for p in x], scene_counts, width=w, label="scene-level chunks", color="#55A868", alpha=0.85)

    # add the count as a number on top of each bar so values are easy to read
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("number of chunks", fontsize=11)
    ax.set_title("utterance-level vs scene-level chunks per play", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    # remove the top and right border lines so the chart looks clean
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved bar chart to: {output_path}")


# compute stats for all three plays and save the outputs
def main() -> None:
    """
    run the full stats pipeline: load scenes, compute stats per play, save csv and chart.
    """
    print("computing dataset statistics...\n")

    # load all scenes from the three plays so we can filter by play inside compute_stats_for_play
    all_scenes = load_all_scenes()

    all_stats = []
    for play_key in PLAY_FILES.keys():
        stats = compute_stats_for_play(play_key, all_scenes)
        # skip a play if we got an empty dict back (means no scenes were found for it)
        if stats:
            all_stats.append(stats)

    print("\n--- dataset statistics summary ---")
    for s in all_stats:
        print(
            f"  {s['play']:20s}  "
            f"scenes={s['total_scenes']:3d}  "
            f"utterances={s['total_utterances']:4d}  "
            f"speakers={s['unique_speakers']:3d}  "
            f"scene_chunks={s['scene_chunks']:3d}  "
            f"avg_tokens={s['avg_chunk_tokens']:.0f}"
        )

    save_stats_csv(all_stats, RESULTS_DIR / "dataset_statistics.csv")
    save_chunk_count_chart(all_stats, RESULTS_DIR / "chunk_counts_chart.png")
    print("\ndone.")


if __name__ == "__main__":
    main()
