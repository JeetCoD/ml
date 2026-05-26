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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PLAY_FILES, RESULTS_DIR
from data_loader import load_all_scenes
from chunking import create_chunks, create_utterance_chunks


def compute_stats_for_play(play_key: str, scenes: list) -> dict:
    """
    compute statistics for one play given its list of scene records.
    """
    def normalise(name: str) -> str:
        return name.lower().replace(" ", "_")

    # filter scenes belonging to this play only
    play_scenes = [
        s for s in scenes
        if normalise(s.get("play", s.get("play_key", ""))) == normalise(play_key)
    ]

    if not play_scenes:
        return {}

    # count unique acts and scenes
    unique_acts   = len(set(s.get("act", "?") for s in play_scenes))
    unique_scenes = len(play_scenes)

    # count unique speakers across all utterances in this play
    all_speakers = set()
    total_utterances = 0
    for scene in play_scenes:
        for utt in scene.get("utterances", []):
            spk = utt.get("speaker", "").strip()
            if spk and spk not in ("STAGE_DIRECTION", ""):
                all_speakers.add(spk)
        total_utterances += len(scene.get("utterances", []))

    # build scene-level chunks for this play and measure length
    scene_chunks = create_chunks(play_scenes)
    chunk_lengths = [len(c.get("text", "")) for c in scene_chunks]
    avg_chars  = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    avg_tokens = avg_chars / 4   # rough estimate: 4 chars per token

    # utterance-level chunks for comparison count
    utt_chunks = create_utterance_chunks(play_scenes)

    # top 5 keywords across all scenes
    all_keywords = []
    for scene in play_scenes:
        kws = scene.get("keywords", [])
        if isinstance(kws, list):
            all_keywords.extend(kws)
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


def save_stats_csv(stats_list: list, output_path: Path) -> None:
    df = pd.DataFrame(stats_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"saved statistics csv to: {output_path}")


def save_chunk_count_chart(stats_list: list, output_path: Path) -> None:
    """
    bar chart comparing scene-level vs utterance-level chunk counts per play.
    """
    labels        = [s["play"].replace("_", "\n") for s in stats_list]
    scene_counts  = [s["scene_chunks"] for s in stats_list]
    utt_counts    = [s["utterance_chunks"] for s in stats_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    x   = range(len(labels))
    w   = 0.35

    b1 = ax.bar([p - w/2 for p in x], utt_counts,  width=w, label="utterance chunks", color="#4C72B0", alpha=0.85)
    b2 = ax.bar([p + w/2 for p in x], scene_counts, width=w, label="scene-level chunks", color="#55A868", alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("number of chunks", fontsize=11)
    ax.set_title("utterance-level vs scene-level chunks per play", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved bar chart to: {output_path}")


def main() -> None:
    print("computing dataset statistics...\n")

    all_scenes = load_all_scenes()

    all_stats = []
    for play_key in PLAY_FILES.keys():
        stats = compute_stats_for_play(play_key, all_scenes)
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
