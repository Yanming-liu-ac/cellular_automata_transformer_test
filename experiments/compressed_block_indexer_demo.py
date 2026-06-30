"""Compressed block-index benchmark for CSA-shaped sparse context reads."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.compressed_block_indexer import run_compressed_block_index_trial
from cellular_transformer.hardware import format_bytes


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def main() -> None:
    print("Compressed block indexer")
    print("context=65536, block=64, blocks=1024, 4-bit summaries, topic/noise stream")
    print()
    headers = [
        "width",
        "select",
        "tail",
        "state",
        "rel_q",
        "idx_hit",
        "hot_idx",
        "cold_idx",
        "idx_cov",
        "combo_hit",
        "cold_cmb",
        "combo_cov",
        "oracle_cv",
        "score_rd",
        "kv_read",
        "reduct",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 174)
    trials = [
        run_compressed_block_index_trial(summary_width=128, selected_blocks=4, tail_blocks=2),
        run_compressed_block_index_trial(summary_width=256, selected_blocks=4, tail_blocks=2),
        run_compressed_block_index_trial(summary_width=256, selected_blocks=8, tail_blocks=2),
        run_compressed_block_index_trial(summary_width=512, selected_blocks=8, tail_blocks=2),
        run_compressed_block_index_trial(summary_width=512, selected_blocks=16, tail_blocks=2),
    ]
    for result in trials:
        row = [
            f"{result.summary_width}",
            f"{result.selected_blocks}",
            f"{result.tail_blocks}",
            format_bytes(result.summary_state_bytes),
            fmt_pct(result.relevant_query_rate),
            fmt_pct(result.index_block_hit_rate),
            fmt_pct(result.hot_index_block_hit_rate),
            fmt_pct(result.cold_index_block_hit_rate),
            fmt_pct(result.index_occurrence_coverage),
            fmt_pct(result.combined_block_hit_rate),
            fmt_pct(result.cold_combined_block_hit_rate),
            fmt_pct(result.combined_occurrence_coverage),
            fmt_pct(result.oracle_occurrence_coverage),
            format_bytes(result.score_bytes_per_query),
            f"{result.combined_token_reads_per_query:0.0f}",
            f"{result.combined_token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- Each block is a CA cell with a low-bit compressed local summary.")
    print("- idx_hit measures whether top-k compressed blocks contain the query token.")
    print("- hot/cold columns expose the rare-token weakness that exact memory must cover.")
    print("- combo adds a short exact tail window, matching a CSA plus local-context path.")
    print("- score_rd is summary traffic; kv_read is selected token blocks, not full context.")


if __name__ == "__main__":
    main()
