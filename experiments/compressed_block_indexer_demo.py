"""Compressed block-index benchmark for CSA-shaped sparse context reads."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.compressed_block_indexer import (
    run_csa_hca_block_state_sweep,
    run_csa_hca_policy_trial,
    run_csa_hca_rare_directory_policy_sweep,
    run_csa_hca_rare_directory_stress_sweep,
    run_csa_hca_rare_directory_sweep,
    run_compressed_block_budget_sweep,
    run_compressed_block_index_trial,
    run_hca_decay_quality_sweep,
    run_hca_lazy_metadata_sweep,
    run_hca_summary_quality_sweep,
)
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
    print()

    sweep = run_compressed_block_budget_sweep(summary_width=256, tail_blocks=2)
    print("Repeated sparse block-read budget")
    print(
        "fixed index: "
        f"state={format_bytes(sweep.summary_state_bytes)}, "
        f"score_rd={format_bytes(sweep.score_bytes_per_query)}/query, "
        f"relevant={fmt_pct(sweep.relevant_query_rate)}"
    )
    headers = [
        "select",
        "avg_blk",
        "hit",
        "hot_hit",
        "cold_hit",
        "coverage",
        "oracle",
        "gap",
        "kv_read",
        "reduct",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 120)
    for point in sweep.points:
        row = [
            f"{point.selected_blocks}",
            f"{point.avg_blocks_read:0.1f}",
            fmt_pct(point.block_hit_rate),
            fmt_pct(point.hot_block_hit_rate),
            fmt_pct(point.cold_block_hit_rate),
            fmt_pct(point.occurrence_coverage),
            fmt_pct(point.oracle_occurrence_coverage),
            fmt_pct(point.oracle_coverage_gap),
            f"{point.token_reads_per_query:0.0f}",
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Budget interpretation:")
    print("- Coverage rises slowly because hot tokens are spread across many blocks.")
    print("- The tiny oracle gap says the compressed score nearly matches exact top-block choice.")
    print("- This supports repeated sparse reads or a separate dense summary for high-frequency tokens.")
    print()

    policy = run_csa_hca_policy_trial(summary_width=256, global_width=2048)
    print("Low-bit CSA/HCA routing policy")
    print(
        "fixed state: "
        f"blocks={format_bytes(policy.block_summary_state_bytes)}, "
        f"global={format_bytes(policy.global_summary_state_bytes)}, "
        f"global_rd={format_bytes(policy.global_summary_read_bytes_per_query)}/query"
    )
    headers = [
        "thresh",
        "hca_q",
        "csa_q",
        "hot_hca",
        "cold_csa",
        "csa_hit",
        "csa_cov",
        "sparse_cv",
        "score_rd",
        "kv_read",
        "reduct",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 132)
    for point in policy.points:
        row = [
            f"{point.hca_threshold}",
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.csa_query_rate),
            fmt_pct(point.hot_to_hca_rate),
            fmt_pct(point.cold_to_csa_rate),
            fmt_pct(point.csa_relevant_hit_rate),
            fmt_pct(point.csa_relevant_coverage),
            fmt_pct(point.policy_sparse_coverage),
            format_bytes(point.block_score_bytes_per_query),
            f"{point.token_reads_per_query:0.0f}",
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Policy interpretation:")
    print("- A tiny global summary can skip block scoring for frequent HCA-path queries.")
    print("- Low thresholds save reads but may send rare/cold queries to the dense path.")
    print("- sparse_cv is low by design when hot queries are delegated to the HCA summary.")
    print("- This is the first explicit CA-side policy knob between CSA and HCA paths.")
    print()

    state = run_csa_hca_block_state_sweep(global_width=2048, hca_threshold=8)
    print("CSA/HCA block-summary state sweep")
    print(
        f"fixed HCA: width={state.global_width}, "
        f"state={format_bytes(state.global_summary_state_bytes)}, "
        f"threshold={state.hca_threshold}, "
        f"global_rd={format_bytes(state.global_summary_read_bytes_per_query)}/query"
    )
    headers = [
        "block",
        "blocks",
        "width",
        "csa",
        "state",
        "score_rd",
        "hca_q",
        "cold_csa",
        "csa_hit",
        "csa_cov",
        "kv_read",
        "reduct",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 144)
    for point in state.points:
        row = [
            f"{point.block_size}",
            f"{point.blocks}",
            f"{point.summary_width}",
            f"{point.csa_blocks}",
            format_bytes(point.block_summary_state_bytes),
            format_bytes(point.block_score_bytes_per_query),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.cold_to_csa_rate),
            fmt_pct(point.csa_relevant_hit_rate),
            fmt_pct(point.csa_relevant_coverage),
            f"{point.token_reads_per_query:0.1f}",
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("State-sweep interpretation:")
    print("- block=128,width=256 halves CSA state to 256KB while preserving measured CSA hit/coverage.")
    print("- block=64,width=128 also uses 256KB but weakens cold-query reliability.")
    print("- block=256,width=256 reaches 128KB but loses too much cold exact recall in this trial.")
    print()

    rare = run_csa_hca_rare_directory_sweep(global_width=2048, hca_threshold=15)
    print("Rare-token block directory repair")
    print(
        f"fixed HCA: width={rare.global_width}, "
        f"threshold={rare.hca_threshold}, "
        f"global_rd={format_bytes(rare.global_summary_read_bytes_per_query)}/query"
    )
    headers = [
        "block",
        "width",
        "dir_k",
        "blk_state",
        "dir_state",
        "combined",
        "base_hit",
        "rep_hit",
        "repair",
        "base_cov",
        "rep_cov",
        "dir_rd",
        "kv_read",
        "reduct",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 168)
    for point in rare.points:
        row = [
            f"{point.block_size}",
            f"{point.summary_width}",
            f"{point.directory_blocks_per_token}",
            format_bytes(point.block_summary_state_bytes),
            format_bytes(point.directory_state_bytes),
            format_bytes(point.block_plus_directory_state_bytes),
            fmt_pct(point.base_csa_relevant_hit_rate),
            fmt_pct(point.repaired_csa_relevant_hit_rate),
            fmt_pct(point.directory_repair_rate),
            fmt_pct(point.base_csa_relevant_coverage),
            fmt_pct(point.repaired_csa_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_reads_per_query:0.1f}",
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Rare-directory interpretation:")
    print("- A small exact directory repairs low-width CSA misses for rare/cold tokens.")
    print("- block=128,width=128,dir_k=6 uses about 159KB and restores measured coverage.")
    print("- This is the cleaner CA split: HCA for frequent context, exact directory for rare block ids.")
    print()

    stress = run_csa_hca_rare_directory_stress_sweep(global_width=2048, hca_threshold=15)
    print("Rare-directory stress sweep")
    print(
        f"block={stress.block_size}, width={stress.summary_width}, "
        f"hca_threshold={stress.hca_threshold}, "
        f"block_state={format_bytes(stress.block_summary_state_bytes)}"
    )
    headers = [
        "scenario",
        "dir_k",
        "dir_state",
        "false_hca",
        "hit",
        "coverage",
        "csa_cov",
        "dir_rd",
        "kv_read",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 158)
    for point in stress.points:
        row = [
            point.scenario,
            f"{point.directory_blocks_per_token}",
            format_bytes(point.directory_state_bytes),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_hit_rate),
            fmt_pct(point.repaired_relevant_coverage),
            fmt_pct(point.repaired_csa_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_reads_per_query:0.1f}",
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Stress interpretation:")
    print("- Raising the HCA gate to threshold 15 prevents most rare false-HCA routes.")
    print("- dir_k=2 handles burst/split rare tokens but not names spread across many blocks.")
    print("- dir_k=6 is the current stress-safe directory setting for repeated rare names.")
    print()

    print("Rare-directory guard comparison")
    guard_runs = [
        ("t8_no_guard", run_csa_hca_rare_directory_stress_sweep(
            global_width=2048,
            hca_threshold=8,
            directory_guard=False,
            directory_blocks=(6,),
            queries=2048,
        )),
        ("t8_guard", run_csa_hca_rare_directory_stress_sweep(
            global_width=2048,
            hca_threshold=8,
            directory_guard=True,
            directory_blocks=(6,),
            queries=2048,
        )),
        ("t15_no_guard", run_csa_hca_rare_directory_stress_sweep(
            global_width=2048,
            hca_threshold=15,
            directory_guard=False,
            directory_blocks=(6,),
            queries=2048,
        )),
    ]
    headers = ["policy", "scenario", "false_hca", "coverage", "dir_rd", "kv_read", "reduct"]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 112)
    for label, result in guard_runs:
        for point in result.points:
            if point.scenario not in ("zipf_reference", "repeated_name"):
                continue
            row = [
                label,
                point.scenario,
                fmt_pct(point.rare_false_hca_rate),
                fmt_pct(point.repaired_relevant_coverage),
                format_bytes(point.directory_read_bytes_per_query),
                f"{point.token_reads_per_query:0.1f}",
                f"{point.token_read_reduction:0.1f}x",
            ]
            print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Guard interpretation:")
    print("- directory_guard makes the exact rare directory an admission override before HCA.")
    print("- t8_guard removes rare false-HCA routes but adds one directory probe per query.")
    print("- t15_no_guard is cheaper on average; t8_guard is the more conservative exact-recall mode.")
    print()

    policy = run_csa_hca_rare_directory_policy_sweep()
    print("Rare-directory admission/fanout policy sweep")
    headers = [
        "policy",
        "scenario",
        "thr",
        "guard",
        "stored",
        "read",
        "false_hca",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 154)
    for point in policy.points:
        if point.scenario not in ("zipf_reference", "repeated_name"):
            continue
        row = [
            point.policy,
            point.scenario,
            f"{point.hca_threshold}",
            "yes" if point.directory_guard else "no",
            f"{point.directory_blocks_per_token}",
            f"{point.directory_read_blocks_per_token}",
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Policy interpretation:")
    print("- Storing 6 directory blocks but reading only 2 saves reads and loses repeated-name coverage.")
    print("- Metadata-driven fanout should read more blocks only for spread-out rare tokens.")
    print("- The next trainable policy should choose threshold, guard, and fanout per token.")
    print()

    quality = run_hca_summary_quality_sweep(threshold=8)
    print("HCA-like global summary quality")
    print(
        f"threshold={quality.threshold}, "
        f"context={quality.context_length}, "
        f"vocab={quality.vocab_size}, "
        f"queries={quality.queries}"
    )
    headers = [
        "width",
        "state",
        "sat",
        "mae",
        "top64",
        "top256",
        "prec",
        "recall",
        "q_acc",
        "false_hca",
        "miss_hca",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 132)
    for point in quality.points:
        row = [
            f"{point.global_width}",
            format_bytes(point.state_bytes),
            fmt_pct(point.saturation_rate),
            f"{point.clipped_mean_abs_error:0.3f}",
            fmt_pct(point.top64_recall),
            fmt_pct(point.top256_recall),
            fmt_pct(point.threshold_precision),
            fmt_pct(point.threshold_recall),
            fmt_pct(point.query_route_accuracy),
            fmt_pct(point.query_false_hca_rate),
            fmt_pct(point.query_missed_hca_rate),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("HCA interpretation:")
    print("- The threshold classifier is the immediate requirement for CSA/HCA routing.")
    print("- top-k recall measures whether the global summary preserves dense topic mass.")
    print("- Saturation shows when 4-bit counters are losing frequency detail.")
    print()

    decay = run_hca_decay_quality_sweep(global_width=2048, threshold=2)
    print("HCA anti-saturation decay sweep")
    print(
        f"fixed summary: width={decay.global_width}, "
        f"state={format_bytes(decay.points[0].state_bytes)}, "
        f"threshold={decay.threshold}"
    )
    headers = [
        "decay",
        "dec_cell",
        "sat",
        "mae",
        "top64",
        "top256",
        "prec",
        "recall",
        "q_acc",
        "false_hca",
        "miss_hca",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 132)
    for point in decay.points:
        label = "none" if point.decay_interval > decay.context_length else str(point.decay_interval)
        row = [
            label,
            f"{point.avg_decay_cells_per_token:0.1f}",
            fmt_pct(point.saturation_rate),
            f"{point.clipped_mean_abs_error:0.3f}",
            fmt_pct(point.top64_recall),
            fmt_pct(point.top256_recall),
            fmt_pct(point.threshold_precision),
            fmt_pct(point.threshold_recall),
            fmt_pct(point.query_route_accuracy),
            fmt_pct(point.query_false_hca_rate),
            fmt_pct(point.query_missed_hca_rate),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Decay interpretation:")
    print("- Decay fixes saturation and recovers top-k dense-topic order for this stream.")
    print("- Decayed state needs its own lower threshold or learned scale metadata.")
    print("- The next HCA state should make decay and threshold/scale trainable rather than fixed.")
    print()

    lazy = run_hca_lazy_metadata_sweep(global_width=2048, threshold=2)
    print("Lazy HCA decay with per-counter epoch metadata")
    headers = [
        "epoch",
        "decay",
        "state",
        "read",
        "saved",
        "top64",
        "top256",
        "q_acc",
        "false_hca",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 132)
    for point in lazy.points:
        row = [
            f"{point.epoch_bits}",
            f"{point.decay_interval}",
            format_bytes(point.state_bytes),
            format_bytes(point.read_bytes_per_query),
            f"{point.explicit_decay_cells_per_token:0.1f}",
            fmt_pct(point.top64_recall),
            fmt_pct(point.top256_recall),
            fmt_pct(point.query_route_accuracy),
            fmt_pct(point.query_false_hca_rate),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Lazy interpretation:")
    print("- 8-bit epochs are enough for 65k context at decay >= 256 in this trial.")
    print("- 4-bit epochs require longer decay intervals and lose some dense-topic quality.")
    print("- This is closer to a chip-realistic HCA state than full-array periodic decay.")


if __name__ == "__main__":
    main()
