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
    run_csa_hca_rare_directory_adaptive_policy_sweep,
    run_csa_hca_rare_directory_joint_policy_sweep,
    run_csa_hca_rare_directory_joint_threshold_sweep,
    run_csa_hca_rare_directory_learned_fanout_sweep,
    run_csa_hca_rare_directory_aware_route_lut_sweep,
    run_csa_hca_rare_directory_bloom_bank_sweep,
    run_csa_hca_rare_directory_bloom_retirement_collision_fanout_sweep,
    run_csa_hca_rare_directory_bloom_retirement_collision_sweep,
    run_csa_hca_rare_directory_bloom_retirement_compression_sweep,
    run_csa_hca_rare_directory_bloom_retirement_sweep,
    run_csa_hca_rare_directory_bloom_sidecar_sweep,
    run_csa_hca_rare_directory_bloom_salt_selection_sweep,
    run_csa_hca_rare_directory_bloom_salt_sweep,
    run_csa_hca_rare_directory_bloom_streaming_update_sweep,
    run_csa_hca_rare_directory_policy_sweep,
    run_csa_hca_rare_directory_presence_sidecar_sweep,
    run_csa_hca_rare_directory_route_lut_sweep,
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
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
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

    adaptive = run_csa_hca_rare_directory_adaptive_policy_sweep()
    print("Metadata-driven rare-directory fanout sweep")
    headers = [
        "policy",
        "scenario",
        "thr",
        "guard",
        "base",
        "expand",
        "span",
        "avg_rd",
        "exp_r",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 188)
    for point in adaptive.points:
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            point.policy,
            point.scenario,
            f"{point.hca_threshold}",
            "yes" if point.directory_guard else "no",
            f"{point.base_read_blocks_per_token}",
            f"{point.expanded_read_blocks_per_token}",
            f"{point.spread_threshold_blocks}",
            f"{point.avg_directory_read_blocks_per_hit:0.2f}",
            fmt_pct(point.expanded_read_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Adaptive fanout interpretation:")
    print("- The span2toN rules are hardware-style metadata proxies, not oracles.")
    print("- It keeps base fanout at 2 and expands only when stored rare blocks are widely spread.")
    print("- This is the first concrete target for replacing the hand fanout with a trained LUT.")
    print()

    learned_fanout = run_csa_hca_rare_directory_learned_fanout_sweep()
    print("Learned low-bit rare-directory fanout LUT")
    print(
        f"training_samples={learned_fanout.training_samples}, "
        f"target={fmt_pct(learned_fanout.coverage_target)}, "
        f"lut_state={format_bytes(learned_fanout.lut.state_bytes)}, "
        f"metadata_features=entry_count/span/CSA-overlap"
    )
    headers = [
        "scenario",
        "false_hca",
        "coverage",
        "avg_rd",
        "exp_r",
        "dir_rd",
        "meta",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 126)
    for point in learned_fanout.points:
        if point.scenario not in ("zipf_reference", "rare_burst", "split_rare", "repeated_name"):
            continue
        row = [
            point.scenario,
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            f"{point.avg_directory_read_blocks_per_hit:0.2f}",
            fmt_pct(point.expanded_read_rate),
            format_bytes(point.directory_read_bytes_per_query),
            format_bytes(point.fanout_metadata_state_bytes),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Learned fanout interpretation:")
    print("- The LUT is trained from self-supervised coverage labels and uses no token identity.")
    print("- CSA-overlap lets it spend fewer directory reads than the hand span2to5 rule.")
    print("- This is the first trainable control-plane block for the rare exact-memory lane.")
    print()

    print("Threshold-15 normal fanout guard sweep")
    headers = [
        "min_read",
        "zfloor",
        "scenario",
        "coverage",
        "avg_rd",
        "exp_r",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 122)
    for min_read, zero_overlap_read_floor in ((2, 0), (2, 3), (3, 0)):
        guard_fanout = run_csa_hca_rare_directory_learned_fanout_sweep(
            hca_threshold=15,
            min_read_blocks_per_token=min_read,
            zero_overlap_read_floor=zero_overlap_read_floor,
            coverage_target=0.95,
        )
        for point in guard_fanout.points:
            if point.scenario not in ("zipf_reference", "rare_burst", "split_rare", "repeated_name"):
                continue
            row = [
                str(min_read),
                str(zero_overlap_read_floor),
                point.scenario,
                fmt_pct(point.repaired_relevant_coverage),
                f"{point.avg_directory_read_blocks_per_hit:0.2f}",
                fmt_pct(point.expanded_read_rate),
                format_bytes(point.directory_read_bytes_per_query),
                f"{point.token_read_reduction:0.1f}x",
            ]
            print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Threshold-15 fanout guard interpretation:")
    print("- The zero-overlap zfloor=3 guard leaves zipf_reference, rare_burst, and repeated_name traffic unchanged.")
    print("- It restores 100% split coverage with 6.53B/query, much lower than the global min_read=3 point.")
    print("- This is the better hardware guard candidate because it triggers only when CSA misses all directory entries.")
    print()

    joint = run_csa_hca_rare_directory_joint_policy_sweep()
    print("Joint HCA-confidence probe and fanout control")
    print(
        f"fanout_lut={format_bytes(joint.lut.state_bytes)}, "
        f"probe_lut={format_bytes(joint.probe_lut.state_bytes)}, "
        f"probe_positive_rate_threshold={fmt_pct(joint.probe_positive_rate_threshold)}"
    )
    headers = [
        "policy",
        "scenario",
        "probe_r",
        "hca_r",
        "false_hca",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 126)
    for point in joint.points:
        if point.policy not in ("confidence_probe", "hca_probe", "always_probe"):
            continue
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            point.policy,
            point.scenario,
            fmt_pct(point.directory_probe_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Joint-control interpretation:")
    print("- HCA bank confidence can suppress directory probes for strong hot tokens.")
    print("- confidence_probe keeps reference traffic near never_probe while retaining rare recall.")
    print("- The remaining false-HCA cases are now an explicit probe-LUT recall/traffic tradeoff.")
    print()

    threshold = run_csa_hca_rare_directory_joint_threshold_sweep()
    print("Joint-control HCA threshold sweep")
    headers = [
        "thr",
        "scenario",
        "probe_r",
        "hca_r",
        "false_hca",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 126)
    for point in threshold.points:
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            f"{point.hca_threshold}",
            point.scenario,
            fmt_pct(point.directory_probe_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Threshold interpretation:")
    print("- Threshold 6 is too permissive for split rare tokens in this stress set.")
    print("- Thresholds 8-15 keep similar rare coverage after joint probe/fanout control.")
    print("- Higher thresholds reduce early probes, making threshold 15 the cheaper exact-recall mode here.")
    print()

    route = run_csa_hca_rare_directory_route_lut_sweep()
    print("Trained HCA route LUT sweep")
    print(
        f"training_samples={route.training_samples}, "
        f"route_lut={format_bytes(route.route_lut.state_bytes)}, "
        f"active_route_buckets={sum(route.route_lut.routes_hca)}"
    )
    headers = [
        "scenario",
        "hca_r",
        "false_hca",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 94)
    for point in route.points:
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            point.scenario,
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Route-LUT interpretation:")
    print("- A 40B route LUT can replace the hand HCA threshold in this diagnostic.")
    print("- It preserves reference HCA routing but is not yet better than threshold15+fanout.")
    print("- The next route table needs richer metadata or a recall-weighted training objective.")
    print()

    aware_route = run_csa_hca_rare_directory_aware_route_lut_sweep()
    print("Directory-aware HCA route LUT sweep")
    print(
        f"training_samples={aware_route.training_samples}, "
        f"route_lut={format_bytes(aware_route.route_lut.state_bytes)}, "
        f"presence_read={format_bytes(aware_route.route_feature_read_bytes)}/query, "
        f"active_route_buckets={sum(aware_route.route_lut.routes_hca)}"
    )
    headers = [
        "scenario",
        "hca_r",
        "false_hca",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 94)
    for point in aware_route.points:
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            point.scenario,
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Directory-aware route interpretation:")
    print("- One rare-directory presence bit is enough to suppress the remaining false-HCA cases.")
    print("- The 80B route table keeps the reference HCA path while exposing a CA-local metadata tradeoff.")
    print("- The next hardware question is whether this sidecar is a real 1-bit SRAM/Bloom read.")
    print()

    sidecar = run_csa_hca_rare_directory_presence_sidecar_sweep()
    print("Presence-sidecar false-positive sweep")
    print(
        f"training_samples={sidecar.training_samples}, "
        f"route_lut={format_bytes(sidecar.route_lut.state_bytes)}, "
        f"presence_read={format_bytes(sidecar.route_feature_read_bytes)}/query"
    )
    headers = [
        "fp",
        "scenario",
        "sidecar",
        "fp_q",
        "hca_r",
        "coverage",
        "dir_rd",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 126)
    for point in sidecar.points:
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            fmt_pct(point.false_positive_rate),
            point.scenario,
            format_bytes(point.sidecar_state_bytes),
            fmt_pct(point.sidecar_false_positive_query_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.repaired_relevant_coverage),
            format_bytes(point.directory_read_bytes_per_query),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Presence-sidecar interpretation:")
    print("- Exact sidecar behavior is the upper bound; Bloom false positives trade state for HCA hot-path loss.")
    print("- Rare recall stays safe because false positives route extra queries to CSA instead of HCA.")
    print("- The hardware target is the largest false-positive rate that keeps reference HCA routing high enough.")
    print()

    bloom = run_csa_hca_rare_directory_bloom_sidecar_sweep()
    print("Physical Bloom presence-sidecar sweep")
    print(
        f"training_samples={bloom.training_samples}, "
        f"route_lut={format_bytes(bloom.route_lut.state_bytes)}, "
        f"banks={bloom.bank_count}"
    )
    headers = [
        "bpe",
        "k",
        "scenario",
        "sidecar",
        "rd",
        "fp_q",
        "hca_r",
        "coverage",
        "q_conf",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 150)
    for point in bloom.points:
        show_reference = point.scenario == "zipf_reference"
        show_stress = (
            point.bits_per_entry == 8
            and point.hash_count == 3
            and point.scenario in ("split_rare", "repeated_name")
        )
        if not (show_reference or show_stress):
            continue
        row = [
            f"{point.bits_per_entry}",
            f"{point.hash_count}",
            point.scenario,
            format_bytes(point.sidecar_state_bytes),
            format_bytes(point.read_bytes_per_query),
            fmt_pct(point.sidecar_false_positive_query_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.repaired_relevant_coverage),
            fmt_pct(point.query_bank_conflict_rate),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Physical Bloom interpretation:")
    print("- The 8 bits/entry, k=3 point keeps reference HCA routing near the ideal sidecar.")
    print("- More hash reads cut false positives but raise read traffic and bank conflicts.")
    print("- This turns the presence sidecar into an explicit SRAM/read-port design knob.")
    print()

    salt = run_csa_hca_rare_directory_bloom_salt_sweep()
    print("Bloom sidecar hash-salt robustness sweep")
    print(
        f"salt_count={salt.salt_count}, "
        f"bits_per_entry={salt.bits_per_entry}, "
        f"k={salt.hash_count}, "
        f"banks={salt.bank_count}"
    )
    mean_hca = sum(point.hca_query_rate for point in salt.points) / len(salt.points)
    mean_hot_fp = sum(point.hot_sidecar_false_positive_rate for point in salt.points) / len(salt.points)
    print(
        f"mean_hca={fmt_pct(mean_hca)}, "
        f"mean_hot_fp={fmt_pct(mean_hot_fp)}"
    )
    headers = [
        "rank",
        "salt",
        "fp_q",
        "hot_fp",
        "hca_r",
        "q_conf",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 110)
    ranked = sorted(salt.points, key=lambda point: point.hca_query_rate)
    selected_salts = ranked[:3] + ranked[-3:]
    for rank, point in enumerate(selected_salts, start=1):
        label = f"worst{rank}" if rank <= 3 else f"best{rank - 3}"
        row = [
            label,
            f"{point.sidecar_salt}",
            fmt_pct(point.sidecar_false_positive_query_rate),
            fmt_pct(point.hot_sidecar_false_positive_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.query_bank_conflict_rate),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Bloom salt interpretation:")
    print("- Hash choice changes hot-token false positives enough to matter for HCA efficiency.")
    print("- The sidecar should choose or learn hash salts against hot-token queries, not only global FPR.")
    print("- The next layout test should compare modulo banking with hash-based bank assignment.")
    print()

    bank = run_csa_hca_rare_directory_bloom_bank_sweep()
    print("Bloom sidecar bank-mapping sweep")
    print(
        f"salt_count={bank.salt_count}, "
        f"bits_per_entry={bank.bits_per_entry}, "
        f"k={bank.hash_count}, "
        f"banks={bank.bank_count}"
    )
    headers = [
        "mode",
        "mean_hca",
        "mean_hotfp",
        "mean_conf",
        "worst_hca",
        "best_hca",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 94)
    for mode in bank.bank_modes:
        mode_points = [point for point in bank.points if point.bank_mode == mode]
        mean_hca = sum(point.hca_query_rate for point in mode_points) / len(mode_points)
        mean_hot_fp = (
            sum(point.hot_sidecar_false_positive_rate for point in mode_points) / len(mode_points)
        )
        mean_conflict = (
            sum(point.query_bank_conflict_rate for point in mode_points) / len(mode_points)
        )
        worst = min(mode_points, key=lambda point: point.hca_query_rate)
        best = max(mode_points, key=lambda point: point.hca_query_rate)
        row = [
            mode,
            fmt_pct(mean_hca),
            fmt_pct(mean_hot_fp),
            fmt_pct(mean_conflict),
            fmt_pct(worst.hca_query_rate),
            fmt_pct(best.hca_query_rate),
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Bloom bank interpretation:")
    print("- Bank mapping does not change Bloom false positives, but it changes read-port pressure.")
    print("- by_hash assigns each hash function to its own bank and removes same-query bank conflicts here.")
    print("- This is the first clean SRAM-layout win for the sidecar path.")
    print()

    selected_salt = run_csa_hca_rare_directory_bloom_salt_selection_sweep()
    print("Bloom sidecar salt-selection sweep")
    print(
        f"selected_index={selected_salt.selected_salt_index}, "
        f"selected_salt={selected_salt.selected_sidecar_salt}, "
        f"metric={selected_salt.selection_metric}"
    )
    headers = [
        "scenario",
        "fp_q",
        "hot_fp",
        "hca_r",
        "coverage",
        "q_conf",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 110)
    for point in selected_salt.eval_points:
        if point.scenario not in ("zipf_reference", "split_rare", "repeated_name"):
            continue
        row = [
            point.scenario,
            fmt_pct(point.sidecar_false_positive_query_rate),
            fmt_pct(point.hot_sidecar_false_positive_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.repaired_relevant_coverage),
            fmt_pct(point.query_bank_conflict_rate),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Bloom salt-selection interpretation:")
    print("- Selecting salt against hot-token false positives recovers most of the ideal HCA hot path.")
    print("- by_hash keeps bank conflicts at zero while salt selection controls false positives.")
    print("- The next step is update scheduling for the selected sidecar under streaming inserts.")
    print()

    streaming = run_csa_hca_rare_directory_bloom_streaming_update_sweep()
    print("Bloom sidecar streaming-update sweep")
    print(
        f"salt={streaming.sidecar_salt}, "
        f"bpe={streaming.bits_per_entry}, "
        f"k={streaming.hash_count}, "
        f"banks={streaming.bank_count}, "
        f"bank_mode={streaming.bank_mode}"
    )
    headers = [
        "scenario",
        "policy",
        "insert",
        "rare_in",
        "hot_poll",
        "upd/tok",
        "bank/tok",
        "hot_fp",
        "hca_r",
        "coverage",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 160)
    for point in streaming.points:
        if point.scenario != "zipf_reference" and point.policy not in (
            "final_oracle",
            "count1",
            "count8",
            "count14",
        ):
            continue
        row = [
            point.scenario,
            point.policy,
            str(point.inserted_tokens),
            fmt_pct(point.inserted_final_rare_rate),
            fmt_pct(point.hot_polluted_token_rate),
            f"{point.update_bytes_per_context_token:0.5f}",
            f"{point.max_bank_update_bytes_per_context_token:0.5f}",
            fmt_pct(point.hot_sidecar_false_positive_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.repaired_relevant_coverage),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Bloom streaming-update interpretation:")
    print("- final_oracle is an upper bound that inserts only final rare-directory tokens.")
    print("- Naive count thresholds insert future hot tokens before they are known hot.")
    print("- That pollution collapses the reference HCA fast path, so the sidecar needs delayed promotion or deletion.")
    print()

    retirement = run_csa_hca_rare_directory_bloom_retirement_sweep()
    print("Counting Bloom hot-retirement sweep")
    print(
        f"salt={retirement.sidecar_salt}, "
        f"bpe={retirement.bits_per_entry}, "
        f"k={retirement.hash_count}, "
        f"counter_bits={retirement.counter_bits}, "
        f"retire={retirement.retire_count_threshold}"
    )
    headers = [
        "scenario",
        "policy",
        "state",
        "insert",
        "active",
        "delete",
        "active_rare",
        "vis_rare",
        "hot_ret",
        "hot_poll",
        "upd/tok",
        "hca_r",
        "coverage",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 184)
    for point in retirement.points:
        if point.scenario != "zipf_reference" and point.policy not in (
            "count1_retire15",
            "count2_retire15",
            "count8_retire15",
            "count14_retire15",
        ):
            continue
        row = [
            point.scenario,
            point.policy,
            f"{point.sidecar_state_bytes / 1024:0.1f}KB",
            str(point.inserted_tokens),
            str(point.active_tokens),
            str(point.deleted_tokens),
            fmt_pct(point.active_final_rare_rate),
            fmt_pct(point.visible_active_rare_rate),
            fmt_pct(point.hot_retired_token_rate),
            fmt_pct(point.hot_polluted_token_rate),
            f"{point.update_bytes_per_context_token:0.5f}",
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.repaired_relevant_coverage),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Counting Bloom interpretation:")
    print("- Hot-token retirement fixes the irreversible pollution found by naive streaming insertion.")
    print("- count1_retire15 matches the oracle sidecar shape but spends about 5x sidecar state.")
    print("- Later insert thresholds reduce update traffic but become a recall-risk knob for rarer tokens.")
    print()

    compression = run_csa_hca_rare_directory_bloom_retirement_compression_sweep()
    print("Counting Bloom compression sweep")
    print(
        f"salt={compression.sidecar_salt}, "
        f"k={compression.hash_count}, "
        f"retire={compression.retire_count_threshold}, "
        f"bank_mode={compression.bank_mode}"
    )
    headers = [
        "scenario",
        "policy",
        "bpe",
        "cbits",
        "state",
        "vis_rare",
        "hot_poll",
        "fp_q",
        "hca_r",
        "coverage",
        "upd/tok",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 160)
    for point in compression.points:
        show_count1 = point.policy == "count1_retire15" and (
            point.scenario == "zipf_reference"
            or (point.bits_per_entry == 8 and point.counter_bits in (1, 2, 4))
        )
        show_count2 = (
            point.policy == "count2_retire15"
            and point.scenario == "zipf_reference"
            and point.bits_per_entry == 8
            and point.counter_bits in (2, 4)
        )
        if not (show_count1 or show_count2):
            continue
        row = [
            point.scenario,
            point.policy,
            str(point.bits_per_entry),
            str(point.counter_bits),
            f"{point.sidecar_state_bytes / 1024:0.1f}KB",
            fmt_pct(point.visible_active_rare_rate),
            fmt_pct(point.hot_polluted_token_rate),
            fmt_pct(point.sidecar_false_positive_query_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.repaired_relevant_coverage),
            f"{point.update_bytes_per_context_token:0.5f}",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Counting Bloom compression interpretation:")
    print("- 8 bits/entry with 2-bit counters keeps visible rare coverage at 100% in this sweep.")
    print("- 1-bit counters cut more SRAM, but they introduce small rare-token visibility loss.")
    print("- The current hardware budget should test c2 against adversarial collision before treating it as robust.")
    print()

    collision = run_csa_hca_rare_directory_bloom_retirement_collision_sweep()
    print("Counting Bloom adversarial-collision sweep")
    print(
        f"rare_tokens={collision.rare_token_count}, "
        f"salt={collision.sidecar_salt}, "
        f"k={collision.hash_count}, "
        f"hca_threshold={collision.hca_threshold}"
    )
    headers = [
        "bpe",
        "cbits",
        "rare_occ",
        "coll/rare",
        "state",
        "collide",
        "miss",
        "overlap",
        "vis_rare",
        "hot_poll",
        "hca_r",
        "false_hca",
        "coverage",
        "reduct",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 176)
    for point in collision.points:
        if point.bits_per_entry != 8:
            continue
        row = [
            str(point.bits_per_entry),
            str(point.counter_bits),
            str(point.rare_occurrences_per_token),
            str(point.colliders_per_rare),
            f"{point.sidecar_state_bytes / 1024:0.1f}KB",
            str(point.collider_tokens),
            str(point.missing_colliders),
            f"{point.mean_slot_overlap:0.2f}",
            fmt_pct(point.visible_active_rare_rate),
            fmt_pct(point.hot_polluted_token_rate),
            fmt_pct(point.hca_query_rate),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    print()
    print("Counting Bloom collision interpretation:")
    print("- 1-bit counters are not robust: adversarial hot deletes wipe rare-token visibility.")
    print("- 2-bit counters improve normal streams but lose visibility under repeated-key multi-collider deletes.")
    print("- 3-bit counters survive the repeated-key 8-collider stress and are the current sidecar budget target.")
    print("- Any remaining repeated-key coverage loss is now a fanout/directory target, not a sidecar deletion target.")
    print()

    fanout_collision = run_csa_hca_rare_directory_bloom_retirement_collision_fanout_sweep(
        min_read_blocks_per_token_values=(2, 3),
        zero_overlap_read_floor_values=(0, 3),
        coverage_targets=(0.95,),
    )
    print("Repeated-key collision fanout budget")
    print(
        f"rare_occ={fanout_collision.rare_occurrences_per_token}, "
        f"colliders/rare={fanout_collision.colliders_per_rare}, "
        f"counter_bits={fanout_collision.counter_bits}, "
        f"dir_k={fanout_collision.directory_blocks_per_token}"
    )
    headers = [
        "min_read",
        "zfloor",
        "target",
        "lut",
        "train",
        "dir_ent/q",
        "dir_B/q",
        "vis_rare",
        "false_hca",
        "coverage",
        "reduct",
    ]
    print(" | ".join(f"{header:>12}" for header in headers))
    print("-" * 148)
    for point in fanout_collision.points:
        row = [
            str(point.min_read_blocks_per_token),
            str(point.zero_overlap_read_floor),
            fmt_pct(point.coverage_target),
            format_bytes(point.fanout_lut_state_bytes),
            str(point.fanout_training_samples),
            f"{point.directory_entries_read_per_query:0.2f}",
            f"{point.directory_read_bytes_per_query:0.2f}",
            fmt_pct(point.visible_active_rare_rate),
            fmt_pct(point.rare_false_hca_rate),
            fmt_pct(point.repaired_relevant_coverage),
            f"{point.token_read_reduction:0.1f}x",
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Repeated-key fanout interpretation:")
    print("- Raising the global minimum directory read from 2 to 3 fixes coverage but over-reads.")
    print("- A zero-overlap zfloor=3 guard also restores 100% coverage with lower directory traffic.")
    print("- The next robust target is the selective zero-overlap guard, not a global min_read=3 rule.")
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
