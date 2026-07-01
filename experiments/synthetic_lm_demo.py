"""Synthetic next-token benchmark for the HARC-CA dual-path prototype."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMConfig,
    run_synthetic_lm_demand_gate_sweep,
    run_synthetic_lm_trial,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def fmt_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024:
            return f"{size:6.2f} {unit}"
        size /= 1024
    return f"{size:6.2f} TB"


def main() -> None:
    configs = [
        ("static", SyntheticLMConfig(dense_width=1024)),
        ("static", SyntheticLMConfig(dense_width=2048)),
        ("static_ph", SyntheticLMConfig(dense_width=2048, candidate_score_source="topic_phase")),
        ("static_sum", SyntheticLMConfig(dense_width=2048, candidate_score_source="dense_topic_sum")),
        ("static", SyntheticLMConfig(dense_width=4096)),
        ("online", SyntheticLMConfig(dense_width=2048, candidate_strategy="online_cache")),
        (
            "online_ph",
            SyntheticLMConfig(
                dense_width=2048,
                candidate_strategy="online_cache",
                candidate_score_source="topic_phase",
            ),
        ),
        (
            "online_tc",
            SyntheticLMConfig(
                dense_width=2048,
                candidate_strategy="online_cache",
                candidate_score_source="topic_cache",
            ),
        ),
        (
            "gated",
            SyntheticLMConfig(
                dense_width=2048,
                candidate_strategy="online_cache",
                candidate_admission_threshold=1,
            ),
        ),
        (
            "gated_ph",
            SyntheticLMConfig(
                dense_width=2048,
                candidate_strategy="online_cache",
                candidate_admission_threshold=1,
                candidate_score_source="topic_phase",
            ),
        ),
        (
            "gated_tc",
            SyntheticLMConfig(
                dense_width=2048,
                candidate_strategy="online_cache",
                candidate_admission_threshold=1,
                candidate_score_source="topic_cache",
            ),
        ),
    ]

    headers = [
        "candidate",
        "dense_w",
        "induct",
        "topic@k",
        "exact_vis",
        "overflow_q",
        "dense_upd",
        "cand_upd",
        "gate_upd",
        "score_upd",
        "score_wr",
        "admit_r",
        "cand_hit",
        "scorer",
        "score_src",
        "avg_cells",
        "memory",
    ]
    print("HARC-CA synthetic next-token benchmark")
    print("exact task: key -> next value; dense task: topic token in top-k candidate shortlist")
    print(" | ".join(f"{h:>11}" for h in headers))
    print("-" * 112)
    for label, config in configs:
        result = run_synthetic_lm_trial(seed=31, config=config)
        row = [
            label,
            f"{config.dense_width}",
            fmt_pct(result.induction_accuracy),
            fmt_pct(result.topic_topk_hit_rate),
            f"{result.exact_avg_visited_cells:0.1f}",
            fmt_pct(result.overflow_query_rate),
            f"{result.dense_update_cells_per_event:0.1f}",
            f"{result.candidate_update_cells_per_event:0.1f}",
            f"{result.candidate_gate_cells_per_event:0.1f}",
            f"{result.candidate_score_cells_per_event:0.1f}",
            f"{result.candidate_score_update_cells_per_event:0.1f}",
            fmt_pct(result.candidate_admission_rate)
            if result.candidate_strategy != "static"
            else "-",
            fmt_pct(result.candidate_cache_hit_rate)
            if result.candidate_strategy != "static"
            else "-",
            result.candidate_scorer_mode,
            result.candidate_score_source,
            f"{result.avg_cells_per_event:0.1f}",
            fmt_bytes(result.total_memory_bytes),
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))

    print()
    print("Interpretation:")
    print("- Induction uses the exact sparse associative lane.")
    print("- Static Topic@k uses an oracle-built candidate pool.")
    print("- Online/gated variants generate candidates from the low-bit cache.")
    print("- Score_upd counts local dense-sketch reads for ranking the candidate shortlist.")
    print("- Topic-phase rows use a separate scoring sketch updated only by topic events.")
    print("- Topic-cache rows use 2 * topic-phase score + candidate-cache score.")
    print("- This is a non-trained inference skeleton, not an LLM quality result.")
    print()

    gate = run_synthetic_lm_demand_gate_sweep()
    print("Synthetic exact-query content gate")
    print(
        f"facts={gate.fact_count}, events={gate.total_events}, "
        f"query_events={gate.query_events}, lut={gate.lut_state_bytes:0.1f}B, "
        f"write_states={gate.lut_write_state_count}/{len(gate.lut.writes)}"
    )
    gate_headers = [
        "policy",
        "wr/tok/t",
        "demand",
        "d_exact",
        "d_err",
        "carrierM",
        "errM",
        "k_ent",
        "k_sat",
    ]
    print(" | ".join(f"{h:>18}" for h in gate_headers))
    print("-" * 176)
    for point in gate.points:
        row = [
            point.policy,
            f"{point.gate_channel_writes_per_token_tick:0.4f}",
            fmt_pct(point.mean_demand_fraction),
            fmt_pct(point.demand_exact_rate),
            fmt_pct(point.demand_mean_abs_error),
            fmt_pct(point.mean_carrier_exact_retention_rate),
            fmt_pct(point.mean_carrier_mean_abs_error),
            f"{point.carrier_final_entropy_bits:0.2f}",
            fmt_pct(point.carrier_final_saturation_fraction),
        ]
        print(" | ".join(f"{cell:>18}" for cell in row))

    print()
    print("Synthetic gate interpretation:")
    print("- Demand comes from exact-memory query events in the mixed synthetic event stream.")
    print("- Topic events do not demand exact fact rows, so the learned gate should avoid full-field refresh.")


if __name__ == "__main__":
    main()
