"""Compare local CA and HARC-CA propagation depth."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.propagation import (
    run_content_gate_sweep,
    run_content_retention_sweep,
    run_dynamic_propagation_sweep,
    run_learned_content_gate_sweep,
    run_learned_demand_content_gate_sweep,
    run_long_rollout_stability_sweep,
    summarize_lengths,
)


def fmt_tick(value: int | None) -> str:
    return "miss" if value is None else str(value)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def fmt_entropy(value: float) -> str:
    return f"{max(0.0, value):0.2f}"


def fmt_flag(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    rows = summarize_lengths([8, 16, 32, 64, 128, 256, 512, 1024])
    headers = [
        "length",
        "line_steps",
        "harc_steps",
        "line_nodes",
        "harc_nodes",
        "line_edges",
        "harc_edges",
    ]

    print("Propagation from newest token to farthest previous token")
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 88)
    for row in rows:
        print(" | ".join(f"{row[h]:>10}" for h in headers))

    print()
    print("Interpretation:")
    print("- line_steps is the lower-bound pain point of a plain radius-1 CA.")
    print("- harc_steps is the multiscale routing depth that HARC-CA targets in hardware.")
    print()

    dynamic = run_dynamic_propagation_sweep()
    print("Low-bit dynamic propagation rollout")
    print(f"bits={dynamic.bits}, ticks={dynamic.ticks}, source=newest token, target=oldest token")
    headers = [
        "length",
        "topology",
        "rule",
        "nodes",
        "edges",
        "target",
        "all_tok",
        "reach",
        "active",
        "sat",
        "mean",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 148)
    for point in dynamic.points:
        row = [
            str(point.length),
            point.topology,
            point.rule,
            str(point.graph_nodes),
            str(point.graph_edges),
            fmt_tick(point.target_reach_tick),
            fmt_tick(point.all_token_reach_tick),
            fmt_pct(point.final_token_reach_fraction),
            fmt_pct(point.final_active_fraction),
            fmt_pct(point.final_saturation_fraction),
            fmt_pct(point.final_mean_level),
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Dynamic interpretation:")
    print("- residual_avg is stable but slow because integer amplitude moves one level per tick.")
    print("- route_max proves that HARC topology can carry a low-bit signal quickly, but it can saturate the route plane.")
    print("- mhc_grouped keeps a separate route/local/envelope state, giving fast reach with a bounded low-bit envelope.")
    print()

    stability = run_long_rollout_stability_sweep()
    print("1,000-tick unforced low-bit stability sweep")
    print(f"bits={stability.bits}, ticks={stability.ticks}, topology={','.join(stability.topologies)}")
    headers = [
        "rule",
        "init",
        "active0",
        "activeF",
        "satF",
        "satPk",
        "meanF",
        "ent0",
        "entF",
        "minEnt",
        "step",
        "collapse",
        "sat",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 168)
    for point in stability.points:
        row = [
            point.rule,
            point.init_mode,
            fmt_pct(point.initial_active_fraction),
            fmt_pct(point.final_active_fraction),
            fmt_pct(point.final_saturation_fraction),
            fmt_pct(point.peak_saturation_fraction),
            fmt_pct(point.final_mean_level),
            fmt_entropy(point.initial_entropy_bits),
            fmt_entropy(point.final_entropy_bits),
            fmt_entropy(point.min_entropy_bits),
            f"{point.mean_abs_step:0.4f}",
            fmt_flag(point.collapsed),
            fmt_flag(point.saturated),
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Long-rollout interpretation:")
    print("- route_max is a fast propagation primitive, not a stable recurrent state; it saturates from random inputs.")
    print("- residual_avg is stable but can collapse structured signals into a low-entropy rest state.")
    print("- mhc_damped is rejected here: simple leakage prevents saturation but collapses the state to zero.")
    print("- mhc_grouped is the current hand-coded stability scaffold, but it still needs trained dynamics to preserve content.")
    print()

    retention = run_content_retention_sweep()
    print("1,000-tick content retention on mHC carrier")
    print(f"bits={retention.bits}, ticks={retention.ticks}, topology={','.join(retention.topologies)}")
    headers = [
        "policy",
        "state_b",
        "refresh",
        "wr/tok/t",
        "ent0",
        "entF",
        "content",
        "c_err",
        "carrierF",
        "carrierM",
        "k_errM",
        "k_ent",
        "k_sat",
    ]
    print(" | ".join(f"{h:>16}" for h in headers))
    print("-" * 218)
    for point in retention.points:
        row = [
            point.policy,
            str(point.state_bits_per_token),
            str(point.refresh_interval),
            f"{point.refresh_channel_writes_per_token_tick:0.4f}",
            fmt_entropy(point.initial_content_entropy_bits),
            fmt_entropy(point.final_content_entropy_bits),
            fmt_pct(point.content_exact_retention_rate),
            fmt_pct(point.content_mean_abs_error),
            fmt_pct(point.carrier_exact_retention_rate),
            fmt_pct(point.mean_carrier_exact_retention_rate),
            fmt_pct(point.mean_carrier_mean_abs_error),
            fmt_entropy(point.carrier_final_entropy_bits),
            fmt_pct(point.carrier_final_saturation_fraction),
        ]
        print(" | ".join(f"{cell:>16}" for cell in row))

    print()
    print("Content-retention interpretation:")
    print("- shared_mhc confirms the failure: a stable carrier alone does not preserve arbitrary content.")
    print("- content_latch preserves token content exactly with one extra low-bit lane, but the carrier still forgets it.")
    print("- carrierF is phase-sensitive; carrierM reports average content visibility across the rollout.")
    print("- refresh policies trade local writes for keeping the dynamic carrier closer to the persistent content lane.")
    print()

    gate = run_content_gate_sweep()
    print("1,000-tick content-to-carrier gate sweep")
    print(f"bits={gate.bits}, ticks={gate.ticks}, topology={','.join(gate.topologies)}")
    headers = [
        "policy",
        "state_b",
        "wr/tok/t",
        "gate_r",
        "content",
        "carrierF",
        "carrierM",
        "errF",
        "errM",
        "k_ent",
        "k_sat",
    ]
    print(" | ".join(f"{h:>16}" for h in headers))
    print("-" * 202)
    for point in gate.points:
        row = [
            point.policy,
            str(point.state_bits_per_token),
            f"{point.gate_channel_writes_per_token_tick:0.4f}",
            fmt_pct(point.mean_gate_fraction),
            fmt_pct(point.content_exact_retention_rate),
            fmt_pct(point.carrier_exact_retention_rate),
            fmt_pct(point.mean_carrier_exact_retention_rate),
            fmt_pct(point.carrier_mean_abs_error),
            fmt_pct(point.mean_carrier_mean_abs_error),
            fmt_entropy(point.carrier_final_entropy_bits),
            fmt_pct(point.carrier_final_saturation_fraction),
        ]
        print(" | ".join(f"{cell:>16}" for cell in row))

    print()
    print("Gate interpretation:")
    print("- mismatch gates are local: they compare the persistent content lane with the carrier lane.")
    print("- budget_top rows are upper bounds for a future learned gate with a hard write budget.")
    print("- A useful gate should beat fixed_refresh16 on average carrier exactness at lower write traffic.")
    print()

    learned_gate = run_learned_content_gate_sweep()
    print("Learned content-to-carrier LUT gate")
    print(
        f"lut={learned_gate.lut.state_bytes:0.1f}B, "
        f"write_states={learned_gate.lut.write_state_count}/{len(learned_gate.lut.writes)}, "
        f"cost={learned_gate.write_cost:0.2f}, "
        f"route_w={learned_gate.route_weight:0.2f}, "
        f"env_w={learned_gate.envelope_weight:0.2f}"
    )
    headers = [
        "policy",
        "wr/tok/t",
        "gate_r",
        "carrierF",
        "carrierM",
        "errF",
        "errM",
        "k_ent",
        "k_sat",
    ]
    print(" | ".join(f"{h:>16}" for h in headers))
    print("-" * 162)
    for point in learned_gate.points:
        row = [
            point.policy,
            f"{point.gate_channel_writes_per_token_tick:0.4f}",
            fmt_pct(point.mean_gate_fraction),
            fmt_pct(point.carrier_exact_retention_rate),
            fmt_pct(point.mean_carrier_exact_retention_rate),
            fmt_pct(point.carrier_mean_abs_error),
            fmt_pct(point.mean_carrier_mean_abs_error),
            fmt_entropy(point.carrier_final_entropy_bits),
            fmt_pct(point.carrier_final_saturation_fraction),
        ]
        print(" | ".join(f"{cell:>16}" for cell in row))

    print()
    print("Learned-gate interpretation:")
    print("- The LUT uses only mismatch, route, and envelope buckets, so its table is hardware-sized.")
    print("- This is a first local controller, not a trained language rule; it should beat at least one hand gate to stay interesting.")
    print()

    demand_gate = run_learned_demand_content_gate_sweep()
    print("Demand-weighted content gate LUT")
    print(
        f"lut={demand_gate.lut.state_bytes:0.1f}B, "
        f"write_states={demand_gate.lut.write_state_count}/{len(demand_gate.lut.writes)}, "
        f"demand={fmt_pct(demand_gate.demand_rate)}, "
        f"cost={demand_gate.write_cost:0.2f}"
    )
    headers = [
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
    print(" | ".join(f"{h:>18}" for h in headers))
    print("-" * 176)
    for point in demand_gate.points:
        row = [
            point.policy,
            f"{point.gate_channel_writes_per_token_tick:0.4f}",
            fmt_pct(point.mean_demand_fraction),
            fmt_pct(point.demand_exact_rate),
            fmt_pct(point.demand_mean_abs_error),
            fmt_pct(point.mean_carrier_exact_retention_rate),
            fmt_pct(point.mean_carrier_mean_abs_error),
            fmt_entropy(point.carrier_final_entropy_bits),
            fmt_pct(point.carrier_final_saturation_fraction),
        ]
        print(" | ".join(f"{cell:>18}" for cell in row))

    print()
    print("Demand-gate interpretation:")
    print("- Demand-weighted metrics score only token cells requested by the route/query lane.")
    print("- A good demand gate can spend writes on requested content without reconstructing the whole carrier.")


if __name__ == "__main__":
    main()
