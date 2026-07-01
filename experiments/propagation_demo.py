"""Compare local CA and HARC-CA propagation depth."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.propagation import run_dynamic_propagation_sweep, summarize_lengths


def fmt_tick(value: int | None) -> str:
    return "miss" if value is None else str(value)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


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


if __name__ == "__main__":
    main()
