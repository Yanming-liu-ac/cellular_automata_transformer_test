"""Sparse Cellular-MoE rule-bank demo."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.cellular_moe import CellularMoE, CellularMoEConfig


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def run_case(name: str, balance_rate: float) -> tuple[str, object]:
    config = CellularMoEConfig(
        length=2048,
        channels=16,
        bits=4,
        rule_banks=6,
        top_k=1,
        active_budget_fraction=0.20,
        balance_rate=balance_rate,
    )
    moe = CellularMoE(config, seed=13)
    moe.randomize_sparse(density=0.06)
    moe.inject_patch(256, 96)
    moe.inject_patch(1400, 64, value=10)
    return name, moe.rollout(128)


def main() -> None:
    rows = [
        run_case("no_balance", 0.0),
        run_case("bias_balance", 0.18),
    ]

    headers = [
        "case",
        "active",
        "upd_reduce",
        "load_cv",
        "saturation",
        "sparse",
        "dense",
        "final_loads",
    ]
    print("Low-bit Cellular-MoE sparse rule-bank rollout")
    print("length=2048, channels=16, bits=4, rule_banks=6, top_k=1, ticks=128")
    print(" | ".join(f"{h:>13}" for h in headers))
    print("-" * 132)
    for name, result in rows:
        row = [
            name,
            fmt_pct(result.avg_active_fraction),
            f"{result.avg_update_reduction:0.1f}x",
            f"{result.avg_load_cv:0.3f}",
            fmt_pct(result.final_saturation_fraction),
            f"{result.total_sparse_rule_updates}",
            f"{result.total_dense_rule_updates}",
            str(result.final_rule_loads),
        ]
        print(" | ".join(f"{cell:>13}" for cell in row))

    print()
    print("Interpretation:")
    print("- upd_reduce compares dense all-rules execution with sparse routed rule updates.")
    print("- bias_balance adjusts routing bias from observed rule loads, without a loss term.")
    print("- This is an execution-shape prototype; learned rule banks come later.")


if __name__ == "__main__":
    main()
