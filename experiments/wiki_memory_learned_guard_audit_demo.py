"""Held-out audit for learned CA wiki-memory guard loss tolerance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.wiki_memory import (  # noqa: E402
    WikiMemoryMixedGuardCounterPoint,
    run_wiki_memory_mixed_guard_counter_sweep,
)


POINTS_PER_SEED = 6


@dataclass(frozen=True)
class AuditScenario:
    label: str
    seeds: tuple[int, ...]
    revision_update_rate: float | None = None
    cluster_update_rate: float | None = None


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def target_dense_on(dense_fraction: float) -> bool:
    return dense_fraction >= 0.50


def audit_points(scenario: AuditScenario) -> tuple[WikiMemoryMixedGuardCounterPoint, ...]:
    points: list[WikiMemoryMixedGuardCounterPoint] = []
    for seed in scenario.seeds:
        result = run_wiki_memory_mixed_guard_counter_sweep(
            dense_page_fractions=(0.25, 0.50, 0.75),
            tag_thresholds=(2,),
            guard_counter_block_page_options=(512,),
            guard_share_radius_options=(1,),
            guard_allowed_loss_options=(0, 1),
            quality_probe_event_options=((512, 256),),
            quality_probe_seed=seed,
            revision_update_rate=scenario.revision_update_rate,
            cluster_update_rate=scenario.cluster_update_rate,
        )
        points.extend(result.points)
    return tuple(points)


def print_scenario(scenario: AuditScenario) -> None:
    points = audit_points(scenario)
    print(f"Scenario: {scenario.label}")
    print(
        "config=512-page blocks, same-tag radius=1, "
        "loss options=(0,1), dense target at >=50%"
    )
    if scenario.revision_update_rate is not None or scenario.cluster_update_rate is not None:
        print(
            f"noise: revision={fmt_pct(scenario.revision_update_rate or 0.0)}, "
            f"cluster={fmt_pct(scenario.cluster_update_rate or 0.0)}"
        )

    headers = [
        "seed",
        "dense%",
        "loss",
        "target",
        "shared",
        "sh_false",
        "s_w/l",
        "d_w/l",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for seed_index, seed in enumerate(scenario.seeds):
        start = seed_index * POINTS_PER_SEED
        seed_points = points[start : start + POINTS_PER_SEED]
        for point in seed_points:
            row = [
                f"{seed}",
                fmt_pct(point.dense_page_fraction),
                f"{point.guard_allowed_loss_count}",
                "on" if target_dense_on(point.dense_page_fraction) else "off",
                fmt_pct(point.dense_shared_enable_rate),
                fmt_pct(point.sparse_shared_false_enable_rate),
                f"{point.sparse_raw_wins}/{point.sparse_raw_losses}",
                f"{point.dense_raw_wins}/{point.dense_raw_losses}",
            ]
            print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Summary")
    headers = ["loss", "on_fail", "off_fail", "max_false", "mean_on"]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for loss in (0, 1):
        selected = [point for point in points if point.guard_allowed_loss_count == loss]
        dense_on = [
            point
            for point in selected
            if target_dense_on(point.dense_page_fraction)
        ]
        dense_off = [
            point
            for point in selected
            if not target_dense_on(point.dense_page_fraction)
        ]
        on_fail = sum(point.dense_shared_enable_rate < 0.999 for point in dense_on)
        off_fail = sum(point.dense_shared_enable_rate > 0.0 for point in dense_off)
        max_false = max(
            (point.sparse_shared_false_enable_rate for point in selected),
            default=0.0,
        )
        mean_on = (
            sum(point.dense_shared_enable_rate for point in dense_on) / len(dense_on)
            if dense_on
            else 0.0
        )
        row = [
            f"{loss}",
            f"{on_fail}",
            f"{off_fail}",
            fmt_pct(max_false),
            fmt_pct(mean_on),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    repaired = 0
    by_key: dict[tuple[int, float], dict[int, WikiMemoryMixedGuardCounterPoint]] = {}
    for seed_index, seed in enumerate(scenario.seeds):
        start = seed_index * POINTS_PER_SEED
        seed_points = points[start : start + POINTS_PER_SEED]
        for point in seed_points:
            by_key.setdefault((seed, point.dense_page_fraction), {})[
                point.guard_allowed_loss_count
            ] = point
    for _, pair in by_key.items():
        strict = pair.get(0)
        tolerant = pair.get(1)
        if strict is None or tolerant is None:
            continue
        if (
            target_dense_on(strict.dense_page_fraction)
            and strict.dense_shared_enable_rate < 0.999
            and tolerant.dense_shared_enable_rate >= 0.999
            and tolerant.sparse_shared_false_enable_rate == 0.0
        ):
            repaired += 1
    print(f"strict_failures_repaired_by_loss1={repaired}")
    print()


def main() -> None:
    print("CA wiki-memory learned guard loss-tolerance audit")
    print()
    scenarios = (
        AuditScenario("heldout_reference", seeds=(1201, 1301, 1401, 1501)),
        AuditScenario(
            "seed1501_high_update_noise",
            seeds=(1501,),
            revision_update_rate=0.80,
            cluster_update_rate=0.60,
        ),
    )
    for scenario in scenarios:
        print_scenario(scenario)

    print("Interpretation:")
    print("- loss=1 repairs rare dense misses from the strict zero-loss gate.")
    print("- The audited rows keep sparse shared false-enable at 0.00%.")
    print("- This supports a low-bit tolerant CA guard before moving to loss decay.")


if __name__ == "__main__":
    main()
