"""Deterministic random-noise audit for learned wiki-memory guard tolerance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.wiki_memory import (  # noqa: E402
    WikiMemoryMixedGuardCounterPoint,
    run_wiki_memory_mixed_guard_counter_sweep,
)


@dataclass(frozen=True)
class RandomNoiseCase:
    label: str
    seed: int
    revision_update_rate: float
    cluster_update_rate: float


@dataclass(frozen=True)
class AuditRow:
    case: RandomNoiseCase
    point: WikiMemoryMixedGuardCounterPoint


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def target_dense_on(dense_fraction: float) -> bool:
    return dense_fraction >= 0.50


def make_cases(count: int = 4, sample_seed: int = 2607) -> tuple[RandomNoiseCase, ...]:
    rng = random.Random(sample_seed)
    cases = []
    for index in range(count):
        cases.append(
            RandomNoiseCase(
                label=f"rnd{index}",
                seed=1701 + index * 100,
                revision_update_rate=round(rng.uniform(0.20, 0.90), 2),
                cluster_update_rate=round(rng.uniform(0.10, 0.75), 2),
            )
        )
    return tuple(cases)


def run_case(case: RandomNoiseCase) -> tuple[AuditRow, ...]:
    result = run_wiki_memory_mixed_guard_counter_sweep(
        dense_page_fractions=(0.25, 0.75),
        tag_thresholds=(2,),
        guard_counter_block_page_options=(512,),
        guard_share_radius_options=(1,),
        guard_allowed_loss_options=(0, 1),
        quality_probe_event_options=((512, 256),),
        quality_probe_seed=case.seed,
        revision_update_rate=case.revision_update_rate,
        cluster_update_rate=case.cluster_update_rate,
    )
    return tuple(AuditRow(case, point) for point in result.points)


def print_rows(rows: tuple[AuditRow, ...]) -> None:
    headers = [
        "case",
        "seed",
        "rev",
        "cluster",
        "dense%",
        "loss",
        "target",
        "shared",
        "sh_false",
        "d_w/l",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for row_item in rows:
        case = row_item.case
        point = row_item.point
        row = [
            case.label,
            f"{case.seed}",
            fmt_pct(case.revision_update_rate),
            fmt_pct(case.cluster_update_rate),
            fmt_pct(point.dense_page_fraction),
            f"{point.guard_allowed_loss_count}",
            "on" if target_dense_on(point.dense_page_fraction) else "off",
            fmt_pct(point.dense_shared_enable_rate),
            fmt_pct(point.sparse_shared_false_enable_rate),
            f"{point.dense_raw_wins}/{point.dense_raw_losses}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))


def repaired_failures(rows: tuple[AuditRow, ...]) -> int:
    by_key: dict[tuple[str, float], dict[int, WikiMemoryMixedGuardCounterPoint]] = {}
    for row_item in rows:
        point = row_item.point
        key = (row_item.case.label, point.dense_page_fraction)
        by_key.setdefault(key, {})[point.guard_allowed_loss_count] = point

    repaired = 0
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
    return repaired


def print_summary(rows: tuple[AuditRow, ...]) -> None:
    print()
    print("Summary by loss")
    headers = ["loss", "on_fail", "off_fail", "max_false", "mean_on", "repairs"]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for loss in (0, 1):
        selected = [
            row_item.point
            for row_item in rows
            if row_item.point.guard_allowed_loss_count == loss
        ]
        dense_on = [
            point for point in selected if target_dense_on(point.dense_page_fraction)
        ]
        dense_off = [
            point for point in selected if not target_dense_on(point.dense_page_fraction)
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
        repairs = repaired_failures(rows) if loss == 1 else 0
        row = [
            f"{loss}",
            f"{on_fail}",
            f"{off_fail}",
            fmt_pct(max_false),
            fmt_pct(mean_on),
            f"{repairs}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))


def main() -> None:
    print("CA wiki-memory learned guard randomized noise audit")
    print(
        "config=512-page blocks, same-tag radius=1, dense fractions=(25%,75%), "
        "loss options=(0,1), random_seed=2607"
    )
    rows: list[AuditRow] = []
    for case in make_cases():
        rows.extend(run_case(case))
    audit_rows = tuple(rows)
    print_rows(audit_rows)
    print_summary(audit_rows)
    print()
    print("Interpretation:")
    print("- This is a deterministic pseudo-random sample of update-noise rates.")
    print("- The target failure is sparse false-enable or dense-on under-coverage.")
    print("- A larger randomized audit should replace this smoke test next.")


if __name__ == "__main__":
    main()
