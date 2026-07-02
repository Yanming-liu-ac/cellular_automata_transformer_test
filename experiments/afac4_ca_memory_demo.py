"""Run the AFAC task-4 CA evidence memory prototype."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.afac_memory import run_afac_memory, write_outputs


def main() -> None:
    dataset_root = ROOT / "public_dataset_a" / "public_dataset_upload"
    output_dir = ROOT / "outputs" / "afac4_ca_memory"
    cache_dir = ROOT / "outputs" / "afac4_cache"

    result = run_afac_memory(dataset_root, cache_dir=cache_dir)
    submission_path, trace_path = write_outputs(result, output_dir)

    summary = result.summary
    print("AFAC task-4 CA evidence memory")
    print("config: rare_fanout_cap=32 max_seed_terms=48 propagation_ticks=2 activation_bits=4")
    print(f"questions: {summary.questions}")
    print(f"documents: {summary.documents}")
    print(f"evidence cells: {summary.cells}")
    print(f"avg baseline cells/question: {summary.average_baseline_cells:.1f}")
    print(f"avg CA touched cells/question: {summary.average_ca_touched_cells:.1f}")
    print(
        "avg CA read reduction vs doc-id full scan: "
        f"{summary.average_ca_read_reduction_vs_baseline:.2%}"
    )
    print(f"avg selected score margin vs baseline: {summary.average_selected_score_margin:.2f}")
    print(f"answer formats: {dict(summary.answer_format_counts)}")
    print(f"domains: {dict(summary.domain_counts)}")
    print(f"submission: {submission_path}")
    print(f"trace: {trace_path}")
    print()

    for decision in result.decisions[:5]:
        selected = [
            option for option in decision.option_decisions if option.selected
        ]
        print(
            f"{decision.qid} {decision.answer_format} answer={decision.answer} "
            f"ca_cells={decision.ca_touched_cells} baseline_cells={decision.baseline_cells}"
        )
        for option in selected:
            top = option.evidence[0] if option.evidence else None
            if top is None:
                print(f"  {option.label}: no evidence")
                continue
            page = f"p{top.page}" if top.page is not None else "no-page"
            print(
                f"  {option.label}: score={option.ca_score:.1f} doc={top.doc_id} "
                f"{page} act={top.activation} num={top.number_recall:.2f}"
            )
            print(f"    {top.text[:180]}")


if __name__ == "__main__":
    main()
