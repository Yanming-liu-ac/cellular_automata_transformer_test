"""AFAC 2026 task-4 local evidence memory prototype.

This module is deliberately retrieval-first.  The target experiment is not a
general LLM replacement; it is a CA-shaped memory layer for long financial
documents:

* documents are split into local evidence cells;
* query/option terms inject sparse 4-bit activation into matching cells;
* activation propagates through document-neighbor cells for a few ticks;
* answer candidates are scored from compact evidence rather than full context.

The implementation keeps a traditional full-scan lexical baseline beside the
CA-style sparse activation path so cost can be compared on the same data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from lxml import html as lxml_html
from pypdf import PdfReader


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-.%]*")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?%?"
    r"(?:\s*[\u4e07\u4ebf\u5343\u767e]\u5143|[\u5e74\u6708\u65e5]|%|"
    r"\u500d|\u4e2a|\u65e5|\u6708|\u5e74)?"
)
_COMPANY_ENTITY_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9()\uff08\uff09\u00b7]{2,}"
    r"(?:\u80a1\u4efd\u6709\u9650\u516c\u53f8|"
    r"\u6709\u9650\u8d23\u4efb\u516c\u53f8|\u6709\u9650\u516c\u53f8)"
)
_PARA_SPLIT_RE = re.compile(r"\n{2,}|(?<=\u3002)\s+|(?<=\uff1b)\s+|(?<=;)\s+")
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_NOISY_HTML_LINES = (
    "English",
    "\u79fb\u52a8\u7aef",
    "\u5fae\u535a",
    "\u5fae\u4fe1",
    "\u65e0\u969c\u788d",
    "\u8bf7\u8f93\u5165\u5173\u952e\u5b57",
    "function check",
    "\u9996\u9875",
    "\u673a\u6784\u6982\u51b5",
    "\u65b0\u95fb\u53d1\u5e03",
    "\u653f\u52a1\u4fe1\u606f",
)


@dataclass(frozen=True)
class AFACQuestion:
    """One multiple-choice question from the public AFAC task-4 set."""

    qid: str
    domain: str
    split: str
    question: str
    options: Mapping[str, str]
    answer_format: str
    qtype: str
    doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCell:
    """One local document cell."""

    cell_id: int
    domain: str
    doc_id: str
    source_path: str
    page: int | None
    ordinal: int
    text: str
    features: frozenset[str]
    numbers: frozenset[str]
    is_heading_like: bool


@dataclass(frozen=True)
class EvidenceHit:
    """A ranked evidence cell for one option or assertion."""

    cell_id: int
    domain: str
    doc_id: str
    page: int | None
    activation: int
    lexical_score: float
    number_recall: float
    term_recall: float
    score: float
    text: str


@dataclass(frozen=True)
class OptionDecision:
    """Decision trace for one answer option."""

    label: str
    option: str
    ca_score: float
    baseline_score: float
    margin_to_baseline: float
    touched_cells: int
    propagated_cells: int
    baseline_cells: int
    ca_read_reduction_vs_baseline: float
    number_recall: float
    term_recall: float
    selected: bool
    evidence: tuple[EvidenceHit, ...]


@dataclass(frozen=True)
class QuestionDecision:
    """Decision trace for one AFAC question."""

    qid: str
    domain: str
    answer_format: str
    answer: str
    doc_ids: tuple[str, ...]
    baseline_cells: int
    ca_touched_cells: int
    ca_read_reduction_vs_baseline: float
    option_decisions: tuple[OptionDecision, ...]


@dataclass(frozen=True)
class AFACMemorySummary:
    """Aggregate run metrics."""

    questions: int
    cells: int
    documents: int
    average_baseline_cells: float
    average_ca_touched_cells: float
    average_ca_read_reduction_vs_baseline: float
    average_selected_score_margin: float
    answer_format_counts: Mapping[str, int]
    domain_counts: Mapping[str, int]


@dataclass(frozen=True)
class AFACRunResult:
    """Full answer run result."""

    summary: AFACMemorySummary
    decisions: tuple[QuestionDecision, ...]

    def submission_rows(self) -> list[dict[str, str]]:
        """Return a compact qid/answer JSON-compatible list."""

        return [{"qid": decision.qid, "answer": decision.answer} for decision in self.decisions]


def load_questions(dataset_root: Path | str) -> tuple[AFACQuestion, ...]:
    """Load all public A-group questions."""

    root = Path(dataset_root)
    question_dir = root / "questions" / "group_a"
    questions: list[AFACQuestion] = []
    for path in sorted(question_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            questions.append(
                AFACQuestion(
                    qid=str(row["qid"]),
                    domain=str(row["domain"]),
                    split=str(row.get("split", "")),
                    question=str(row["question"]),
                    options={str(k): str(v) for k, v in row["options"].items()},
                    answer_format=str(row["answer_format"]),
                    qtype=str(row.get("type", "")),
                    doc_ids=tuple(str(value) for value in row["doc_ids"]),
                )
            )
    return tuple(questions)


class AFACDocumentStore:
    """Extract and cache task-4 documents."""

    def __init__(self, dataset_root: Path | str, cache_dir: Path | str | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.raw_root = self.dataset_root / "raw"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, domain: str, doc_id: str) -> Path:
        """Resolve a competition doc_id to a local file."""

        domain_root = self.raw_root / domain
        if domain in {"financial_contracts", "insurance", "research"}:
            candidates = [domain_root / f"{doc_id}.pdf", domain_root / f"{doc_id}.PDF"]
        elif domain == "financial_reports":
            candidates = [
                domain_root / f"{doc_id}.pdf",
                domain_root / f"{doc_id}.PDF",
            ]
        elif domain == "regulatory":
            candidates = [
                domain_root / "txt" / f"{doc_id}.txt",
                domain_root / "html" / f"{doc_id}.html",
                domain_root / "attachments" / f"{doc_id}.pdf",
                domain_root / "attachments" / f"{doc_id}.PDF",
            ]
        else:
            candidates = [domain_root / f"{doc_id}.txt", domain_root / f"{doc_id}.pdf"]

        for path in candidates:
            if path.exists():
                return path

        lowered = doc_id.lower()
        for path in domain_root.rglob("*"):
            if path.is_file() and path.stem.lower() == lowered:
                return path
        raise FileNotFoundError(f"cannot resolve {domain}/{doc_id}")

    def load_pages(self, domain: str, doc_id: str) -> tuple[tuple[int | None, str], Path]:
        """Return page-like text units for one document."""

        path = self.resolve_path(domain, doc_id)
        cached = self._read_cache(path)
        if cached is not None:
            return cached, path

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _extract_pdf_pages(path)
        elif suffix in {".html", ".htm"}:
            pages = ((None, _extract_html_text(path)),)
        else:
            pages = ((None, _read_text_lossy(path)),)

        pages = tuple((page, _clean_text(text)) for page, text in pages if _clean_text(text))
        self._write_cache(path, pages)
        return pages, path

    def _cache_path(self, source_path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        key = "_".join(source_path.parts[-4:])
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, source_path: Path) -> tuple[tuple[int | None, str], ...] | None:
        cache_path = self._cache_path(source_path)
        if cache_path is None or not cache_path.exists():
            return None
        cached_mtime = cache_path.stat().st_mtime
        if cached_mtime < source_path.stat().st_mtime:
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return tuple((row["page"], row["text"]) for row in payload)

    def _write_cache(self, source_path: Path, pages: Sequence[tuple[int | None, str]]) -> None:
        cache_path = self._cache_path(source_path)
        if cache_path is None:
            return
        payload = [{"page": page, "text": text} for page, text in pages]
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class AFACEvidenceMemory:
    """Sparse 4-bit CA evidence memory for AFAC task 4."""

    def __init__(
        self,
        cells: Sequence[EvidenceCell],
        *,
        rare_fanout_cap: int = 96,
        max_seed_terms: int = 96,
        activation_bits: int = 4,
        propagation_ticks: int = 3,
    ) -> None:
        if activation_bits not in (2, 4, 8):
            raise ValueError("activation_bits must be 2, 4, or 8")
        self.cells = tuple(cells)
        self.rare_fanout_cap = int(rare_fanout_cap)
        self.max_seed_terms = int(max_seed_terms)
        self.activation_bits = int(activation_bits)
        self.max_activation = (1 << activation_bits) - 1
        self.propagation_ticks = int(propagation_ticks)
        self.by_doc: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.feature_index: dict[str, list[int]] = defaultdict(list)
        for cell in self.cells:
            self.by_doc[(cell.domain, cell.doc_id)].append(cell.cell_id)
            for feature in cell.features:
                self.feature_index[feature].append(cell.cell_id)
        for ids in self.by_doc.values():
            ids.sort()

    @classmethod
    def build(
        cls,
        store: AFACDocumentStore,
        questions: Sequence[AFACQuestion],
        *,
        chunk_chars: int = 900,
        overlap_chars: int = 120,
        rare_fanout_cap: int = 96,
        max_seed_terms: int = 96,
        activation_bits: int = 4,
        propagation_ticks: int = 3,
    ) -> "AFACEvidenceMemory":
        """Build cells only for documents referenced by the question set."""

        needed = sorted({(q.domain, doc_id) for q in questions for doc_id in q.doc_ids})
        cells: list[EvidenceCell] = []
        for domain, doc_id in needed:
            pages, source_path = store.load_pages(domain, doc_id)
            ordinal = 0
            for page, text in pages:
                for chunk in _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars):
                    features = extract_features(chunk)
                    if not features:
                        continue
                    cell_id = len(cells)
                    cells.append(
                        EvidenceCell(
                            cell_id=cell_id,
                            domain=domain,
                            doc_id=doc_id,
                            source_path=str(source_path),
                            page=page,
                            ordinal=ordinal,
                            text=chunk,
                            features=frozenset(features),
                            numbers=frozenset(extract_numbers(chunk)),
                            is_heading_like=_looks_like_heading(chunk),
                        )
                    )
                    ordinal += 1
        return cls(
            cells,
            rare_fanout_cap=rare_fanout_cap,
            max_seed_terms=max_seed_terms,
            activation_bits=activation_bits,
            propagation_ticks=propagation_ticks,
        )

    def answer_question(self, question: AFACQuestion, *, top_k: int = 3) -> QuestionDecision:
        """Answer one question using option-wise CA evidence activation."""

        doc_filter = {(question.domain, doc_id) for doc_id in question.doc_ids}
        baseline_cells = sum(len(self.by_doc.get(key, ())) for key in doc_filter)
        option_decisions: list[OptionDecision] = []
        for label, option in sorted(question.options.items()):
            option_decisions.append(
                self._score_option(
                    question=question,
                    label=label,
                    option=option,
                    doc_filter=doc_filter,
                    baseline_cells=baseline_cells,
                    top_k=top_k,
                )
            )

        selected_labels = _select_labels(question.answer_format, option_decisions)
        selected = set(selected_labels)
        adjusted = tuple(
            OptionDecision(
                label=decision.label,
                option=decision.option,
                ca_score=decision.ca_score,
                baseline_score=decision.baseline_score,
                margin_to_baseline=decision.margin_to_baseline,
                touched_cells=decision.touched_cells,
                propagated_cells=decision.propagated_cells,
                baseline_cells=decision.baseline_cells,
                ca_read_reduction_vs_baseline=decision.ca_read_reduction_vs_baseline,
                number_recall=decision.number_recall,
                term_recall=decision.term_recall,
                selected=decision.label in selected,
                evidence=decision.evidence,
            )
            for decision in option_decisions
        )
        touched = sum(decision.touched_cells for decision in adjusted)
        denom = max(1, baseline_cells * max(1, len(adjusted)))
        return QuestionDecision(
            qid=question.qid,
            domain=question.domain,
            answer_format=question.answer_format,
            answer="".join(selected_labels),
            doc_ids=question.doc_ids,
            baseline_cells=baseline_cells,
            ca_touched_cells=touched,
            ca_read_reduction_vs_baseline=1.0 - min(1.0, touched / denom),
            option_decisions=adjusted,
        )

    def _score_option(
        self,
        *,
        question: AFACQuestion,
        label: str,
        option: str,
        doc_filter: set[tuple[str, str]],
        baseline_cells: int,
        top_k: int,
    ) -> OptionDecision:
        query_features = extract_features(f"{question.question} {option}")
        assertion_text = question.question if question.answer_format.lower() == "tf" else option
        option_features = extract_features(assertion_text)
        query_numbers = set(extract_numbers(f"{question.question} {option}"))
        option_numbers = set(extract_numbers(assertion_text))
        seed_terms = self._rank_seed_terms(query_features, option_features)

        activation: dict[int, int] = {}
        touched: set[int] = set()
        for term, weight in seed_terms:
            ids = self.feature_index.get(term, ())
            if len(ids) > self.rare_fanout_cap:
                continue
            pulse = max(1, min(self.max_activation, weight))
            for cell_id in ids:
                cell = self.cells[cell_id]
                if (cell.domain, cell.doc_id) not in doc_filter:
                    continue
                activation[cell_id] = min(self.max_activation, activation.get(cell_id, 0) + pulse)
                touched.add(cell_id)

        propagated = self._propagate(activation, doc_filter)
        touched.update(propagated)
        hits = self._rank_hits(
            propagated,
            query_features=query_features,
            option_features=option_features,
            query_numbers=query_numbers,
            option_numbers=option_numbers,
            top_k=top_k,
        )
        baseline_hits = self._baseline_hits(
            doc_filter=doc_filter,
            query_features=query_features,
            option_features=option_features,
            query_numbers=query_numbers,
            option_numbers=option_numbers,
            top_k=top_k,
        )
        ca_score = hits[0].score if hits else 0.0
        baseline_score = baseline_hits[0].score if baseline_hits else 0.0
        number_recall = hits[0].number_recall if hits else 0.0
        term_recall = hits[0].term_recall if hits else 0.0
        read_reduction = 1.0 - min(1.0, len(touched) / max(1, baseline_cells))
        return OptionDecision(
            label=label,
            option=option,
            ca_score=ca_score,
            baseline_score=baseline_score,
            margin_to_baseline=ca_score - baseline_score,
            touched_cells=len(touched),
            propagated_cells=len(propagated),
            baseline_cells=baseline_cells,
            ca_read_reduction_vs_baseline=read_reduction,
            number_recall=number_recall,
            term_recall=term_recall,
            selected=False,
            evidence=tuple(hits),
        )

    def _rank_seed_terms(
        self,
        query_features: set[str],
        option_features: set[str],
    ) -> list[tuple[str, int]]:
        weighted: list[tuple[str, int]] = []
        for feature in query_features:
            fanout = len(self.feature_index.get(feature, ()))
            if fanout == 0:
                continue
            rarity = 4 if fanout <= 8 else 3 if fanout <= 32 else 2 if fanout <= 96 else 1
            if feature in option_features:
                rarity += 2
            if _is_number_like(feature):
                rarity += 3
            elif len(feature) >= 4:
                rarity += 1
            weighted.append((feature, min(self.max_activation, rarity)))
        weighted.sort(key=lambda item: (-item[1], len(self.feature_index.get(item[0], ())), item[0]))
        return weighted[: self.max_seed_terms]

    def _propagate(
        self,
        activation: Mapping[int, int],
        doc_filter: set[tuple[str, str]],
    ) -> dict[int, int]:
        current = dict(activation)
        for _ in range(self.propagation_ticks):
            updated = dict(current)
            for cell_id, value in current.items():
                if value <= 2:
                    continue
                cell = self.cells[cell_id]
                key = (cell.domain, cell.doc_id)
                if key not in doc_filter:
                    continue
                doc_ids = self.by_doc[key]
                pos = cell.ordinal
                pulse = max(1, value - 3)
                for neighbor_pos in (pos - 1, pos + 1):
                    if 0 <= neighbor_pos < len(doc_ids):
                        neighbor_id = doc_ids[neighbor_pos]
                        updated[neighbor_id] = max(updated.get(neighbor_id, 0), pulse)
                if cell.is_heading_like and pos + 2 < len(doc_ids):
                    neighbor_id = doc_ids[pos + 2]
                    updated[neighbor_id] = max(updated.get(neighbor_id, 0), pulse - 1)
            current = {cell_id: min(self.max_activation, value) for cell_id, value in updated.items()}
        return current

    def _rank_hits(
        self,
        activation: Mapping[int, int],
        *,
        query_features: set[str],
        option_features: set[str],
        query_numbers: set[str],
        option_numbers: set[str],
        top_k: int,
    ) -> list[EvidenceHit]:
        hits: list[EvidenceHit] = []
        for cell_id, active in activation.items():
            cell = self.cells[cell_id]
            hit = _score_cell(
                cell,
                query_features=query_features,
                option_features=option_features,
                query_numbers=query_numbers,
                option_numbers=option_numbers,
                activation=active,
            )
            if hit.score > 0:
                hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, -hit.activation, hit.doc_id, hit.page or 0))
        return hits[:top_k]

    def _baseline_hits(
        self,
        *,
        doc_filter: set[tuple[str, str]],
        query_features: set[str],
        option_features: set[str],
        query_numbers: set[str],
        option_numbers: set[str],
        top_k: int,
    ) -> list[EvidenceHit]:
        hits: list[EvidenceHit] = []
        for key in doc_filter:
            for cell_id in self.by_doc.get(key, ()):
                cell = self.cells[cell_id]
                hit = _score_cell(
                    cell,
                    query_features=query_features,
                    option_features=option_features,
                    query_numbers=query_numbers,
                    option_numbers=option_numbers,
                    activation=0,
                )
                if hit.score > 0:
                    hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id, hit.page or 0))
        return hits[:top_k]


def run_afac_memory(
    dataset_root: Path | str,
    *,
    cache_dir: Path | str | None = None,
    limit: int | None = None,
    chunk_chars: int = 900,
    overlap_chars: int = 120,
    rare_fanout_cap: int = 32,
    max_seed_terms: int = 48,
    activation_bits: int = 4,
    propagation_ticks: int = 2,
) -> AFACRunResult:
    """Run the CA evidence memory over the public A-group questions."""

    questions = load_questions(dataset_root)
    if limit is not None:
        questions = questions[: int(limit)]
    store = AFACDocumentStore(dataset_root, cache_dir=cache_dir)
    memory = AFACEvidenceMemory.build(
        store,
        questions,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        rare_fanout_cap=rare_fanout_cap,
        max_seed_terms=max_seed_terms,
        activation_bits=activation_bits,
        propagation_ticks=propagation_ticks,
    )
    decisions = tuple(memory.answer_question(question) for question in questions)
    return AFACRunResult(
        summary=_summarize(memory, decisions),
        decisions=decisions,
    )


def write_outputs(result: AFACRunResult, output_dir: Path | str) -> tuple[Path, Path]:
    """Write submission rows and detailed evidence traces."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    submission_path = out / "afac4_ca_memory_submission.json"
    trace_path = out / "afac4_ca_memory_trace.json"
    submission_path.write_text(
        json.dumps(result.submission_rows(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trace_payload = {
        "summary": {
            "questions": result.summary.questions,
            "cells": result.summary.cells,
            "documents": result.summary.documents,
            "average_baseline_cells": result.summary.average_baseline_cells,
            "average_ca_touched_cells": result.summary.average_ca_touched_cells,
            "average_ca_read_reduction_vs_baseline": (
                result.summary.average_ca_read_reduction_vs_baseline
            ),
            "average_selected_score_margin": result.summary.average_selected_score_margin,
            "answer_format_counts": dict(result.summary.answer_format_counts),
            "domain_counts": dict(result.summary.domain_counts),
        },
        "decisions": [_decision_to_dict(decision) for decision in result.decisions],
    }
    trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return submission_path, trace_path


def extract_features(text: str) -> set[str]:
    """Extract stable lexical/number features without external tokenizers."""

    normalized = _normalize(text)
    features: set[str] = set()
    for entity in extract_entities(normalized):
        features.add(f"ent:{entity}")
    for number in extract_numbers(normalized):
        features.add(number)
    for token in _LATIN_RE.findall(normalized):
        if len(token) >= 2:
            features.add(token.lower())
    for segment in _CJK_RE.findall(normalized):
        if len(segment) <= 1:
            continue
        if len(segment) <= 8:
            features.add(segment)
        for n in (2, 3, 4, 5, 6):
            if len(segment) < n:
                continue
            for i in range(0, len(segment) - n + 1):
                gram = segment[i : i + n]
                if not _is_stop_gram(gram):
                    features.add(gram)
    return features


def extract_entities(text: str) -> set[str]:
    """Extract high-precision organization-like entities."""

    normalized = _normalize(text)
    entities: set[str] = set()
    for raw in _COMPANY_ENTITY_RE.findall(normalized):
        entity = raw
        for marker in (
            "\u4e3a",
            "\u662f",
            "\u7531",
            "\u79f0\u4e3a",
            "\u540d\u79f0",
        ):
            if marker in entity:
                entity = entity.rsplit(marker, 1)[-1]
        entity = entity.strip(" ,\uff0c\u3002;\uff1b:\uff1a()\uff08\uff09[]\u3010\u3011")
        if len(entity) >= 4:
            entities.add(entity)
    return entities


def extract_numbers(text: str) -> set[str]:
    """Extract normalized numeric/date features."""

    normalized = _normalize(text)
    numbers = set()
    for match in _NUMBER_RE.findall(normalized):
        token = re.sub(r"\s+", "", match)
        if token:
            numbers.add(token)
            stripped = token.rstrip("%\u5e74\u6708\u65e5\u500d\u4e2a")
            if stripped and stripped != token:
                numbers.add(stripped)
    return numbers


def _extract_pdf_pages(path: Path) -> tuple[tuple[int | None, str], ...]:
    reader = PdfReader(str(path))
    pages: list[tuple[int | None, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append((index, text))
    return tuple(pages)


def _extract_html_text(path: Path) -> str:
    raw = path.read_bytes()
    root = lxml_html.fromstring(raw)
    for node in root.xpath("//script|//style|//noscript"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)
    lines = []
    for line in html.unescape(root.text_content()).splitlines():
        clean = _SPACE_RE.sub(" ", line).strip()
        if not clean:
            continue
        if any(marker in clean for marker in _NOISY_HTML_LINES):
            continue
        lines.append(clean)
    return "\n".join(lines)


def _read_text_lossy(path: Path) -> str:
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> Iterable[str]:
    paragraphs = [part.strip() for part in _PARA_SPLIT_RE.split(text) if part.strip()]
    current = ""
    for para in paragraphs:
        if len(para) > chunk_chars:
            if current:
                yield current
                current = ""
            step = max(1, chunk_chars - overlap_chars)
            for start in range(0, len(para), step):
                chunk = para[start : start + chunk_chars].strip()
                if chunk:
                    yield chunk
            continue
        if not current:
            current = para
        elif len(current) + 1 + len(para) <= chunk_chars:
            current = f"{current}\n{para}"
        else:
            yield current
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail}\n{para}".strip() if tail else para
    if current:
        yield current


def _score_cell(
    cell: EvidenceCell,
    *,
    query_features: set[str],
    option_features: set[str],
    query_numbers: set[str],
    option_numbers: set[str],
    activation: int,
) -> EvidenceHit:
    common_terms = cell.features & query_features
    option_terms = cell.features & option_features
    option_entities = {feature for feature in option_features if feature.startswith("ent:")}
    matched_entities = cell.features & option_entities
    common_numbers = cell.numbers & query_numbers
    option_common_numbers = cell.numbers & option_numbers
    term_recall = len(common_terms) / max(1, min(len(query_features), 48))
    option_term_recall = len(option_terms) / max(1, min(len(option_features), 32))
    number_recall = len(option_common_numbers) / max(1, len(option_numbers))
    number_score = 4.0 * len(common_numbers) + 5.0 * len(option_common_numbers)
    lexical = (
        0.35 * len(common_terms)
        + 1.85 * len(option_terms)
        + 10.0 * len(matched_entities)
        + number_score
    )
    if option_numbers and number_recall == 0.0:
        lexical *= 0.35
    if option_entities and not matched_entities:
        lexical *= 0.45
    if not option_numbers and option_features and option_term_recall < 0.10:
        lexical *= 0.55
    if cell.is_heading_like:
        lexical *= 0.92
    score = lexical + 0.65 * activation
    return EvidenceHit(
        cell_id=cell.cell_id,
        domain=cell.domain,
        doc_id=cell.doc_id,
        page=cell.page,
        activation=int(activation),
        lexical_score=float(lexical),
        number_recall=float(number_recall),
        term_recall=float(term_recall),
        score=float(score),
        text=_clip_text(cell.text, 420),
    )


def _select_labels(answer_format: str, decisions: Sequence[OptionDecision]) -> tuple[str, ...]:
    if not decisions:
        return ()
    ordered = sorted(decisions, key=lambda item: item.label)
    ranked = sorted(decisions, key=lambda item: item.ca_score, reverse=True)
    fmt = answer_format.lower()
    if fmt in {"mcq", "single"}:
        return (ranked[0].label,)
    if fmt == "tf":
        true_decision = next((item for item in ordered if item.label == "A"), ranked[0])
        threshold = _tf_threshold(true_decision)
        return ("A",) if true_decision.ca_score >= threshold else ("B",)

    best = ranked[0].ca_score
    scores = [item.ca_score for item in ranked]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / max(1, len(scores))
    spread = math.sqrt(variance)
    threshold = max(best * 0.72, mean + 0.15 * spread, 8.0)
    selected = [
        item.label
        for item in ordered
        if item.ca_score >= threshold and _option_gate_passes(item, best)
    ]
    if not selected:
        selected = [ranked[0].label]
    return tuple(selected)


def _tf_threshold(decision: OptionDecision) -> float:
    if decision.number_recall == 0.0 and extract_numbers(decision.option):
        return max(20.0, decision.ca_score + 1.0)
    return 12.0


def _option_gate_passes(decision: OptionDecision, best: float) -> bool:
    if extract_numbers(decision.option) and decision.number_recall < 0.5:
        return decision.ca_score >= best * 0.92
    return True


def _summarize(
    memory: AFACEvidenceMemory,
    decisions: Sequence[QuestionDecision],
) -> AFACMemorySummary:
    questions = len(decisions)
    avg_baseline = sum(item.baseline_cells for item in decisions) / max(1, questions)
    avg_touched = sum(item.ca_touched_cells for item in decisions) / max(1, questions)
    avg_reduction = (
        sum(item.ca_read_reduction_vs_baseline for item in decisions) / max(1, questions)
    )
    selected_margins = []
    for decision in decisions:
        selected_margins.extend(
            item.ca_score - item.baseline_score
            for item in decision.option_decisions
            if item.selected
        )
    return AFACMemorySummary(
        questions=questions,
        cells=len(memory.cells),
        documents=len(memory.by_doc),
        average_baseline_cells=avg_baseline,
        average_ca_touched_cells=avg_touched,
        average_ca_read_reduction_vs_baseline=avg_reduction,
        average_selected_score_margin=(
            sum(selected_margins) / len(selected_margins) if selected_margins else 0.0
        ),
        answer_format_counts=dict(Counter(item.answer_format for item in decisions)),
        domain_counts=dict(Counter(item.domain for item in decisions)),
    )


def _decision_to_dict(decision: QuestionDecision) -> dict[str, object]:
    return {
        "qid": decision.qid,
        "domain": decision.domain,
        "answer_format": decision.answer_format,
        "answer": decision.answer,
        "doc_ids": list(decision.doc_ids),
        "baseline_cells": decision.baseline_cells,
        "ca_touched_cells": decision.ca_touched_cells,
        "ca_read_reduction_vs_baseline": decision.ca_read_reduction_vs_baseline,
        "options": [
            {
                "label": option.label,
                "option": option.option,
                "selected": option.selected,
                "ca_score": option.ca_score,
                "baseline_score": option.baseline_score,
                "touched_cells": option.touched_cells,
                "propagated_cells": option.propagated_cells,
                "number_recall": option.number_recall,
                "term_recall": option.term_recall,
                "evidence": [
                    {
                        "doc_id": hit.doc_id,
                        "page": hit.page,
                        "activation": hit.activation,
                        "score": hit.score,
                        "number_recall": hit.number_recall,
                        "text": hit.text,
                    }
                    for hit in option.evidence
                ],
            }
            for option in decision.option_decisions
        ],
    }


def _clean_text(text: str) -> str:
    normalized = _normalize(text)
    normalized = normalized.replace("\u3000", " ")
    normalized = _SPACE_RE.sub(" ", normalized)
    lines = [line.strip() for line in normalized.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _clip_text(text: str, limit: int) -> str:
    compact = _SPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _looks_like_heading(text: str) -> bool:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    number_chars = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341"
    pattern = rf"\u7b2c[{number_chars}0-9]+[\u7ae0\u8282\u6761]|^[{number_chars}]+[\u3001.]"
    return len(first_line) <= 40 and bool(re.search(pattern, first_line))


def _is_number_like(feature: str) -> bool:
    return any(ch.isdigit() for ch in feature)


def _is_stop_gram(gram: str) -> bool:
    stop = {
        "\u4ee5\u4e0b",
        "\u4e0b\u5217",
        "\u5173\u4e8e",
        "\u6839\u636e",
        "\u7ed3\u5408",
        "\u6587\u6863",
        "\u6b63\u786e",
        "\u9519\u8bef",
        "\u5224\u65ad",
        "\u63cf\u8ff0",
        "\u4fe1\u606f",
        "\u5185\u5bb9",
        "\u76f8\u5173",
        "\u662f\u5426",
        "\u54ea\u4e9b",
        "\u4e00\u4efd",
        "\u4e24\u4efd",
        "\u7b2c\u4e00",
        "\u7b2c\u4e8c",
    }
    return gram in stop
