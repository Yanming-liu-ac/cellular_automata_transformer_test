"""CA-native wiki-memory benchmark.

This module models a small external knowledge fabric: pages contain exact facts,
pages are grouped under low-bit summaries, page links support a second hop, and
updates dirty only local page/group summaries. The benchmark is deliberately
synthetic, but it measures the hardware question directly: can local triggered
refresh keep mutable knowledge queryable without scanning the whole wiki?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .retrieval import keyed_hash


@dataclass(frozen=True)
class WikiMemoryConfig:
    """Synthetic wiki-memory geometry."""

    page_count: int = 256
    facts_per_page: int = 4
    topic_count: int = 64
    links_per_page: int = 4
    group_size: int = 16
    selected_groups: int = 4
    selected_pages: int = 8
    summary_banks: int = 4
    summary_width: int = 256
    summary_bits: int = 4
    query_events: int = 512
    update_events: int = 256
    multihop_query_rate: float = 0.35
    recent_update_query_rate: float = 0.45

    def __post_init__(self) -> None:
        if self.page_count <= 0:
            raise ValueError("page_count must be positive")
        if self.facts_per_page <= 0:
            raise ValueError("facts_per_page must be positive")
        if self.topic_count <= 0:
            raise ValueError("topic_count must be positive")
        if self.links_per_page <= 0:
            raise ValueError("links_per_page must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.page_count % self.group_size != 0:
            raise ValueError("page_count must be divisible by group_size")
        if self.selected_groups <= 0:
            raise ValueError("selected_groups must be positive")
        if self.selected_pages <= 0:
            raise ValueError("selected_pages must be positive")
        if self.summary_banks <= 0:
            raise ValueError("summary_banks must be positive")
        if self.summary_width <= 0:
            raise ValueError("summary_width must be positive")
        if self.summary_bits not in (2, 4, 8):
            raise ValueError("summary_bits must be one of 2, 4, 8")
        if self.query_events <= 0:
            raise ValueError("query_events must be positive")
        if self.update_events < 0:
            raise ValueError("update_events must be non-negative")
        if not 0.0 <= self.multihop_query_rate <= 1.0:
            raise ValueError("multihop_query_rate must be in [0, 1]")
        if not 0.0 <= self.recent_update_query_rate <= 1.0:
            raise ValueError("recent_update_query_rate must be in [0, 1]")

    @property
    def group_count(self) -> int:
        return self.page_count // self.group_size

    @property
    def max_summary_value(self) -> int:
        return (1 << self.summary_bits) - 1

    @property
    def summary_state_bytes(self) -> float:
        cells = (self.page_count + self.group_count) * self.summary_banks * self.summary_width
        return cells * self.summary_bits / 8.0

    @property
    def metadata_state_bytes(self) -> float:
        dirty_bits = self.page_count + self.group_count
        page_versions = self.page_count * 16
        links = self.page_count * self.links_per_page * 16
        fact_payload = self.page_count * self.facts_per_page * 64
        return (dirty_bits + page_versions + links + fact_payload) / 8.0

    @property
    def state_bytes(self) -> float:
        return self.summary_state_bytes + self.metadata_state_bytes


@dataclass(frozen=True)
class WikiMemoryRefreshPolicy:
    """Local refresh policy for dirty wiki summaries."""

    name: str
    dirty_threshold: int
    max_age: int
    refresh_on_update: bool = False
    error_book_repair: bool = False

    def __post_init__(self) -> None:
        if self.dirty_threshold <= 0:
            raise ValueError("dirty_threshold must be positive")
        if self.max_age < 0:
            raise ValueError("max_age must be non-negative")


@dataclass(frozen=True)
class WikiMemoryTrialPoint:
    """One wiki-memory policy measurement."""

    policy: str
    dirty_threshold: int
    max_age: int
    refresh_on_update: bool
    error_book_repair: bool
    queries: int
    updates: int
    single_hop_recall: float
    multihop_recall: float
    overall_recall: float
    recent_update_recall: float
    stale_miss_rate: float
    route_miss_rate: float
    provenance_precision: float
    cells_read_per_query: float
    flat_cells_read_per_query: float
    read_reduction_rate: float
    cells_written_per_update: float
    refresh_events: int
    mean_pages_refreshed: float
    mean_groups_refreshed: float
    error_book_repairs: int
    error_book_recoveries: int
    dirty_pages_end: int
    state_bytes: float


@dataclass(frozen=True)
class WikiMemorySweepResult:
    """Synthetic wiki-memory policy sweep."""

    page_count: int
    facts_per_page: int
    links_per_page: int
    group_size: int
    selected_groups: int
    selected_pages: int
    summary_banks: int
    summary_width: int
    summary_bits: int
    query_events: int
    update_events: int
    state_bytes: float
    points: Tuple[WikiMemoryTrialPoint, ...]


@dataclass(frozen=True)
class _RouteResult:
    found: bool
    cells_read: int
    selected_pages: Tuple[int, ...]
    source_page: int | None


class _SyntheticWikiMemory:
    """Mutable synthetic wiki backed by low-bit page and group summaries."""

    def __init__(self, config: WikiMemoryConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.fact_keys = np.zeros((config.page_count, config.facts_per_page), dtype=np.int64)
        self.fact_values = np.zeros_like(self.fact_keys)
        self.page_versions = np.zeros(config.page_count, dtype=np.int32)
        self.page_topics = np.arange(config.page_count, dtype=np.int32) % config.topic_count
        self.links = np.zeros((config.page_count, config.links_per_page), dtype=np.int32)
        self.page_summary = np.zeros(
            (config.page_count, config.summary_banks, config.summary_width),
            dtype=np.uint8,
        )
        self.group_summary = np.zeros(
            (config.group_count, config.summary_banks, config.summary_width),
            dtype=np.uint8,
        )
        self.dirty_pages = np.zeros(config.page_count, dtype=np.bool_)
        self.dirty_groups = np.zeros(config.group_count, dtype=np.bool_)
        self.summary_age = 0
        self.update_cursor = 0
        self.recent_pages: List[int] = []
        self._initialize_wiki()
        self._refresh_pages(np.arange(config.page_count, dtype=np.int32))
        self._refresh_groups(np.arange(config.group_count, dtype=np.int32))

    def _initialize_wiki(self) -> None:
        for page in range(self.config.page_count):
            for slot in range(self.config.facts_per_page):
                key = int(keyed_hash(page * 1009 + slot, 17) & ((1 << 31) - 1))
                self.fact_keys[page, slot] = key
                self.fact_values[page, slot] = self._value_for_key(key)

        for page in range(self.config.page_count):
            same_topic = np.flatnonzero(self.page_topics == self.page_topics[page])
            same_topic = same_topic[same_topic != page]
            if len(same_topic) < self.config.links_per_page:
                pool = np.arange(self.config.page_count, dtype=np.int32)
                pool = pool[pool != page]
            else:
                pool = same_topic
            self.links[page] = self.rng.choice(
                pool,
                size=self.config.links_per_page,
                replace=len(pool) < self.config.links_per_page,
            )

    def _slots(self, key: int) -> Tuple[int, ...]:
        return tuple(
            int(keyed_hash(int(key), 3000 + bank) % self.config.summary_width)
            for bank in range(self.config.summary_banks)
        )

    def _value_for_key(self, key: int) -> int:
        return int(keyed_hash(int(key), 7103) & ((1 << 31) - 1))

    def _group_for_page(self, page: int) -> int:
        return int(page) // self.config.group_size

    def _refresh_pages(self, pages: np.ndarray) -> int:
        if len(pages) == 0:
            return 0
        touched = 0
        max_value = self.config.max_summary_value
        for page in pages.astype(np.int32):
            self.page_summary[page].fill(0)
            touched += self.config.summary_banks * self.config.summary_width
            for key in self.fact_keys[page]:
                for bank, slot in enumerate(self._slots(int(key))):
                    value = int(self.page_summary[page, bank, slot])
                    if value < max_value:
                        self.page_summary[page, bank, slot] = value + 1
                    touched += 1
            self.dirty_pages[page] = False
            self.dirty_groups[self._group_for_page(int(page))] = True
        return touched

    def _refresh_groups(self, groups: np.ndarray) -> int:
        if len(groups) == 0:
            return 0
        touched = 0
        for group in groups.astype(np.int32):
            start = int(group) * self.config.group_size
            end = start + self.config.group_size
            self.group_summary[group] = np.max(self.page_summary[start:end], axis=0)
            touched += (
                self.config.group_size * self.config.summary_banks * self.config.summary_width
                + self.config.summary_banks * self.config.summary_width
            )
            self.dirty_groups[group] = False
        return touched

    def _refresh_dirty(self) -> Tuple[int, int, int]:
        pages = np.flatnonzero(self.dirty_pages).astype(np.int32)
        page_cells = self._refresh_pages(pages)
        groups = np.flatnonzero(self.dirty_groups).astype(np.int32)
        group_cells = self._refresh_groups(groups)
        if len(pages) > 0 or len(groups) > 0:
            self.summary_age = 0
        return page_cells + group_cells, int(len(pages)), int(len(groups))

    def maybe_refresh(self, policy: WikiMemoryRefreshPolicy) -> Tuple[int, int, int, bool]:
        dirty_count = int(np.count_nonzero(self.dirty_pages))
        age_due = policy.max_age > 0 and self.summary_age >= policy.max_age
        count_due = dirty_count >= policy.dirty_threshold
        if dirty_count > 0 and (age_due or count_due):
            cells, pages, groups = self._refresh_dirty()
            return cells, pages, groups, True
        return 0, 0, 0, False

    def update_fact(self, policy: WikiMemoryRefreshPolicy) -> Tuple[int, int, int, bool]:
        page = int(self.rng.integers(0, self.config.page_count))
        slot = int(self.rng.integers(0, self.config.facts_per_page))
        new_key = int(keyed_hash(1_000_003 + self.update_cursor * 65537 + page * 257 + slot, 29))
        new_key &= (1 << 31) - 1
        self.fact_keys[page, slot] = new_key
        self.fact_values[page, slot] = self._value_for_key(new_key)
        self.page_versions[page] += 1
        self.dirty_pages[page] = True
        self.dirty_groups[self._group_for_page(page)] = True
        self.summary_age += 1
        self.update_cursor += 1
        self.recent_pages.append(page)
        if len(self.recent_pages) > 64:
            self.recent_pages = self.recent_pages[-64:]

        update_cells = 4
        if policy.refresh_on_update:
            refresh_cells, pages, groups = self._refresh_dirty()
            return update_cells + refresh_cells, pages, groups, refresh_cells > 0
        return update_cells, 0, 0, False

    def _score_groups(self, key: int) -> np.ndarray:
        slots = self._slots(key)
        scores = np.zeros(self.config.group_count, dtype=np.int32)
        for bank, slot in enumerate(slots):
            scores += self.group_summary[:, bank, slot].astype(np.int32)
        return scores

    def _score_pages(self, key: int, pages: np.ndarray) -> np.ndarray:
        slots = self._slots(key)
        scores = np.zeros(len(pages), dtype=np.int32)
        for bank, slot in enumerate(slots):
            scores += self.page_summary[pages, bank, slot].astype(np.int32)
        return scores

    def _top_indices(self, scores: np.ndarray, count: int) -> np.ndarray:
        if len(scores) == 0:
            return np.empty(0, dtype=np.int32)
        tiebreaker = np.arange(len(scores), dtype=np.int32)
        order = np.lexsort((tiebreaker, scores))[::-1]
        return order[: min(count, len(order))].astype(np.int32)

    def route_key(self, key: int) -> _RouteResult:
        group_scores = self._score_groups(key)
        group_local = self._top_indices(group_scores, self.config.selected_groups)
        candidate_pages: List[int] = []
        for group in group_local:
            start = int(group) * self.config.group_size
            candidate_pages.extend(range(start, start + self.config.group_size))
        candidate_array = np.array(candidate_pages, dtype=np.int32)
        page_scores = self._score_pages(key, candidate_array)
        page_local = self._top_indices(page_scores, self.config.selected_pages)
        selected_pages = tuple(int(candidate_array[index]) for index in page_local)
        cells_read = (
            self.config.group_count * self.config.summary_banks
            + len(candidate_array) * self.config.summary_banks
            + len(selected_pages) * self.config.facts_per_page
        )
        for page in selected_pages:
            if bool(np.any(self.fact_keys[page] == int(key))):
                return _RouteResult(True, cells_read, selected_pages, page)
        return _RouteResult(False, cells_read, selected_pages, None)

    def repair_page(self, page: int) -> Tuple[int, int, int]:
        page_array = np.array([int(page)], dtype=np.int32)
        page_cells = self._refresh_pages(page_array)
        group_cells = self._refresh_groups(np.array([self._group_for_page(page)], dtype=np.int32))
        return page_cells + group_cells, 1, 1

    def sample_query(self) -> Tuple[str, int, int, int, bool]:
        use_recent = (
            len(self.recent_pages) > 0
            and self.rng.random() < self.config.recent_update_query_rate
        )
        if use_recent:
            page = int(self.recent_pages[int(self.rng.integers(0, len(self.recent_pages)))])
        else:
            page = int(self.rng.integers(0, self.config.page_count))
        slot = int(self.rng.integers(0, self.config.facts_per_page))

        if self.rng.random() < self.config.multihop_query_rate:
            target_page = int(self.links[page, int(self.rng.integers(0, self.config.links_per_page))])
            target_slot = int(self.rng.integers(0, self.config.facts_per_page))
            return (
                "multihop",
                int(self.fact_keys[page, slot]),
                int(self.fact_keys[target_page, target_slot]),
                target_page,
                use_recent,
            )
        return (
            "single",
            int(self.fact_keys[page, slot]),
            int(self.fact_values[page, slot]),
            page,
            use_recent,
        )

    def answer_query(self, query: Tuple[str, int, int, int, bool]) -> Tuple[bool, bool, bool, int]:
        kind, route_key, target, target_page, _ = query
        routed = self.route_key(route_key)
        cells_read = routed.cells_read
        if not routed.found:
            stale = bool(self.dirty_pages[target_page])
            return False, stale, False, cells_read

        if kind == "single":
            page = int(routed.source_page) if routed.source_page is not None else -1
            found = bool(page == target_page and np.any(self.fact_values[page] == int(target)))
            stale = bool((not found) and self.dirty_pages[target_page])
            return found, stale, found, cells_read

        cells_read += self.config.links_per_page
        link_pages = self.links[int(routed.source_page)]
        cells_read += self.config.links_per_page * self.config.facts_per_page
        for page in link_pages:
            if int(page) == target_page and bool(np.any(self.fact_keys[page] == int(target))):
                return True, False, True, cells_read
        return False, bool(self.dirty_pages[target_page]), False, cells_read


def _trial(policy: WikiMemoryRefreshPolicy, config: WikiMemoryConfig, seed: int) -> WikiMemoryTrialPoint:
    wiki = _SyntheticWikiMemory(config, seed)
    event_types = np.array(["query"] * config.query_events + ["update"] * config.update_events)
    wiki.rng.shuffle(event_types)

    queries = 0
    updates = 0
    single_queries = 0
    multihop_queries = 0
    single_hits = 0
    multihop_hits = 0
    recent_queries = 0
    recent_hits = 0
    stale_misses = 0
    route_misses = 0
    provenance_hits = 0
    total_cells_read = 0
    total_flat_cells_read = 0
    total_cells_written = 0
    refresh_events = 0
    pages_refreshed = 0
    groups_refreshed = 0
    error_repairs = 0
    error_recoveries = 0

    for event_type in event_types:
        if event_type == "update":
            cells, pages, groups, refreshed = wiki.update_fact(policy)
            total_cells_written += cells
            updates += 1
            if refreshed:
                refresh_events += 1
                pages_refreshed += pages
                groups_refreshed += groups
            continue

        cells, pages, groups, refreshed = wiki.maybe_refresh(policy)
        total_cells_written += cells
        if refreshed:
            refresh_events += 1
            pages_refreshed += pages
            groups_refreshed += groups

        query = wiki.sample_query()
        kind, _, _, target_page, recent = query
        hit, stale, precise, cells_read = wiki.answer_query(query)
        queries += 1
        total_cells_read += cells_read
        total_flat_cells_read += config.page_count * config.facts_per_page
        if kind == "single":
            single_queries += 1
            single_hits += int(hit)
        else:
            multihop_queries += 1
            multihop_hits += int(hit)
        recent_queries += int(recent)
        recent_hits += int(recent and hit)
        stale_misses += int((not hit) and stale)
        route_misses += int(not hit)
        provenance_hits += int(hit and precise)

        if (not hit) and policy.error_book_repair:
            repair_cells, repair_pages, repair_groups = wiki.repair_page(target_page)
            total_cells_written += repair_cells
            refresh_events += 1
            pages_refreshed += repair_pages
            groups_refreshed += repair_groups
            error_repairs += 1
            repaired_hit, _, _, _ = wiki.answer_query(query)
            error_recoveries += int(repaired_hit)

    overall_hits = single_hits + multihop_hits
    cells_read_per_query = total_cells_read / float(queries)
    flat_cells_read_per_query = total_flat_cells_read / float(queries)
    cells_written_per_update = (
        total_cells_written / float(updates) if updates > 0 else 0.0
    )
    return WikiMemoryTrialPoint(
        policy=policy.name,
        dirty_threshold=policy.dirty_threshold,
        max_age=policy.max_age,
        refresh_on_update=policy.refresh_on_update,
        error_book_repair=policy.error_book_repair,
        queries=queries,
        updates=updates,
        single_hop_recall=single_hits / float(single_queries) if single_queries else 0.0,
        multihop_recall=multihop_hits / float(multihop_queries) if multihop_queries else 0.0,
        overall_recall=overall_hits / float(queries),
        recent_update_recall=recent_hits / float(recent_queries) if recent_queries else 0.0,
        stale_miss_rate=stale_misses / float(queries),
        route_miss_rate=route_misses / float(queries),
        provenance_precision=provenance_hits / float(overall_hits) if overall_hits else 0.0,
        cells_read_per_query=cells_read_per_query,
        flat_cells_read_per_query=flat_cells_read_per_query,
        read_reduction_rate=1.0 - cells_read_per_query / flat_cells_read_per_query,
        cells_written_per_update=cells_written_per_update,
        refresh_events=refresh_events,
        mean_pages_refreshed=pages_refreshed / float(refresh_events) if refresh_events else 0.0,
        mean_groups_refreshed=groups_refreshed / float(refresh_events) if refresh_events else 0.0,
        error_book_repairs=error_repairs,
        error_book_recoveries=error_recoveries,
        dirty_pages_end=int(np.count_nonzero(wiki.dirty_pages)),
        state_bytes=config.state_bytes,
    )


def run_wiki_memory_sweep(
    config: WikiMemoryConfig | None = None,
    policies: Tuple[WikiMemoryRefreshPolicy, ...] = (
        WikiMemoryRefreshPolicy("exact_update", dirty_threshold=1, max_age=0, refresh_on_update=True),
        WikiMemoryRefreshPolicy("trigger16_age16", dirty_threshold=16, max_age=16),
        WikiMemoryRefreshPolicy(
            "trigger16_age16_errorbook",
            dirty_threshold=16,
            max_age=16,
            error_book_repair=True,
        ),
        WikiMemoryRefreshPolicy("trigger32_age64", dirty_threshold=32, max_age=64),
        WikiMemoryRefreshPolicy("stale_no_refresh", dirty_threshold=1_000_000, max_age=0),
    ),
    seed: int = 91,
) -> WikiMemorySweepResult:
    """Run a synthetic mutable wiki-memory policy sweep."""

    sweep_config = config or WikiMemoryConfig()
    points = tuple(_trial(policy=policy, config=sweep_config, seed=seed) for policy in policies)
    return WikiMemorySweepResult(
        page_count=sweep_config.page_count,
        facts_per_page=sweep_config.facts_per_page,
        links_per_page=sweep_config.links_per_page,
        group_size=sweep_config.group_size,
        selected_groups=sweep_config.selected_groups,
        selected_pages=sweep_config.selected_pages,
        summary_banks=sweep_config.summary_banks,
        summary_width=sweep_config.summary_width,
        summary_bits=sweep_config.summary_bits,
        query_events=sweep_config.query_events,
        update_events=sweep_config.update_events,
        state_bytes=sweep_config.state_bytes,
        points=points,
    )
