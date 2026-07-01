"""CA-native wiki-memory benchmark.

This module models a small external knowledge fabric: pages contain exact facts,
pages are grouped under low-bit summaries, page links support a second hop, and
updates dirty only local page/group summaries. The benchmark is deliberately
synthetic, but it measures the hardware question directly: can local triggered
refresh keep mutable knowledge queryable without scanning the whole wiki?
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np

from .retrieval import keyed_hash


_DEFAULT_FANOUT_TRAIN_SEEDS = tuple(range(11, 11 + 64 * 18, 18))


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
    adaptive_max_groups: int = 32
    adaptive_score_margin: int = 0
    summary_banks: int = 4
    summary_width: int = 256
    summary_bits: int = 4
    query_events: int = 512
    update_events: int = 256
    multihop_query_rate: float = 0.35
    recent_update_query_rate: float = 0.45
    revision_update_rate: float = 0.50
    error_probe_query_rate: float = 0.25
    contradiction_clusters: int = 32
    cluster_sources: int = 3
    cluster_update_rate: float = 0.30
    cluster_query_rate: float = 0.25

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
        if self.adaptive_max_groups <= 0:
            raise ValueError("adaptive_max_groups must be positive")
        if self.adaptive_score_margin < 0:
            raise ValueError("adaptive_score_margin must be non-negative")
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
        if not 0.0 <= self.revision_update_rate <= 1.0:
            raise ValueError("revision_update_rate must be in [0, 1]")
        if not 0.0 <= self.error_probe_query_rate <= 1.0:
            raise ValueError("error_probe_query_rate must be in [0, 1]")
        if self.contradiction_clusters < 0:
            raise ValueError("contradiction_clusters must be non-negative")
        if self.cluster_sources <= 0:
            raise ValueError("cluster_sources must be positive")
        if self.contradiction_clusters * self.cluster_sources > self.page_count:
            raise ValueError("cluster sources must fit in page_count")
        if not 0.0 <= self.cluster_update_rate <= 1.0:
            raise ValueError("cluster_update_rate must be in [0, 1]")
        if not 0.0 <= self.cluster_query_rate <= 1.0:
            raise ValueError("cluster_query_rate must be in [0, 1]")

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
        cluster_links = self.contradiction_clusters * self.cluster_sources * 16
        return (dirty_bits + page_versions + links + fact_payload + cluster_links) / 8.0

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
    cluster_repair: bool = False

    def __post_init__(self) -> None:
        if self.dirty_threshold <= 0:
            raise ValueError("dirty_threshold must be positive")
        if self.max_age < 0:
            raise ValueError("max_age must be non-negative")


@dataclass(frozen=True)
class WikiMemoryTrialPoint:
    """One wiki-memory policy measurement."""

    policy: str
    route_mode: str
    dirty_threshold: int
    max_age: int
    refresh_on_update: bool
    error_book_repair: bool
    cluster_repair: bool
    queries: int
    updates: int
    single_hop_recall: float
    multihop_recall: float
    overall_recall: float
    recent_update_recall: float
    stale_miss_rate: float
    route_miss_rate: float
    value_miss_rate: float
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
    error_probe_queries: int
    error_probe_recall: float
    cluster_queries: int
    cluster_recall: float
    cluster_consistency_rate: float
    key_updates: int
    revision_updates: int
    cluster_updates: int
    cluster_repair_events: int
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
    revision_update_rate: float
    error_probe_query_rate: float
    contradiction_clusters: int
    cluster_sources: int
    cluster_update_rate: float
    cluster_query_rate: float
    state_bytes: float
    points: Tuple[WikiMemoryTrialPoint, ...]
    flat_points: Tuple[WikiMemoryTrialPoint, ...]


@dataclass(frozen=True)
class WikiMemoryScalingPoint:
    """One page-count scaling comparison between CA and flat retrieval."""

    page_count: int
    facts_per_page: int
    group_size: int
    selected_groups: int
    selected_pages: int
    contradiction_clusters: int
    state_bytes: float
    ca_overall_recall: float
    flat_overall_recall: float
    ca_cluster_consistency_rate: float
    flat_cluster_consistency_rate: float
    ca_cells_read_per_query: float
    flat_cells_read_per_query: float
    exact_scan_cells_per_query: float
    ca_cells_written_per_update: float
    flat_cells_written_per_update: float
    ca_read_reduction_vs_flat: float
    ca_read_reduction_vs_exact_scan: float


@dataclass(frozen=True)
class WikiMemoryScalingResult:
    """Page-count scaling sweep for routed CA versus flat page-summary scan."""

    policy: str
    query_events: int
    update_events: int
    summary_banks: int
    summary_width: int
    summary_bits: int
    points: Tuple[WikiMemoryScalingPoint, ...]


@dataclass(frozen=True)
class WikiMemoryDensityPoint:
    """One facts-per-page and summary-width pressure point."""

    page_count: int
    facts_per_page: int
    summary_width: int
    contradiction_clusters: int
    state_bytes: float
    ca_overall_recall: float
    flat_overall_recall: float
    ca_cluster_consistency_rate: float
    flat_cluster_consistency_rate: float
    ca_cells_read_per_query: float
    flat_cells_read_per_query: float
    exact_scan_cells_per_query: float
    ca_cells_written_per_update: float
    flat_cells_written_per_update: float
    ca_read_reduction_vs_flat: float
    ca_read_reduction_vs_exact_scan: float


@dataclass(frozen=True)
class WikiMemoryDensityResult:
    """Facts-per-page and summary-width pressure sweep."""

    policy: str
    page_count: int
    query_events: int
    update_events: int
    summary_banks: int
    summary_bits: int
    points: Tuple[WikiMemoryDensityPoint, ...]


@dataclass(frozen=True)
class WikiMemoryFanoutLUT:
    """Low-bit local policy table for wiki group read fanout."""

    base_groups: int
    max_groups: int
    target_route_coverage: float
    top_score_buckets: int
    base_score_buckets: int
    gap_buckets: int
    exact_tie_bounds: Tuple[int, ...]
    near_tie_bounds: Tuple[int, ...]
    fanout_bits: int
    fanouts: Tuple[int, ...]
    training_examples: int
    train_seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.base_groups <= 0:
            raise ValueError("base_groups must be positive")
        if self.max_groups < self.base_groups:
            raise ValueError("max_groups must be >= base_groups")
        if not 0.0 < self.target_route_coverage <= 1.0:
            raise ValueError("target_route_coverage must be in (0, 1]")
        if self.top_score_buckets <= 0:
            raise ValueError("top_score_buckets must be positive")
        if self.base_score_buckets <= 0:
            raise ValueError("base_score_buckets must be positive")
        if self.gap_buckets <= 0:
            raise ValueError("gap_buckets must be positive")
        if len(self.exact_tie_bounds) == 0:
            raise ValueError("exact_tie_bounds must not be empty")
        if any(bound <= 0 for bound in self.exact_tie_bounds):
            raise ValueError("exact_tie_bounds must be positive")
        if tuple(sorted(self.exact_tie_bounds)) != self.exact_tie_bounds:
            raise ValueError("exact_tie_bounds must be sorted")
        if len(self.near_tie_bounds) == 0:
            raise ValueError("near_tie_bounds must not be empty")
        if any(bound <= 0 for bound in self.near_tie_bounds):
            raise ValueError("near_tie_bounds must be positive")
        if tuple(sorted(self.near_tie_bounds)) != self.near_tie_bounds:
            raise ValueError("near_tie_bounds must be sorted")
        if self.fanout_bits <= 0:
            raise ValueError("fanout_bits must be positive")
        expected = (
            self.top_score_buckets
            * self.base_score_buckets
            * self.gap_buckets
            * len(self.exact_tie_bounds)
            * len(self.near_tie_bounds)
        )
        if len(self.fanouts) != expected:
            raise ValueError("fanouts length does not match feature dimensions")
        for fanout in self.fanouts:
            if not self.base_groups <= int(fanout) <= self.max_groups:
                raise ValueError("fanout outside configured bounds")
        if self.training_examples < 0:
            raise ValueError("training_examples must be non-negative")

    @property
    def state_bytes(self) -> float:
        return len(self.fanouts) * self.fanout_bits / 8.0

    def predict(self, group_scores: np.ndarray, group_order: np.ndarray) -> int:
        if len(group_order) == 0:
            return self.base_groups
        index = _fanout_lut_index(
            group_scores=group_scores,
            group_order=group_order,
            base_groups=self.base_groups,
            top_score_buckets=self.top_score_buckets,
            base_score_buckets=self.base_score_buckets,
            gap_buckets=self.gap_buckets,
            exact_tie_bounds=self.exact_tie_bounds,
            near_tie_bounds=self.near_tie_bounds,
        )
        return int(self.fanouts[index])


@dataclass(frozen=True)
class WikiMemoryFanoutPoint:
    """One fixed or adaptive group-fanout route point."""

    route_label: str
    selected_groups: int
    adaptive_max_groups: int
    adaptive_score_margin: int
    ca_overall_recall: float
    flat_overall_recall: float
    ca_cluster_consistency_rate: float
    ca_cells_read_per_query: float
    flat_cells_read_per_query: float
    exact_scan_cells_per_query: float
    ca_cells_written_per_update: float
    ca_read_reduction_vs_flat: float
    ca_read_reduction_vs_exact_scan: float
    target_route_coverage: float = 0.0
    fanout_lut_state_bytes: float = 0.0
    fanout_training_examples: int = 0


@dataclass(frozen=True)
class WikiMemoryFanoutResult:
    """Fixed versus adaptive group-fanout sweep for dense pages."""

    policy: str
    page_count: int
    facts_per_page: int
    summary_width: int
    query_events: int
    update_events: int
    points: Tuple[WikiMemoryFanoutPoint, ...]


@dataclass(frozen=True)
class WikiMemoryLearnedFanoutGridPoint:
    """One learned fanout LUT point across page count and page density."""

    page_count: int
    facts_per_page: int
    summary_width: int
    fixed_overall_recall: float
    adaptive_overall_recall: float
    learned_overall_recall: float
    flat_overall_recall: float
    fixed_cells_read_per_query: float
    adaptive_cells_read_per_query: float
    learned_cells_read_per_query: float
    flat_cells_read_per_query: float
    exact_scan_cells_per_query: float
    learned_cells_written_per_update: float
    learned_read_reduction_vs_flat: float
    learned_read_reduction_vs_adaptive: float
    learned_read_reduction_vs_exact_scan: float
    fanout_lut_state_bytes: float
    fanout_training_examples: int


@dataclass(frozen=True)
class WikiMemoryLearnedFanoutGridResult:
    """Learned fanout LUT sweep across wiki size and page density."""

    policy: str
    target_route_coverage: float
    query_events: int
    update_events: int
    summary_banks: int
    summary_width: int
    summary_bits: int
    points: Tuple[WikiMemoryLearnedFanoutGridPoint, ...]


@dataclass(frozen=True)
class WikiMemoryDenseTilePoint:
    """One dense-page routing-tile comparison."""

    page_count: int
    facts_per_page: int
    summary_width: int
    baseline_group_size: int
    dense_group_size: int
    baseline_max_groups: int
    dense_max_groups: int
    baseline_overall_recall: float
    dense_overall_recall: float
    flat_overall_recall: float
    baseline_cells_read_per_query: float
    dense_cells_read_per_query: float
    flat_cells_read_per_query: float
    dense_cells_written_per_update: float
    baseline_state_bytes: float
    dense_state_bytes: float
    baseline_lut_state_bytes: float
    dense_lut_state_bytes: float
    dense_read_reduction_vs_flat: float
    dense_read_reduction_vs_baseline: float
    dense_state_increase_bytes: float
    dense_training_examples: int


@dataclass(frozen=True)
class WikiMemoryDenseTileResult:
    """Dense-page small-tile fanout comparison."""

    policy: str
    target_route_coverage: float
    query_events: int
    update_events: int
    summary_banks: int
    summary_width: int
    summary_bits: int
    points: Tuple[WikiMemoryDenseTilePoint, ...]


@dataclass(frozen=True)
class WikiMemoryDensityAwareTilePoint:
    """One mixed sparse/dense region tile-sizing point."""

    total_pages: int
    dense_page_fraction: float
    dense_query_fraction: float
    sparse_pages: int
    dense_pages: int
    sparse_facts_per_page: int
    dense_facts_per_page: int
    dense_tile_enabled: bool
    baseline_overall_recall: float
    aware_overall_recall: float
    all_dense_overall_recall: float
    flat_overall_recall: float
    baseline_cells_read_per_query: float
    aware_cells_read_per_query: float
    all_dense_cells_read_per_query: float
    flat_cells_read_per_query: float
    baseline_state_bytes: float
    aware_state_bytes: float
    all_dense_state_bytes: float
    density_tag_state_bytes: float
    aware_read_reduction_vs_flat: float
    aware_read_reduction_vs_baseline: float
    aware_state_increase_vs_baseline: float
    aware_state_saving_vs_all_dense: float
    aware_training_examples: int


@dataclass(frozen=True)
class WikiMemoryDensityAwareTileResult:
    """Density-aware mixed-region tile-sizing sweep."""

    policy: str
    target_route_coverage: float
    query_events: int
    update_events: int
    summary_banks: int
    summary_width: int
    summary_bits: int
    region_directory_cells_per_query: int
    points: Tuple[WikiMemoryDensityAwareTilePoint, ...]


@dataclass(frozen=True)
class WikiMemoryDensityTagPoint:
    """One refresh-derived density-tag tile decision point."""

    total_pages: int
    dense_page_fraction: float
    tag_threshold: int
    sparse_density_tag: int
    dense_density_tag: int
    tag_dense_enabled: bool
    guard_dense_enabled: bool
    baseline_overall_recall: float
    tag_only_overall_recall: float
    guarded_overall_recall: float
    flat_overall_recall: float
    baseline_cells_read_per_query: float
    tag_only_cells_read_per_query: float
    guarded_cells_read_per_query: float
    flat_cells_read_per_query: float
    tag_only_read_reduction_vs_flat: float
    guarded_read_reduction_vs_flat: float
    tag_only_state_bytes: float
    guarded_state_bytes: float
    baseline_state_bytes: float
    density_tag_state_bytes: float
    sparse_probe_baseline_recall: float
    sparse_probe_dense_recall: float
    dense_probe_baseline_recall: float
    dense_probe_dense_recall: float


@dataclass(frozen=True)
class WikiMemoryDensityTagResult:
    """Refresh-derived density tag threshold sweep."""

    policy: str
    target_route_coverage: float
    query_events: int
    update_events: int
    summary_banks: int
    summary_width: int
    summary_bits: int
    density_tag_bits: int
    region_directory_cells_per_query: int
    quality_probe_queries: int
    quality_probe_updates: int
    quality_probe_min_gain: float
    points: Tuple[WikiMemoryDensityTagPoint, ...]


@dataclass(frozen=True)
class _RouteResult:
    found: bool
    cells_read: int
    selected_pages: Tuple[int, ...]
    source_page: int | None


@dataclass(frozen=True)
class _WikiQuery:
    kind: str
    route_key: int
    target: int
    source_page: int
    source_slot: int
    target_page: int
    target_slot: int
    recent: bool
    cluster_id: int = -1


@dataclass(frozen=True)
class _AnswerResult:
    hit: bool
    stale: bool
    precise: bool
    cells_read: int
    route_found: bool


class _SyntheticWikiMemory:
    """Mutable synthetic wiki backed by low-bit page and group summaries."""

    def __init__(self, config: WikiMemoryConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.fact_keys = np.zeros((config.page_count, config.facts_per_page), dtype=np.int64)
        self.fact_values = np.zeros_like(self.fact_keys)
        self.truth_fact_keys = np.zeros_like(self.fact_keys)
        self.truth_fact_values = np.zeros_like(self.fact_values)
        self.cluster_by_page_slot = np.full(
            (config.page_count, config.facts_per_page),
            -1,
            dtype=np.int32,
        )
        self.cluster_pages = np.zeros(
            (config.contradiction_clusters, config.cluster_sources),
            dtype=np.int32,
        )
        self.cluster_slots = np.zeros_like(self.cluster_pages)
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
        self.recent_clusters: List[int] = []
        self._initialize_wiki()
        self._refresh_pages(np.arange(config.page_count, dtype=np.int32))
        self._refresh_groups(np.arange(config.group_count, dtype=np.int32))

    def _initialize_wiki(self) -> None:
        for page in range(self.config.page_count):
            for slot in range(self.config.facts_per_page):
                key = int(keyed_hash(page * 1009 + slot, 17) & ((1 << 31) - 1))
                self.fact_keys[page, slot] = key
                self.fact_values[page, slot] = self._value_for_key(key)
                self.truth_fact_keys[page, slot] = self.fact_keys[page, slot]
                self.truth_fact_values[page, slot] = self.fact_values[page, slot]

        for cluster in range(self.config.contradiction_clusters):
            key = int(keyed_hash(5_000_003 + cluster, 57) & ((1 << 31) - 1))
            value = int(keyed_hash(6_000_003 + cluster, 59) & ((1 << 31) - 1))
            for source in range(self.config.cluster_sources):
                page = cluster * self.config.cluster_sources + source
                slot = 0
                self.cluster_pages[cluster, source] = page
                self.cluster_slots[cluster, source] = slot
                self.cluster_by_page_slot[page, slot] = cluster
                self.fact_keys[page, slot] = key
                self.fact_values[page, slot] = value
                self.truth_fact_keys[page, slot] = key
                self.truth_fact_values[page, slot] = value

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
            self.fact_keys[page] = self.truth_fact_keys[page]
            self.fact_values[page] = self.truth_fact_values[page]
            self.page_summary[page].fill(0)
            touched += self.config.summary_banks * self.config.summary_width
            touched += self.config.facts_per_page * 2
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

    def _mark_page_dirty(self, page: int) -> None:
        self.page_versions[page] += 1
        self.dirty_pages[page] = True
        self.dirty_groups[self._group_for_page(page)] = True
        self.recent_pages.append(page)

    def _sample_non_cluster_fact(self) -> Tuple[int, int]:
        for _ in range(64):
            page = int(self.rng.integers(0, self.config.page_count))
            slot = int(self.rng.integers(0, self.config.facts_per_page))
            if int(self.cluster_by_page_slot[page, slot]) < 0:
                return page, slot
        free_slots = np.argwhere(self.cluster_by_page_slot < 0)
        if len(free_slots) == 0:
            raise RuntimeError("no non-cluster fact slots available")
        index = int(self.rng.integers(0, len(free_slots)))
        return int(free_slots[index, 0]), int(free_slots[index, 1])

    def _sample_non_cluster_slot_for_page(self, page: int) -> int:
        slots = np.flatnonzero(self.cluster_by_page_slot[page] < 0)
        if len(slots) == 0:
            return int(self.rng.integers(0, self.config.facts_per_page))
        return int(slots[int(self.rng.integers(0, len(slots)))])

    def _update_cluster(self) -> Tuple[int, str]:
        cluster = int(self.rng.integers(0, self.config.contradiction_clusters))
        key = int(self.truth_fact_keys[self.cluster_pages[cluster, 0], self.cluster_slots[cluster, 0]])
        new_value = int(
            keyed_hash(7_000_003 + self.update_cursor * 65537 + key * 197, 61)
            & ((1 << 31) - 1)
        )
        for page, slot in zip(self.cluster_pages[cluster], self.cluster_slots[cluster]):
            self.truth_fact_values[int(page), int(slot)] = new_value
            self._mark_page_dirty(int(page))
        self.recent_clusters.append(cluster)
        if len(self.recent_clusters) > 64:
            self.recent_clusters = self.recent_clusters[-64:]
        return self.config.cluster_sources * 4, "cluster"

    def update_fact(self, policy: WikiMemoryRefreshPolicy) -> Tuple[int, int, int, bool, str]:
        if (
            self.config.contradiction_clusters > 0
            and self.rng.random() < self.config.cluster_update_rate
        ):
            update_cells, update_kind = self._update_cluster()
        else:
            page, slot = self._sample_non_cluster_fact()
            if self.rng.random() < self.config.revision_update_rate:
                key = int(self.truth_fact_keys[page, slot])
                new_value = int(
                    keyed_hash(2_000_003 + self.update_cursor * 65537 + key * 131, 43)
                    & ((1 << 31) - 1)
                )
                self.truth_fact_values[page, slot] = new_value
                update_kind = "revision"
            else:
                new_key = int(
                    keyed_hash(1_000_003 + self.update_cursor * 65537 + page * 257 + slot, 29)
                )
                new_key &= (1 << 31) - 1
                self.truth_fact_keys[page, slot] = new_key
                self.truth_fact_values[page, slot] = self._value_for_key(new_key)
                update_kind = "key"
            self._mark_page_dirty(page)
            update_cells = 4
        self.summary_age += 1
        self.update_cursor += 1
        if len(self.recent_pages) > 64:
            self.recent_pages = self.recent_pages[-64:]

        if policy.refresh_on_update:
            refresh_cells, pages, groups = self._refresh_dirty()
            return update_cells + refresh_cells, pages, groups, refresh_cells > 0, update_kind
        return update_cells, 0, 0, False, update_kind

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

    def _hierarchical_route_from_groups(
        self,
        key: int,
        group_indices: np.ndarray,
        group_scan_cells: int,
    ) -> _RouteResult:
        candidate_pages: List[int] = []
        for group in group_indices:
            start = int(group) * self.config.group_size
            candidate_pages.extend(range(start, start + self.config.group_size))
        candidate_array = np.array(candidate_pages, dtype=np.int32)
        page_scores = self._score_pages(key, candidate_array)
        page_local = self._top_indices(page_scores, self.config.selected_pages)
        selected_pages = tuple(int(candidate_array[index]) for index in page_local)
        cells_read = (
            group_scan_cells
            + len(candidate_array) * self.config.summary_banks
            + len(selected_pages) * self.config.facts_per_page
        )
        for page in selected_pages:
            if bool(np.any(self.fact_keys[page] == int(key))):
                return _RouteResult(True, cells_read, selected_pages, page)
        return _RouteResult(False, cells_read, selected_pages, None)

    def route_key(self, key: int) -> _RouteResult:
        group_scores = self._score_groups(key)
        group_local = self._top_indices(group_scores, self.config.selected_groups)
        group_scan_cells = self.config.group_count * self.config.summary_banks
        return self._hierarchical_route_from_groups(key, group_local, group_scan_cells)

    def adaptive_route_key(self, key: int) -> _RouteResult:
        group_scores = self._score_groups(key)
        group_scan_cells = self.config.group_count * self.config.summary_banks
        group_order = self._top_indices(group_scores, len(group_scores))
        if len(group_order) == 0:
            return self._hierarchical_route_from_groups(key, group_order, group_scan_cells)

        base_count = min(self.config.selected_groups, len(group_order))
        max_count = min(self.config.adaptive_max_groups, len(group_order))
        kth_score = int(group_scores[group_order[base_count - 1]])
        top_score = int(group_scores[group_order[0]])
        if top_score <= 0:
            selected_count = base_count
        else:
            threshold = max(1, kth_score - self.config.adaptive_score_margin)
            near_tie_count = int(np.count_nonzero(group_scores[group_order] >= threshold))
            selected_count = min(max_count, max(base_count, near_tie_count))
        return self._hierarchical_route_from_groups(
            key,
            group_order[:selected_count],
            group_scan_cells,
        )

    def lut_route_key(self, key: int, fanout_lut: WikiMemoryFanoutLUT) -> _RouteResult:
        group_scores = self._score_groups(key)
        group_scan_cells = self.config.group_count * self.config.summary_banks
        group_order = self._top_indices(group_scores, len(group_scores))
        selected_count = min(
            len(group_order),
            max(self.config.selected_groups, fanout_lut.predict(group_scores, group_order)),
        )
        return self._hierarchical_route_from_groups(
            key,
            group_order[:selected_count],
            group_scan_cells,
        )

    def flat_route_key(self, key: int) -> _RouteResult:
        page_array = np.arange(self.config.page_count, dtype=np.int32)
        page_scores = self._score_pages(key, page_array)
        page_local = self._top_indices(page_scores, self.config.selected_pages)
        selected_pages = tuple(int(page_array[index]) for index in page_local)
        cells_read = (
            self.config.page_count * self.config.summary_banks
            + len(selected_pages) * self.config.facts_per_page
        )
        for page in selected_pages:
            if bool(np.any(self.fact_keys[page] == int(key))):
                return _RouteResult(True, cells_read, selected_pages, page)
        return _RouteResult(False, cells_read, selected_pages, None)

    def repair_pages(self, pages: Tuple[int, ...]) -> Tuple[int, int, int]:
        clean_pages = tuple(sorted(set(int(page) for page in pages)))
        if len(clean_pages) == 0:
            return 0, 0, 0
        page_array = np.array(clean_pages, dtype=np.int32)
        page_cells = self._refresh_pages(page_array)
        groups = tuple(sorted({self._group_for_page(page) for page in clean_pages}))
        group_cells = self._refresh_groups(np.array(groups, dtype=np.int32))
        return page_cells + group_cells, len(clean_pages), len(groups)

    def repair_query_pages(
        self, query: _WikiQuery, policy: WikiMemoryRefreshPolicy
    ) -> Tuple[int, int, int, bool]:
        if policy.cluster_repair and query.cluster_id >= 0:
            pages = tuple(int(page) for page in self.cluster_pages[query.cluster_id])
            cells, page_count, group_count = self.repair_pages(pages)
            return cells, page_count, group_count, True
        cells, page_count, group_count = self.repair_pages(
            (query.source_page, query.target_page)
        )
        return cells, page_count, group_count, False

    def _make_cluster_query(self, cluster: int, page: int | None = None) -> _WikiQuery:
        source_index = int(self.rng.integers(0, self.config.cluster_sources))
        if page is None:
            page = int(self.cluster_pages[cluster, source_index])
            slot = int(self.cluster_slots[cluster, source_index])
        else:
            matches = np.flatnonzero(self.cluster_pages[cluster] == int(page))
            source_index = int(matches[0]) if len(matches) else source_index
            slot = int(self.cluster_slots[cluster, source_index])
        return _WikiQuery(
            kind="single",
            route_key=int(self.truth_fact_keys[int(page), slot]),
            target=int(self.truth_fact_values[int(page), slot]),
            source_page=int(page),
            source_slot=slot,
            target_page=int(page),
            target_slot=slot,
            recent=cluster in self.recent_clusters,
            cluster_id=cluster,
        )

    def sample_query(self) -> _WikiQuery:
        if (
            self.config.contradiction_clusters > 0
            and self.rng.random() < self.config.cluster_query_rate
        ):
            if len(self.recent_clusters) > 0 and self.rng.random() < self.config.recent_update_query_rate:
                cluster = int(self.recent_clusters[int(self.rng.integers(0, len(self.recent_clusters)))])
            else:
                cluster = int(self.rng.integers(0, self.config.contradiction_clusters))
            return self._make_cluster_query(cluster)

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
            slot = self._sample_non_cluster_slot_for_page(page)
            target_page = int(self.links[page, int(self.rng.integers(0, self.config.links_per_page))])
            target_slot = int(self.rng.integers(0, self.config.facts_per_page))
            return _WikiQuery(
                kind="multihop",
                route_key=int(self.truth_fact_keys[page, slot]),
                target=int(self.truth_fact_keys[target_page, target_slot]),
                source_page=page,
                source_slot=slot,
                target_page=target_page,
                target_slot=target_slot,
                recent=use_recent,
            )
        cluster_id = int(self.cluster_by_page_slot[page, slot])
        return _WikiQuery(
            kind="single",
            route_key=int(self.truth_fact_keys[page, slot]),
            target=int(self.truth_fact_values[page, slot]),
            source_page=page,
            source_slot=slot,
            target_page=page,
            target_slot=slot,
            recent=use_recent,
            cluster_id=cluster_id,
        )

    def refresh_query_target(self, query: _WikiQuery) -> _WikiQuery:
        if query.kind == "multihop":
            target = int(self.truth_fact_keys[query.target_page, query.target_slot])
        else:
            target = int(self.truth_fact_values[query.target_page, query.target_slot])
        return _WikiQuery(
            kind=query.kind,
            route_key=int(self.truth_fact_keys[query.source_page, query.source_slot]),
            target=target,
            source_page=query.source_page,
            source_slot=query.source_slot,
            target_page=query.target_page,
            target_slot=query.target_slot,
            recent=query.recent,
            cluster_id=query.cluster_id,
        )

    def refresh_error_probe_query(self, query: _WikiQuery) -> _WikiQuery:
        if query.cluster_id >= 0:
            return self._make_cluster_query(query.cluster_id)
        return self.refresh_query_target(query)

    def cluster_consistent(self, cluster: int) -> bool:
        pages = self.cluster_pages[cluster]
        slots = self.cluster_slots[cluster]
        target = int(self.truth_fact_values[int(pages[0]), int(slots[0])])
        for page, slot in zip(pages, slots):
            if bool(self.dirty_pages[int(page)]):
                return False
            if int(self.fact_values[int(page), int(slot)]) != target:
                return False
        return True

    def _query_stale(self, query: _WikiQuery) -> bool:
        if query.cluster_id >= 0:
            pages = self.cluster_pages[query.cluster_id]
            return bool(np.any(self.dirty_pages[pages]))
        return bool(self.dirty_pages[query.source_page] or self.dirty_pages[query.target_page])

    def answer_query(
        self,
        query: _WikiQuery,
        route_mode: str = "hierarchical",
        fanout_lut: WikiMemoryFanoutLUT | None = None,
    ) -> _AnswerResult:
        if route_mode == "hierarchical":
            routed = self.route_key(query.route_key)
        elif route_mode == "adaptive":
            routed = self.adaptive_route_key(query.route_key)
        elif route_mode == "lut":
            if fanout_lut is None:
                raise ValueError("fanout_lut is required for route_mode=lut")
            routed = self.lut_route_key(query.route_key, fanout_lut)
        elif route_mode == "flat":
            routed = self.flat_route_key(query.route_key)
        else:
            raise ValueError("route_mode must be hierarchical, adaptive, lut, or flat")
        cells_read = routed.cells_read
        if not routed.found:
            return _AnswerResult(False, self._query_stale(query), False, cells_read, False)

        if query.kind == "single":
            page = int(routed.source_page) if routed.source_page is not None else -1
            if query.cluster_id >= 0:
                pages = self.cluster_pages[query.cluster_id]
                found = bool(
                    page in set(int(item) for item in pages)
                    and np.any(self.fact_values[page] == int(query.target))
                )
            else:
                found = bool(
                    page == query.target_page and np.any(self.fact_values[page] == int(query.target))
                )
            stale = bool((not found) and self._query_stale(query))
            return _AnswerResult(found, stale, found, cells_read, True)

        cells_read += self.config.links_per_page
        link_pages = self.links[int(routed.source_page)]
        cells_read += self.config.links_per_page * self.config.facts_per_page
        for page in link_pages:
            if int(page) == query.target_page and bool(
                np.any(self.fact_keys[page] == int(query.target))
            ):
                return _AnswerResult(True, False, True, cells_read, True)
        return _AnswerResult(False, self._query_stale(query), False, cells_read, True)


def _trial(
    policy: WikiMemoryRefreshPolicy,
    config: WikiMemoryConfig,
    seed: int,
    route_mode: str = "hierarchical",
    fanout_lut: WikiMemoryFanoutLUT | None = None,
) -> WikiMemoryTrialPoint:
    wiki = _SyntheticWikiMemory(config, seed)
    event_types = np.array(["query"] * config.query_events + ["update"] * config.update_events)
    wiki.rng.shuffle(event_types)

    queries = 0
    updates = 0
    single_queries = 0
    multihop_queries = 0
    cluster_queries = 0
    single_hits = 0
    multihop_hits = 0
    cluster_hits = 0
    recent_queries = 0
    recent_hits = 0
    stale_misses = 0
    route_misses = 0
    value_misses = 0
    provenance_hits = 0
    total_cells_read = 0
    total_flat_cells_read = 0
    total_cells_written = 0
    refresh_events = 0
    pages_refreshed = 0
    groups_refreshed = 0
    error_repairs = 0
    error_recoveries = 0
    error_probe_queries = 0
    error_probe_hits = 0
    cluster_consistency_hits = 0
    key_updates = 0
    revision_updates = 0
    cluster_updates = 0
    cluster_repair_events = 0
    error_book: List[_WikiQuery] = []

    for event_type in event_types:
        if event_type == "update":
            cells, pages, groups, refreshed, update_kind = wiki.update_fact(policy)
            total_cells_written += cells
            updates += 1
            key_updates += int(update_kind == "key")
            revision_updates += int(update_kind == "revision")
            cluster_updates += int(update_kind == "cluster")
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

        is_error_probe = (
            len(error_book) > 0 and wiki.rng.random() < config.error_probe_query_rate
        )
        if is_error_probe:
            error_probe_queries += 1
            query = wiki.refresh_error_probe_query(
                error_book[int(wiki.rng.integers(0, len(error_book)))]
            )
        else:
            query = wiki.sample_query()
        answer = wiki.answer_query(query, route_mode=route_mode, fanout_lut=fanout_lut)
        queries += 1
        total_cells_read += answer.cells_read
        total_flat_cells_read += config.page_count * config.facts_per_page
        if query.kind == "single":
            single_queries += 1
            single_hits += int(answer.hit)
        else:
            multihop_queries += 1
            multihop_hits += int(answer.hit)
        recent_queries += int(query.recent)
        recent_hits += int(query.recent and answer.hit)
        cluster_queries += int(query.cluster_id >= 0)
        cluster_hits += int(query.cluster_id >= 0 and answer.hit)
        stale_misses += int((not answer.hit) and answer.stale)
        route_misses += int(not answer.route_found)
        value_misses += int(answer.route_found and not answer.hit)
        provenance_hits += int(answer.hit and answer.precise)
        error_probe_hits += int(is_error_probe and answer.hit)

        recovered = False
        if (not answer.hit) and policy.error_book_repair:
            repair_cells, repair_pages, repair_groups, cluster_repair = wiki.repair_query_pages(
                query,
                policy,
            )
            total_cells_written += repair_cells
            refresh_events += 1
            pages_refreshed += repair_pages
            groups_refreshed += repair_groups
            error_repairs += 1
            cluster_repair_events += int(cluster_repair)
            repaired = wiki.answer_query(query, route_mode=route_mode, fanout_lut=fanout_lut)
            recovered = repaired.hit
            error_recoveries += int(recovered)

        if query.cluster_id >= 0 and policy.cluster_repair and not wiki.cluster_consistent(
            query.cluster_id
        ):
            repair_cells, repair_pages, repair_groups = wiki.repair_pages(
                tuple(int(page) for page in wiki.cluster_pages[query.cluster_id])
            )
            total_cells_written += repair_cells
            refresh_events += 1
            pages_refreshed += repair_pages
            groups_refreshed += repair_groups
            cluster_repair_events += 1

        if query.cluster_id >= 0:
            cluster_consistency_hits += int(wiki.cluster_consistent(query.cluster_id))

        if not answer.hit:
            error_book.append(query)
            if len(error_book) > 128:
                error_book = error_book[-128:]

    overall_hits = single_hits + multihop_hits
    cells_read_per_query = total_cells_read / float(queries)
    flat_cells_read_per_query = total_flat_cells_read / float(queries)
    cells_written_per_update = (
        total_cells_written / float(updates) if updates > 0 else 0.0
    )
    return WikiMemoryTrialPoint(
        policy=policy.name,
        route_mode=route_mode,
        dirty_threshold=policy.dirty_threshold,
        max_age=policy.max_age,
        refresh_on_update=policy.refresh_on_update,
        error_book_repair=policy.error_book_repair,
        cluster_repair=policy.cluster_repair,
        queries=queries,
        updates=updates,
        single_hop_recall=single_hits / float(single_queries) if single_queries else 0.0,
        multihop_recall=multihop_hits / float(multihop_queries) if multihop_queries else 0.0,
        overall_recall=overall_hits / float(queries),
        recent_update_recall=recent_hits / float(recent_queries) if recent_queries else 0.0,
        stale_miss_rate=stale_misses / float(queries),
        route_miss_rate=route_misses / float(queries),
        value_miss_rate=value_misses / float(queries),
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
        error_probe_queries=error_probe_queries,
        error_probe_recall=(
            error_probe_hits / float(error_probe_queries) if error_probe_queries else 0.0
        ),
        cluster_queries=cluster_queries,
        cluster_recall=cluster_hits / float(cluster_queries) if cluster_queries else 0.0,
        cluster_consistency_rate=(
            cluster_consistency_hits / float(cluster_queries) if cluster_queries else 0.0
        ),
        key_updates=key_updates,
        revision_updates=revision_updates,
        cluster_updates=cluster_updates,
        cluster_repair_events=cluster_repair_events,
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
        WikiMemoryRefreshPolicy(
            "trigger16_age16_clusterbook",
            dirty_threshold=16,
            max_age=16,
            error_book_repair=True,
            cluster_repair=True,
        ),
        WikiMemoryRefreshPolicy("trigger32_age64", dirty_threshold=32, max_age=64),
        WikiMemoryRefreshPolicy("stale_no_refresh", dirty_threshold=1_000_000, max_age=0),
    ),
    flat_policies: Tuple[WikiMemoryRefreshPolicy, ...] = (
        WikiMemoryRefreshPolicy(
            "flat_exact_update",
            dirty_threshold=1,
            max_age=0,
            refresh_on_update=True,
        ),
        WikiMemoryRefreshPolicy(
            "flat_trigger16_clusterbook",
            dirty_threshold=16,
            max_age=16,
            error_book_repair=True,
            cluster_repair=True,
        ),
        WikiMemoryRefreshPolicy("flat_stale_no_refresh", dirty_threshold=1_000_000, max_age=0),
    ),
    seed: int = 91,
) -> WikiMemorySweepResult:
    """Run a synthetic mutable wiki-memory policy sweep."""

    sweep_config = config or WikiMemoryConfig()
    points = tuple(
        _trial(policy=policy, config=sweep_config, seed=seed, route_mode="hierarchical")
        for policy in policies
    )
    flat_points = tuple(
        _trial(policy=policy, config=sweep_config, seed=seed, route_mode="flat")
        for policy in flat_policies
    )
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
        revision_update_rate=sweep_config.revision_update_rate,
        error_probe_query_rate=sweep_config.error_probe_query_rate,
        contradiction_clusters=sweep_config.contradiction_clusters,
        cluster_sources=sweep_config.cluster_sources,
        cluster_update_rate=sweep_config.cluster_update_rate,
        cluster_query_rate=sweep_config.cluster_query_rate,
        state_bytes=sweep_config.state_bytes,
        points=points,
        flat_points=flat_points,
    )


def _scaling_config(page_count: int) -> WikiMemoryConfig:
    group_size = 16
    cluster_sources = 3
    contradiction_clusters = max(1, min(page_count // 8, page_count // cluster_sources))
    return WikiMemoryConfig(
        page_count=page_count,
        facts_per_page=4,
        topic_count=max(8, min(256, page_count // 4)),
        links_per_page=4,
        group_size=group_size,
        selected_groups=4,
        selected_pages=8,
        summary_banks=4,
        summary_width=256,
        summary_bits=4,
        query_events=512,
        update_events=256,
        contradiction_clusters=contradiction_clusters,
        cluster_sources=cluster_sources,
    )


def _density_config(
    page_count: int,
    facts_per_page: int,
    summary_width: int,
) -> WikiMemoryConfig:
    group_size = 16
    cluster_sources = 3
    contradiction_clusters = max(1, min(page_count // 8, page_count // cluster_sources))
    return WikiMemoryConfig(
        page_count=page_count,
        facts_per_page=facts_per_page,
        topic_count=max(8, min(256, page_count // 4)),
        links_per_page=4,
        group_size=group_size,
        selected_groups=4,
        selected_pages=8,
        summary_banks=4,
        summary_width=summary_width,
        summary_bits=4,
        query_events=512,
        update_events=256,
        contradiction_clusters=contradiction_clusters,
        cluster_sources=cluster_sources,
    )


def _fanout_near_tie_bucket(count: int, bounds: Tuple[int, ...]) -> int:
    for index, bound in enumerate(bounds):
        if count <= bound:
            return index
    return len(bounds) - 1


def _fanout_lut_index(
    group_scores: np.ndarray,
    group_order: np.ndarray,
    base_groups: int,
    top_score_buckets: int,
    base_score_buckets: int,
    gap_buckets: int,
    exact_tie_bounds: Tuple[int, ...],
    near_tie_bounds: Tuple[int, ...],
) -> int:
    if len(group_order) == 0:
        return 0
    base_count = min(base_groups, len(group_order))
    top_score = max(0, int(group_scores[group_order[0]]))
    base_score = max(0, int(group_scores[group_order[base_count - 1]]))
    score_gap = max(0, top_score - base_score)
    exact_tie_count = int(np.count_nonzero(group_scores[group_order] >= base_score))
    near_threshold = max(1, base_score - 1)
    near_tie_count = int(np.count_nonzero(group_scores[group_order] >= near_threshold))

    top_bucket = min(top_score, top_score_buckets - 1)
    base_bucket = min(base_score, base_score_buckets - 1)
    gap_bucket = min(score_gap, gap_buckets - 1)
    exact_bucket = _fanout_near_tie_bucket(exact_tie_count, exact_tie_bounds)
    near_bucket = _fanout_near_tie_bucket(near_tie_count, near_tie_bounds)
    return (
        (
            ((top_bucket * base_score_buckets + base_bucket) * gap_buckets + gap_bucket)
            * len(exact_tie_bounds)
            + exact_bucket
        )
        * len(near_tie_bounds)
        + near_bucket
    )


def _minimum_route_fanout(
    wiki: _SyntheticWikiMemory,
    key: int,
    group_order: np.ndarray,
    fanout_values: Tuple[int, ...],
) -> int | None:
    group_scan_cells = wiki.config.group_count * wiki.config.summary_banks
    for fanout in fanout_values:
        result = wiki._hierarchical_route_from_groups(
            key,
            group_order[: min(int(fanout), len(group_order))],
            group_scan_cells,
        )
        if result.found:
            return int(fanout)
    return None


def train_wiki_memory_fanout_lut(
    config: WikiMemoryConfig,
    policy: WikiMemoryRefreshPolicy,
    train_seeds: Tuple[int, ...] = _DEFAULT_FANOUT_TRAIN_SEEDS,
    base_groups: int = 4,
    max_groups: int = 32,
    target_route_coverage: float = 0.95,
    fanout_values: Tuple[int, ...] = (4, 8, 16, 32),
    top_score_buckets: int = 1,
    base_score_buckets: int = 8,
    gap_buckets: int = 4,
    exact_tie_bounds: Tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64),
    near_tie_bounds: Tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64),
) -> WikiMemoryFanoutLUT:
    """Train a low-bit fanout table from minimal-route self-supervision."""

    if base_groups <= 0:
        raise ValueError("base_groups must be positive")
    if max_groups < base_groups:
        raise ValueError("max_groups must be >= base_groups")
    clean_fanouts = tuple(
        value
        for value in sorted(set(int(item) for item in fanout_values))
        if base_groups <= value <= max_groups
    )
    if base_groups not in clean_fanouts:
        clean_fanouts = tuple(sorted(set(clean_fanouts + (base_groups,))))
    if max_groups not in clean_fanouts:
        clean_fanouts = tuple(sorted(set(clean_fanouts + (max_groups,))))
    if len(clean_fanouts) == 0:
        raise ValueError("fanout_values must include at least one usable value")

    table_size = (
        top_score_buckets
        * base_score_buckets
        * gap_buckets
        * len(exact_tie_bounds)
        * len(near_tie_bounds)
    )
    label_counts = np.zeros((table_size, len(clean_fanouts)), dtype=np.int32)
    label_index = {fanout: index for index, fanout in enumerate(clean_fanouts)}
    training_examples = 0
    training_config = replace(
        config,
        selected_groups=base_groups,
        adaptive_max_groups=max_groups,
        adaptive_score_margin=1,
    )

    for seed in train_seeds:
        wiki = _SyntheticWikiMemory(training_config, int(seed))
        event_types = np.array(
            ["query"] * training_config.query_events
            + ["update"] * training_config.update_events
        )
        wiki.rng.shuffle(event_types)
        error_book: List[_WikiQuery] = []

        for event_type in event_types:
            if event_type == "update":
                wiki.update_fact(policy)
                continue

            wiki.maybe_refresh(policy)
            if len(error_book) > 0 and wiki.rng.random() < training_config.error_probe_query_rate:
                query = wiki.refresh_error_probe_query(
                    error_book[int(wiki.rng.integers(0, len(error_book)))]
                )
            else:
                query = wiki.sample_query()

            group_scores = wiki._score_groups(query.route_key)
            group_order = wiki._top_indices(group_scores, len(group_scores))
            label = _minimum_route_fanout(
                wiki,
                query.route_key,
                group_order,
                clean_fanouts,
            )
            if label is not None:
                index = _fanout_lut_index(
                    group_scores=group_scores,
                    group_order=group_order,
                    base_groups=base_groups,
                    top_score_buckets=top_score_buckets,
                    base_score_buckets=base_score_buckets,
                    gap_buckets=gap_buckets,
                    exact_tie_bounds=exact_tie_bounds,
                    near_tie_bounds=near_tie_bounds,
                )
                label_counts[index, label_index[label]] += 1
                training_examples += 1

            answer = wiki.answer_query(query, route_mode="adaptive")
            if (not answer.hit) and policy.error_book_repair:
                wiki.repair_query_pages(query, policy)
            if (
                query.cluster_id >= 0
                and policy.cluster_repair
                and not wiki.cluster_consistent(query.cluster_id)
            ):
                wiki.repair_pages(tuple(int(page) for page in wiki.cluster_pages[query.cluster_id]))
            if not answer.hit:
                error_book.append(query)
                if len(error_book) > 128:
                    error_book = error_book[-128:]

    fanouts = np.full(table_size, max_groups, dtype=np.int32)
    for index, counts in enumerate(label_counts):
        total = int(np.sum(counts))
        if total == 0:
            continue
        cumulative = 0
        chosen = max_groups
        for fanout, count in zip(clean_fanouts, counts):
            cumulative += int(count)
            if cumulative / float(total) >= target_route_coverage:
                chosen = int(fanout)
                break
        fanouts[index] = chosen

    fanout_bits = max(1, (len(clean_fanouts) - 1).bit_length())
    return WikiMemoryFanoutLUT(
        base_groups=base_groups,
        max_groups=max_groups,
        target_route_coverage=target_route_coverage,
        top_score_buckets=top_score_buckets,
        base_score_buckets=base_score_buckets,
        gap_buckets=gap_buckets,
        exact_tie_bounds=exact_tie_bounds,
        near_tie_bounds=near_tie_bounds,
        fanout_bits=fanout_bits,
        fanouts=tuple(int(value) for value in fanouts),
        training_examples=training_examples,
        train_seeds=tuple(int(seed) for seed in train_seeds),
    )


def run_wiki_memory_scaling_sweep(
    page_counts: Tuple[int, ...] = (256, 512, 1024, 2048),
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryScalingResult:
    """Compare hierarchical CA routing with flat page-summary scans as wiki grows."""

    clean_counts = tuple(dict.fromkeys(int(count) for count in page_counts))
    if len(clean_counts) == 0:
        raise ValueError("page_counts must not be empty")

    points = []
    for page_count in clean_counts:
        config = _scaling_config(page_count)
        ca_point = _trial(
            policy=policy,
            config=config,
            seed=seed,
            route_mode="hierarchical",
        )
        flat_point = _trial(
            policy=policy,
            config=config,
            seed=seed,
            route_mode="flat",
        )
        exact_scan = float(config.page_count * config.facts_per_page)
        points.append(
            WikiMemoryScalingPoint(
                page_count=config.page_count,
                facts_per_page=config.facts_per_page,
                group_size=config.group_size,
                selected_groups=config.selected_groups,
                selected_pages=config.selected_pages,
                contradiction_clusters=config.contradiction_clusters,
                state_bytes=config.state_bytes,
                ca_overall_recall=ca_point.overall_recall,
                flat_overall_recall=flat_point.overall_recall,
                ca_cluster_consistency_rate=ca_point.cluster_consistency_rate,
                flat_cluster_consistency_rate=flat_point.cluster_consistency_rate,
                ca_cells_read_per_query=ca_point.cells_read_per_query,
                flat_cells_read_per_query=flat_point.cells_read_per_query,
                exact_scan_cells_per_query=exact_scan,
                ca_cells_written_per_update=ca_point.cells_written_per_update,
                flat_cells_written_per_update=flat_point.cells_written_per_update,
                ca_read_reduction_vs_flat=(
                    1.0 - ca_point.cells_read_per_query / flat_point.cells_read_per_query
                    if flat_point.cells_read_per_query > 0.0
                    else 0.0
                ),
                ca_read_reduction_vs_exact_scan=(
                    1.0 - ca_point.cells_read_per_query / exact_scan
                    if exact_scan > 0.0
                    else 0.0
                ),
            )
        )

    first_config = _scaling_config(clean_counts[0])
    return WikiMemoryScalingResult(
        policy=policy.name,
        query_events=first_config.query_events,
        update_events=first_config.update_events,
        summary_banks=first_config.summary_banks,
        summary_width=first_config.summary_width,
        summary_bits=first_config.summary_bits,
        points=tuple(points),
    )


def run_wiki_memory_density_sweep(
    page_count: int = 1024,
    facts_per_page_values: Tuple[int, ...] = (4, 8, 16, 32),
    summary_width_values: Tuple[int, ...] = (128, 256),
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryDensityResult:
    """Pressure-test page density and low-bit summary width."""

    clean_facts = tuple(dict.fromkeys(int(value) for value in facts_per_page_values))
    clean_widths = tuple(dict.fromkeys(int(value) for value in summary_width_values))
    if len(clean_facts) == 0:
        raise ValueError("facts_per_page_values must not be empty")
    if len(clean_widths) == 0:
        raise ValueError("summary_width_values must not be empty")

    points = []
    for summary_width in clean_widths:
        for facts_per_page in clean_facts:
            config = _density_config(
                page_count=page_count,
                facts_per_page=facts_per_page,
                summary_width=summary_width,
            )
            ca_point = _trial(
                policy=policy,
                config=config,
                seed=seed,
                route_mode="hierarchical",
            )
            flat_point = _trial(
                policy=policy,
                config=config,
                seed=seed,
                route_mode="flat",
            )
            exact_scan = float(config.page_count * config.facts_per_page)
            points.append(
                WikiMemoryDensityPoint(
                    page_count=config.page_count,
                    facts_per_page=config.facts_per_page,
                    summary_width=config.summary_width,
                    contradiction_clusters=config.contradiction_clusters,
                    state_bytes=config.state_bytes,
                    ca_overall_recall=ca_point.overall_recall,
                    flat_overall_recall=flat_point.overall_recall,
                    ca_cluster_consistency_rate=ca_point.cluster_consistency_rate,
                    flat_cluster_consistency_rate=flat_point.cluster_consistency_rate,
                    ca_cells_read_per_query=ca_point.cells_read_per_query,
                    flat_cells_read_per_query=flat_point.cells_read_per_query,
                    exact_scan_cells_per_query=exact_scan,
                    ca_cells_written_per_update=ca_point.cells_written_per_update,
                    flat_cells_written_per_update=flat_point.cells_written_per_update,
                    ca_read_reduction_vs_flat=(
                        1.0 - ca_point.cells_read_per_query / flat_point.cells_read_per_query
                        if flat_point.cells_read_per_query > 0.0
                        else 0.0
                    ),
                    ca_read_reduction_vs_exact_scan=(
                        1.0 - ca_point.cells_read_per_query / exact_scan
                        if exact_scan > 0.0
                        else 0.0
                    ),
                )
            )

    first_config = _density_config(page_count, clean_facts[0], clean_widths[0])
    return WikiMemoryDensityResult(
        policy=policy.name,
        page_count=page_count,
        query_events=first_config.query_events,
        update_events=first_config.update_events,
        summary_banks=first_config.summary_banks,
        summary_bits=first_config.summary_bits,
        points=tuple(points),
    )


def run_wiki_memory_fanout_sweep(
    page_count: int = 1024,
    facts_per_page: int = 16,
    summary_width: int = 256,
    fixed_group_values: Tuple[int, ...] = (4, 8, 16, 32),
    adaptive_settings: Tuple[Tuple[int, int, int], ...] = (
        (4, 16, 0),
        (4, 16, 1),
        (4, 32, 0),
        (4, 32, 1),
    ),
    learned_targets: Tuple[float, ...] = (0.98, 1.0),
    train_seeds: Tuple[int, ...] = _DEFAULT_FANOUT_TRAIN_SEEDS,
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryFanoutResult:
    """Compare fixed group fanout with adaptive near-tie fanout."""

    points = []
    flat_config = _density_config(page_count, facts_per_page, summary_width)
    flat_point = _trial(
        policy=policy,
        config=flat_config,
        seed=seed,
        route_mode="flat",
    )
    exact_scan = float(flat_config.page_count * flat_config.facts_per_page)

    for selected_groups in tuple(dict.fromkeys(int(value) for value in fixed_group_values)):
        base_config = _density_config(page_count, facts_per_page, summary_width)
        config = replace(
            base_config,
            selected_groups=selected_groups,
            adaptive_max_groups=max(selected_groups, base_config.adaptive_max_groups),
        )
        ca_point = _trial(
            policy=policy,
            config=config,
            seed=seed,
            route_mode="hierarchical",
        )
        points.append(
            WikiMemoryFanoutPoint(
                route_label=f"fixed_g{selected_groups}",
                selected_groups=selected_groups,
                adaptive_max_groups=selected_groups,
                adaptive_score_margin=0,
                ca_overall_recall=ca_point.overall_recall,
                flat_overall_recall=flat_point.overall_recall,
                ca_cluster_consistency_rate=ca_point.cluster_consistency_rate,
                ca_cells_read_per_query=ca_point.cells_read_per_query,
                flat_cells_read_per_query=flat_point.cells_read_per_query,
                exact_scan_cells_per_query=exact_scan,
                ca_cells_written_per_update=ca_point.cells_written_per_update,
                ca_read_reduction_vs_flat=(
                    1.0 - ca_point.cells_read_per_query / flat_point.cells_read_per_query
                    if flat_point.cells_read_per_query > 0.0
                    else 0.0
                ),
                ca_read_reduction_vs_exact_scan=(
                    1.0 - ca_point.cells_read_per_query / exact_scan
                    if exact_scan > 0.0
                    else 0.0
                ),
            )
        )

    for base_groups, max_groups, margin in tuple(
        dict.fromkeys(
            (int(base), int(maximum), int(score_margin))
            for base, maximum, score_margin in adaptive_settings
        )
    ):
        base_config = _density_config(page_count, facts_per_page, summary_width)
        config = replace(
            base_config,
            selected_groups=base_groups,
            adaptive_max_groups=max_groups,
            adaptive_score_margin=margin,
        )
        ca_point = _trial(
            policy=policy,
            config=config,
            seed=seed,
            route_mode="adaptive",
        )
        points.append(
            WikiMemoryFanoutPoint(
                route_label=f"adaptive_g{base_groups}_max{max_groups}_m{margin}",
                selected_groups=base_groups,
                adaptive_max_groups=max_groups,
                adaptive_score_margin=margin,
                ca_overall_recall=ca_point.overall_recall,
                flat_overall_recall=flat_point.overall_recall,
                ca_cluster_consistency_rate=ca_point.cluster_consistency_rate,
                ca_cells_read_per_query=ca_point.cells_read_per_query,
                flat_cells_read_per_query=flat_point.cells_read_per_query,
                exact_scan_cells_per_query=exact_scan,
                ca_cells_written_per_update=ca_point.cells_written_per_update,
                ca_read_reduction_vs_flat=(
                    1.0 - ca_point.cells_read_per_query / flat_point.cells_read_per_query
                    if flat_point.cells_read_per_query > 0.0
                    else 0.0
                ),
                ca_read_reduction_vs_exact_scan=(
                    1.0 - ca_point.cells_read_per_query / exact_scan
                    if exact_scan > 0.0
                    else 0.0
                ),
            )
        )

    for target in tuple(dict.fromkeys(float(value) for value in learned_targets)):
        base_groups = 4
        max_groups = 32
        config = replace(
            _density_config(page_count, facts_per_page, summary_width),
            selected_groups=base_groups,
            adaptive_max_groups=max_groups,
            adaptive_score_margin=1,
        )
        fanout_lut = train_wiki_memory_fanout_lut(
            config=config,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=base_groups,
            max_groups=max_groups,
            target_route_coverage=target,
        )
        ca_point = _trial(
            policy=policy,
            config=config,
            seed=seed,
            route_mode="lut",
            fanout_lut=fanout_lut,
        )
        points.append(
            WikiMemoryFanoutPoint(
                route_label=f"learned_lut_t{int(round(target * 100)):02d}",
                selected_groups=base_groups,
                adaptive_max_groups=max_groups,
                adaptive_score_margin=1,
                ca_overall_recall=ca_point.overall_recall,
                flat_overall_recall=flat_point.overall_recall,
                ca_cluster_consistency_rate=ca_point.cluster_consistency_rate,
                ca_cells_read_per_query=ca_point.cells_read_per_query,
                flat_cells_read_per_query=flat_point.cells_read_per_query,
                exact_scan_cells_per_query=exact_scan,
                ca_cells_written_per_update=ca_point.cells_written_per_update,
                ca_read_reduction_vs_flat=(
                    1.0 - ca_point.cells_read_per_query / flat_point.cells_read_per_query
                    if flat_point.cells_read_per_query > 0.0
                    else 0.0
                ),
                ca_read_reduction_vs_exact_scan=(
                    1.0 - ca_point.cells_read_per_query / exact_scan
                    if exact_scan > 0.0
                    else 0.0
                ),
                target_route_coverage=target,
                fanout_lut_state_bytes=fanout_lut.state_bytes,
                fanout_training_examples=fanout_lut.training_examples,
            )
        )

    return WikiMemoryFanoutResult(
        policy=policy.name,
        page_count=page_count,
        facts_per_page=facts_per_page,
        summary_width=summary_width,
        query_events=flat_config.query_events,
        update_events=flat_config.update_events,
        points=tuple(points),
    )


def run_wiki_memory_learned_fanout_grid_sweep(
    page_counts: Tuple[int, ...] = (512, 1024, 2048),
    facts_per_page_values: Tuple[int, ...] = (8, 16, 32),
    summary_width: int = 256,
    target_route_coverage: float = 1.0,
    train_seeds: Tuple[int, ...] = _DEFAULT_FANOUT_TRAIN_SEEDS,
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryLearnedFanoutGridResult:
    """Test learned fanout across wiki size and page-density pressure."""

    clean_counts = tuple(dict.fromkeys(int(value) for value in page_counts))
    clean_facts = tuple(dict.fromkeys(int(value) for value in facts_per_page_values))
    if len(clean_counts) == 0:
        raise ValueError("page_counts must not be empty")
    if len(clean_facts) == 0:
        raise ValueError("facts_per_page_values must not be empty")

    points = []
    for page_count in clean_counts:
        for facts_per_page in clean_facts:
            config = replace(
                _density_config(page_count, facts_per_page, summary_width),
                selected_groups=4,
                adaptive_max_groups=32,
                adaptive_score_margin=1,
            )
            flat_point = _trial(
                policy=policy,
                config=config,
                seed=seed,
                route_mode="flat",
            )
            fixed_point = _trial(
                policy=policy,
                config=config,
                seed=seed,
                route_mode="hierarchical",
            )
            adaptive_point = _trial(
                policy=policy,
                config=config,
                seed=seed,
                route_mode="adaptive",
            )
            fanout_lut = train_wiki_memory_fanout_lut(
                config=config,
                policy=policy,
                train_seeds=train_seeds,
                base_groups=4,
                max_groups=32,
                target_route_coverage=target_route_coverage,
            )
            learned_point = _trial(
                policy=policy,
                config=config,
                seed=seed,
                route_mode="lut",
                fanout_lut=fanout_lut,
            )
            exact_scan = float(config.page_count * config.facts_per_page)
            points.append(
                WikiMemoryLearnedFanoutGridPoint(
                    page_count=config.page_count,
                    facts_per_page=config.facts_per_page,
                    summary_width=config.summary_width,
                    fixed_overall_recall=fixed_point.overall_recall,
                    adaptive_overall_recall=adaptive_point.overall_recall,
                    learned_overall_recall=learned_point.overall_recall,
                    flat_overall_recall=flat_point.overall_recall,
                    fixed_cells_read_per_query=fixed_point.cells_read_per_query,
                    adaptive_cells_read_per_query=adaptive_point.cells_read_per_query,
                    learned_cells_read_per_query=learned_point.cells_read_per_query,
                    flat_cells_read_per_query=flat_point.cells_read_per_query,
                    exact_scan_cells_per_query=exact_scan,
                    learned_cells_written_per_update=learned_point.cells_written_per_update,
                    learned_read_reduction_vs_flat=(
                        1.0
                        - learned_point.cells_read_per_query / flat_point.cells_read_per_query
                        if flat_point.cells_read_per_query > 0.0
                        else 0.0
                    ),
                    learned_read_reduction_vs_adaptive=(
                        1.0
                        - learned_point.cells_read_per_query
                        / adaptive_point.cells_read_per_query
                        if adaptive_point.cells_read_per_query > 0.0
                        else 0.0
                    ),
                    learned_read_reduction_vs_exact_scan=(
                        1.0 - learned_point.cells_read_per_query / exact_scan
                        if exact_scan > 0.0
                        else 0.0
                    ),
                    fanout_lut_state_bytes=fanout_lut.state_bytes,
                    fanout_training_examples=fanout_lut.training_examples,
                )
            )

    first_config = _density_config(clean_counts[0], clean_facts[0], summary_width)
    return WikiMemoryLearnedFanoutGridResult(
        policy=policy.name,
        target_route_coverage=target_route_coverage,
        query_events=first_config.query_events,
        update_events=first_config.update_events,
        summary_banks=first_config.summary_banks,
        summary_width=first_config.summary_width,
        summary_bits=first_config.summary_bits,
        points=tuple(points),
    )


def run_wiki_memory_dense_tile_sweep(
    page_counts: Tuple[int, ...] = (1024, 2048),
    facts_per_page_values: Tuple[int, ...] = (16, 32),
    summary_width: int = 256,
    baseline_group_size: int = 16,
    dense_group_size: int = 4,
    baseline_max_groups: int = 32,
    dense_max_groups: int = 48,
    target_route_coverage: float = 1.0,
    train_seeds: Tuple[int, ...] = _DEFAULT_FANOUT_TRAIN_SEEDS,
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryDenseTileResult:
    """Compare standard and dense routing tiles for high-density wiki pages."""

    clean_counts = tuple(dict.fromkeys(int(value) for value in page_counts))
    clean_facts = tuple(dict.fromkeys(int(value) for value in facts_per_page_values))
    if len(clean_counts) == 0:
        raise ValueError("page_counts must not be empty")
    if len(clean_facts) == 0:
        raise ValueError("facts_per_page_values must not be empty")
    if baseline_group_size <= 0 or dense_group_size <= 0:
        raise ValueError("group sizes must be positive")
    if baseline_max_groups <= 0 or dense_max_groups <= 0:
        raise ValueError("max group counts must be positive")

    points = []
    for page_count in clean_counts:
        for facts_per_page in clean_facts:
            base_config = _density_config(page_count, facts_per_page, summary_width)
            baseline_config = replace(
                base_config,
                group_size=baseline_group_size,
                selected_groups=4,
                adaptive_max_groups=baseline_max_groups,
                adaptive_score_margin=1,
            )
            dense_config = replace(
                base_config,
                group_size=dense_group_size,
                selected_groups=4,
                adaptive_max_groups=dense_max_groups,
                adaptive_score_margin=1,
            )

            baseline_lut = train_wiki_memory_fanout_lut(
                config=baseline_config,
                policy=policy,
                train_seeds=train_seeds,
                base_groups=4,
                max_groups=baseline_max_groups,
                target_route_coverage=target_route_coverage,
            )
            dense_lut = train_wiki_memory_fanout_lut(
                config=dense_config,
                policy=policy,
                train_seeds=train_seeds,
                base_groups=4,
                max_groups=dense_max_groups,
                target_route_coverage=target_route_coverage,
            )
            baseline_point = _trial(
                policy=policy,
                config=baseline_config,
                seed=seed,
                route_mode="lut",
                fanout_lut=baseline_lut,
            )
            dense_point = _trial(
                policy=policy,
                config=dense_config,
                seed=seed,
                route_mode="lut",
                fanout_lut=dense_lut,
            )
            flat_point = _trial(
                policy=policy,
                config=baseline_config,
                seed=seed,
                route_mode="flat",
            )
            points.append(
                WikiMemoryDenseTilePoint(
                    page_count=page_count,
                    facts_per_page=facts_per_page,
                    summary_width=summary_width,
                    baseline_group_size=baseline_group_size,
                    dense_group_size=dense_group_size,
                    baseline_max_groups=baseline_max_groups,
                    dense_max_groups=dense_max_groups,
                    baseline_overall_recall=baseline_point.overall_recall,
                    dense_overall_recall=dense_point.overall_recall,
                    flat_overall_recall=flat_point.overall_recall,
                    baseline_cells_read_per_query=baseline_point.cells_read_per_query,
                    dense_cells_read_per_query=dense_point.cells_read_per_query,
                    flat_cells_read_per_query=flat_point.cells_read_per_query,
                    dense_cells_written_per_update=dense_point.cells_written_per_update,
                    baseline_state_bytes=baseline_config.state_bytes,
                    dense_state_bytes=dense_config.state_bytes,
                    baseline_lut_state_bytes=baseline_lut.state_bytes,
                    dense_lut_state_bytes=dense_lut.state_bytes,
                    dense_read_reduction_vs_flat=(
                        1.0 - dense_point.cells_read_per_query / flat_point.cells_read_per_query
                        if flat_point.cells_read_per_query > 0.0
                        else 0.0
                    ),
                    dense_read_reduction_vs_baseline=(
                        1.0
                        - dense_point.cells_read_per_query
                        / baseline_point.cells_read_per_query
                        if baseline_point.cells_read_per_query > 0.0
                        else 0.0
                    ),
                    dense_state_increase_bytes=(
                        dense_config.state_bytes
                        + dense_lut.state_bytes
                        - baseline_config.state_bytes
                        - baseline_lut.state_bytes
                    ),
                    dense_training_examples=dense_lut.training_examples,
                )
            )

    first_config = _density_config(clean_counts[0], clean_facts[0], summary_width)
    return WikiMemoryDenseTileResult(
        policy=policy.name,
        target_route_coverage=target_route_coverage,
        query_events=first_config.query_events,
        update_events=first_config.update_events,
        summary_banks=first_config.summary_banks,
        summary_width=first_config.summary_width,
        summary_bits=first_config.summary_bits,
        points=tuple(points),
    )


def _weighted_pair(sparse_value: float, dense_value: float, dense_weight: float) -> float:
    return (1.0 - dense_weight) * sparse_value + dense_weight * dense_value


def run_wiki_memory_density_aware_tile_sweep(
    total_pages: int = 2048,
    dense_page_fractions: Tuple[float, ...] = (0.25, 0.50, 0.75),
    sparse_facts_per_page: int = 8,
    dense_facts_per_page: int = 32,
    summary_width: int = 256,
    sparse_group_size: int = 16,
    dense_group_size: int = 4,
    sparse_max_groups: int = 32,
    dense_max_groups: int = 48,
    target_route_coverage: float = 1.0,
    train_seeds: Tuple[int, ...] = _DEFAULT_FANOUT_TRAIN_SEEDS,
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryDensityAwareTileResult:
    """Compare uniform and density-aware tile sizing in mixed wiki regions."""

    if total_pages <= 0:
        raise ValueError("total_pages must be positive")
    if total_pages % sparse_group_size != 0 or total_pages % dense_group_size != 0:
        raise ValueError("total_pages must be divisible by both group sizes")
    clean_fractions = tuple(dict.fromkeys(float(value) for value in dense_page_fractions))
    if len(clean_fractions) == 0:
        raise ValueError("dense_page_fractions must not be empty")

    points = []
    first_config = _density_config(total_pages, sparse_facts_per_page, summary_width)
    region_directory_cells = 2 * first_config.summary_banks
    density_tag_state_bytes = total_pages / 8.0
    for dense_fraction in clean_fractions:
        if not 0.0 < dense_fraction < 1.0:
            raise ValueError("dense_page_fractions must be in (0, 1)")
        dense_pages = int(round(total_pages * dense_fraction / sparse_group_size))
        dense_pages *= sparse_group_size
        dense_pages = max(sparse_group_size, min(total_pages - sparse_group_size, dense_pages))
        sparse_pages = total_pages - dense_pages
        dense_query_weight = dense_pages / float(total_pages)

        sparse_base = replace(
            _density_config(sparse_pages, sparse_facts_per_page, summary_width),
            group_size=sparse_group_size,
            selected_groups=4,
            adaptive_max_groups=sparse_max_groups,
            adaptive_score_margin=1,
        )
        dense_base = replace(
            _density_config(dense_pages, dense_facts_per_page, summary_width),
            group_size=sparse_group_size,
            selected_groups=4,
            adaptive_max_groups=sparse_max_groups,
            adaptive_score_margin=1,
        )
        sparse_dense = replace(
            _density_config(sparse_pages, sparse_facts_per_page, summary_width),
            group_size=dense_group_size,
            selected_groups=4,
            adaptive_max_groups=dense_max_groups,
            adaptive_score_margin=1,
        )
        dense_dense = replace(
            _density_config(dense_pages, dense_facts_per_page, summary_width),
            group_size=dense_group_size,
            selected_groups=4,
            adaptive_max_groups=dense_max_groups,
            adaptive_score_margin=1,
        )

        sparse_base_lut = train_wiki_memory_fanout_lut(
            config=sparse_base,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=sparse_max_groups,
            target_route_coverage=target_route_coverage,
        )
        dense_base_lut = train_wiki_memory_fanout_lut(
            config=dense_base,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=sparse_max_groups,
            target_route_coverage=target_route_coverage,
        )
        sparse_dense_lut = train_wiki_memory_fanout_lut(
            config=sparse_dense,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=dense_max_groups,
            target_route_coverage=target_route_coverage,
        )
        dense_dense_lut = train_wiki_memory_fanout_lut(
            config=dense_dense,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=dense_max_groups,
            target_route_coverage=target_route_coverage,
        )

        sparse_base_point = _trial(
            policy=policy,
            config=sparse_base,
            seed=seed,
            route_mode="lut",
            fanout_lut=sparse_base_lut,
        )
        dense_base_point = _trial(
            policy=policy,
            config=dense_base,
            seed=seed,
            route_mode="lut",
            fanout_lut=dense_base_lut,
        )
        sparse_dense_point = _trial(
            policy=policy,
            config=sparse_dense,
            seed=seed,
            route_mode="lut",
            fanout_lut=sparse_dense_lut,
        )
        dense_dense_point = _trial(
            policy=policy,
            config=dense_dense,
            seed=seed,
            route_mode="lut",
            fanout_lut=dense_dense_lut,
        )
        sparse_flat_point = _trial(
            policy=policy,
            config=sparse_base,
            seed=seed,
            route_mode="flat",
        )
        dense_flat_point = _trial(
            policy=policy,
            config=dense_base,
            seed=seed,
            route_mode="flat",
        )

        dense_tile_enabled = (
            dense_dense_point.overall_recall >= dense_base_point.overall_recall
        )
        aware_dense_point = dense_dense_point if dense_tile_enabled else dense_base_point
        aware_dense_config = dense_dense if dense_tile_enabled else dense_base
        aware_dense_lut = dense_dense_lut if dense_tile_enabled else dense_base_lut

        baseline_recall = _weighted_pair(
            sparse_base_point.overall_recall,
            dense_base_point.overall_recall,
            dense_query_weight,
        )
        aware_recall = _weighted_pair(
            sparse_base_point.overall_recall,
            aware_dense_point.overall_recall,
            dense_query_weight,
        )
        all_dense_recall = _weighted_pair(
            sparse_dense_point.overall_recall,
            dense_dense_point.overall_recall,
            dense_query_weight,
        )
        flat_recall = _weighted_pair(
            sparse_flat_point.overall_recall,
            dense_flat_point.overall_recall,
            dense_query_weight,
        )

        baseline_read = _weighted_pair(
            sparse_base_point.cells_read_per_query,
            dense_base_point.cells_read_per_query,
            dense_query_weight,
        ) + region_directory_cells
        aware_read = _weighted_pair(
            sparse_base_point.cells_read_per_query,
            aware_dense_point.cells_read_per_query,
            dense_query_weight,
        ) + region_directory_cells
        all_dense_read = _weighted_pair(
            sparse_dense_point.cells_read_per_query,
            dense_dense_point.cells_read_per_query,
            dense_query_weight,
        ) + region_directory_cells
        flat_read = _weighted_pair(
            sparse_flat_point.cells_read_per_query,
            dense_flat_point.cells_read_per_query,
            dense_query_weight,
        ) + region_directory_cells

        baseline_state = (
            sparse_base.state_bytes
            + dense_base.state_bytes
            + sparse_base_lut.state_bytes
            + dense_base_lut.state_bytes
        )
        aware_state = (
            sparse_base.state_bytes
            + aware_dense_config.state_bytes
            + sparse_base_lut.state_bytes
            + aware_dense_lut.state_bytes
            + density_tag_state_bytes
        )
        all_dense_state = (
            sparse_dense.state_bytes
            + dense_dense.state_bytes
            + sparse_dense_lut.state_bytes
            + dense_dense_lut.state_bytes
        )

        points.append(
            WikiMemoryDensityAwareTilePoint(
                total_pages=total_pages,
                dense_page_fraction=dense_pages / float(total_pages),
                dense_query_fraction=dense_query_weight,
                sparse_pages=sparse_pages,
                dense_pages=dense_pages,
                sparse_facts_per_page=sparse_facts_per_page,
                dense_facts_per_page=dense_facts_per_page,
                dense_tile_enabled=dense_tile_enabled,
                baseline_overall_recall=baseline_recall,
                aware_overall_recall=aware_recall,
                all_dense_overall_recall=all_dense_recall,
                flat_overall_recall=flat_recall,
                baseline_cells_read_per_query=baseline_read,
                aware_cells_read_per_query=aware_read,
                all_dense_cells_read_per_query=all_dense_read,
                flat_cells_read_per_query=flat_read,
                baseline_state_bytes=baseline_state,
                aware_state_bytes=aware_state,
                all_dense_state_bytes=all_dense_state,
                density_tag_state_bytes=density_tag_state_bytes,
                aware_read_reduction_vs_flat=(
                    1.0 - aware_read / flat_read if flat_read > 0.0 else 0.0
                ),
                aware_read_reduction_vs_baseline=(
                    1.0 - aware_read / baseline_read if baseline_read > 0.0 else 0.0
                ),
                aware_state_increase_vs_baseline=(
                    aware_state - baseline_state
                ),
                aware_state_saving_vs_all_dense=(
                    1.0 - aware_state / all_dense_state if all_dense_state > 0.0 else 0.0
                ),
                aware_training_examples=(
                    sparse_base_lut.training_examples + aware_dense_lut.training_examples
                ),
            )
        )

    return WikiMemoryDensityAwareTileResult(
        policy=policy.name,
        target_route_coverage=target_route_coverage,
        query_events=first_config.query_events,
        update_events=first_config.update_events,
        summary_banks=first_config.summary_banks,
        summary_width=first_config.summary_width,
        summary_bits=first_config.summary_bits,
        region_directory_cells_per_query=region_directory_cells,
        points=tuple(points),
    )


def _refresh_density_tag(
    facts_per_page: int,
    density_tag_bits: int,
    facts_per_tag_step: int,
) -> int:
    max_tag = (1 << density_tag_bits) - 1
    if facts_per_tag_step <= 0:
        raise ValueError("facts_per_tag_step must be positive")
    return min(max_tag, max(0, int(facts_per_page) // facts_per_tag_step))


def run_wiki_memory_density_tag_sweep(
    total_pages: int = 2048,
    dense_page_fractions: Tuple[float, ...] = (0.25, 0.50, 0.75),
    tag_thresholds: Tuple[int, ...] = (2, 3, 4),
    sparse_facts_per_page: int = 8,
    dense_facts_per_page: int = 32,
    summary_width: int = 256,
    sparse_group_size: int = 16,
    dense_group_size: int = 4,
    sparse_max_groups: int = 32,
    dense_max_groups: int = 48,
    density_tag_bits: int = 2,
    facts_per_tag_step: int = 8,
    quality_probe_queries: int = 128,
    quality_probe_updates: int = 64,
    quality_probe_min_gain: float = 0.02,
    quality_probe_seed: int = 901,
    target_route_coverage: float = 1.0,
    train_seeds: Tuple[int, ...] = _DEFAULT_FANOUT_TRAIN_SEEDS,
    policy: WikiMemoryRefreshPolicy = WikiMemoryRefreshPolicy(
        "trigger16_age16_clusterbook",
        dirty_threshold=16,
        max_age=16,
        error_book_repair=True,
        cluster_repair=True,
    ),
    seed: int = 91,
) -> WikiMemoryDensityTagResult:
    """Sweep low-bit density tags generated by normal summary refresh."""

    if total_pages <= 0:
        raise ValueError("total_pages must be positive")
    if total_pages % sparse_group_size != 0 or total_pages % dense_group_size != 0:
        raise ValueError("total_pages must be divisible by both group sizes")
    if density_tag_bits <= 0:
        raise ValueError("density_tag_bits must be positive")
    if quality_probe_queries <= 0:
        raise ValueError("quality_probe_queries must be positive")
    if quality_probe_updates < 0:
        raise ValueError("quality_probe_updates must be non-negative")
    if quality_probe_min_gain < 0.0:
        raise ValueError("quality_probe_min_gain must be non-negative")
    clean_fractions = tuple(dict.fromkeys(float(value) for value in dense_page_fractions))
    clean_thresholds = tuple(dict.fromkeys(int(value) for value in tag_thresholds))
    if len(clean_fractions) == 0:
        raise ValueError("dense_page_fractions must not be empty")
    if len(clean_thresholds) == 0:
        raise ValueError("tag_thresholds must not be empty")

    first_config = _density_config(total_pages, sparse_facts_per_page, summary_width)
    region_directory_cells = 2 * first_config.summary_banks
    density_tag_state_bytes = total_pages * density_tag_bits / 8.0
    sparse_tag = _refresh_density_tag(
        sparse_facts_per_page,
        density_tag_bits,
        facts_per_tag_step,
    )
    dense_tag = _refresh_density_tag(
        dense_facts_per_page,
        density_tag_bits,
        facts_per_tag_step,
    )
    points = []

    for dense_fraction in clean_fractions:
        if not 0.0 < dense_fraction < 1.0:
            raise ValueError("dense_page_fractions must be in (0, 1)")
        dense_pages = int(round(total_pages * dense_fraction / sparse_group_size))
        dense_pages *= sparse_group_size
        dense_pages = max(sparse_group_size, min(total_pages - sparse_group_size, dense_pages))
        sparse_pages = total_pages - dense_pages
        dense_query_weight = dense_pages / float(total_pages)

        sparse_base = replace(
            _density_config(sparse_pages, sparse_facts_per_page, summary_width),
            group_size=sparse_group_size,
            selected_groups=4,
            adaptive_max_groups=sparse_max_groups,
            adaptive_score_margin=1,
        )
        dense_base = replace(
            _density_config(dense_pages, dense_facts_per_page, summary_width),
            group_size=sparse_group_size,
            selected_groups=4,
            adaptive_max_groups=sparse_max_groups,
            adaptive_score_margin=1,
        )
        sparse_dense = replace(
            _density_config(sparse_pages, sparse_facts_per_page, summary_width),
            group_size=dense_group_size,
            selected_groups=4,
            adaptive_max_groups=dense_max_groups,
            adaptive_score_margin=1,
        )
        dense_dense = replace(
            _density_config(dense_pages, dense_facts_per_page, summary_width),
            group_size=dense_group_size,
            selected_groups=4,
            adaptive_max_groups=dense_max_groups,
            adaptive_score_margin=1,
        )

        sparse_base_lut = train_wiki_memory_fanout_lut(
            config=sparse_base,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=sparse_max_groups,
            target_route_coverage=target_route_coverage,
        )
        dense_base_lut = train_wiki_memory_fanout_lut(
            config=dense_base,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=sparse_max_groups,
            target_route_coverage=target_route_coverage,
        )
        sparse_dense_lut = train_wiki_memory_fanout_lut(
            config=sparse_dense,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=dense_max_groups,
            target_route_coverage=target_route_coverage,
        )
        dense_dense_lut = train_wiki_memory_fanout_lut(
            config=dense_dense,
            policy=policy,
            train_seeds=train_seeds,
            base_groups=4,
            max_groups=dense_max_groups,
            target_route_coverage=target_route_coverage,
        )

        sparse_base_point = _trial(policy, sparse_base, seed, "lut", sparse_base_lut)
        dense_base_point = _trial(policy, dense_base, seed, "lut", dense_base_lut)
        sparse_dense_point = _trial(policy, sparse_dense, seed, "lut", sparse_dense_lut)
        dense_dense_point = _trial(policy, dense_dense, seed, "lut", dense_dense_lut)
        sparse_flat_point = _trial(policy, sparse_base, seed, "flat")
        dense_flat_point = _trial(policy, dense_base, seed, "flat")
        sparse_base_probe = _trial(
            policy,
            replace(
                sparse_base,
                query_events=quality_probe_queries,
                update_events=quality_probe_updates,
            ),
            quality_probe_seed,
            "lut",
            sparse_base_lut,
        )
        sparse_dense_probe = _trial(
            policy,
            replace(
                sparse_dense,
                query_events=quality_probe_queries,
                update_events=quality_probe_updates,
            ),
            quality_probe_seed,
            "lut",
            sparse_dense_lut,
        )
        dense_base_probe = _trial(
            policy,
            replace(
                dense_base,
                query_events=quality_probe_queries,
                update_events=quality_probe_updates,
            ),
            quality_probe_seed,
            "lut",
            dense_base_lut,
        )
        dense_dense_probe = _trial(
            policy,
            replace(
                dense_dense,
                query_events=quality_probe_queries,
                update_events=quality_probe_updates,
            ),
            quality_probe_seed,
            "lut",
            dense_dense_lut,
        )

        baseline_recall = _weighted_pair(
            sparse_base_point.overall_recall,
            dense_base_point.overall_recall,
            dense_query_weight,
        )
        flat_recall = _weighted_pair(
            sparse_flat_point.overall_recall,
            dense_flat_point.overall_recall,
            dense_query_weight,
        )
        baseline_read = _weighted_pair(
            sparse_base_point.cells_read_per_query,
            dense_base_point.cells_read_per_query,
            dense_query_weight,
        ) + region_directory_cells
        flat_read = _weighted_pair(
            sparse_flat_point.cells_read_per_query,
            dense_flat_point.cells_read_per_query,
            dense_query_weight,
        ) + region_directory_cells
        baseline_state = (
            sparse_base.state_bytes
            + dense_base.state_bytes
            + sparse_base_lut.state_bytes
            + dense_base_lut.state_bytes
        )

        for threshold in clean_thresholds:
            sparse_tag_enabled = sparse_tag >= threshold
            dense_tag_enabled = dense_tag >= threshold
            sparse_tag_point = sparse_dense_point if sparse_tag_enabled else sparse_base_point
            dense_tag_point = dense_dense_point if dense_tag_enabled else dense_base_point
            sparse_tag_config = sparse_dense if sparse_tag_enabled else sparse_base
            dense_tag_config = dense_dense if dense_tag_enabled else dense_base
            sparse_tag_lut = sparse_dense_lut if sparse_tag_enabled else sparse_base_lut
            dense_tag_lut = dense_dense_lut if dense_tag_enabled else dense_base_lut

            sparse_guard_enabled = (
                sparse_tag_enabled
                and sparse_dense_probe.overall_recall
                >= sparse_base_probe.overall_recall + quality_probe_min_gain
            )
            dense_guard_enabled = (
                dense_tag_enabled
                and dense_dense_probe.overall_recall
                >= dense_base_probe.overall_recall + quality_probe_min_gain
            )
            sparse_guard_point = sparse_dense_point if sparse_guard_enabled else sparse_base_point
            dense_guard_point = dense_dense_point if dense_guard_enabled else dense_base_point
            sparse_guard_config = sparse_dense if sparse_guard_enabled else sparse_base
            dense_guard_config = dense_dense if dense_guard_enabled else dense_base
            sparse_guard_lut = sparse_dense_lut if sparse_guard_enabled else sparse_base_lut
            dense_guard_lut = dense_dense_lut if dense_guard_enabled else dense_base_lut

            tag_only_recall = _weighted_pair(
                sparse_tag_point.overall_recall,
                dense_tag_point.overall_recall,
                dense_query_weight,
            )
            guarded_recall = _weighted_pair(
                sparse_guard_point.overall_recall,
                dense_guard_point.overall_recall,
                dense_query_weight,
            )
            tag_only_read = _weighted_pair(
                sparse_tag_point.cells_read_per_query,
                dense_tag_point.cells_read_per_query,
                dense_query_weight,
            ) + region_directory_cells
            guarded_read = _weighted_pair(
                sparse_guard_point.cells_read_per_query,
                dense_guard_point.cells_read_per_query,
                dense_query_weight,
            ) + region_directory_cells
            tag_only_state = (
                sparse_tag_config.state_bytes
                + dense_tag_config.state_bytes
                + sparse_tag_lut.state_bytes
                + dense_tag_lut.state_bytes
                + density_tag_state_bytes
            )
            guarded_state = (
                sparse_guard_config.state_bytes
                + dense_guard_config.state_bytes
                + sparse_guard_lut.state_bytes
                + dense_guard_lut.state_bytes
                + density_tag_state_bytes
            )

            points.append(
                WikiMemoryDensityTagPoint(
                    total_pages=total_pages,
                    dense_page_fraction=dense_pages / float(total_pages),
                    tag_threshold=threshold,
                    sparse_density_tag=sparse_tag,
                    dense_density_tag=dense_tag,
                    tag_dense_enabled=dense_tag_enabled,
                    guard_dense_enabled=dense_guard_enabled,
                    baseline_overall_recall=baseline_recall,
                    tag_only_overall_recall=tag_only_recall,
                    guarded_overall_recall=guarded_recall,
                    flat_overall_recall=flat_recall,
                    baseline_cells_read_per_query=baseline_read,
                    tag_only_cells_read_per_query=tag_only_read,
                    guarded_cells_read_per_query=guarded_read,
                    flat_cells_read_per_query=flat_read,
                    tag_only_read_reduction_vs_flat=(
                        1.0 - tag_only_read / flat_read if flat_read > 0.0 else 0.0
                    ),
                    guarded_read_reduction_vs_flat=(
                        1.0 - guarded_read / flat_read if flat_read > 0.0 else 0.0
                    ),
                    tag_only_state_bytes=tag_only_state,
                    guarded_state_bytes=guarded_state,
                    baseline_state_bytes=baseline_state,
                    density_tag_state_bytes=density_tag_state_bytes,
                    sparse_probe_baseline_recall=sparse_base_probe.overall_recall,
                    sparse_probe_dense_recall=sparse_dense_probe.overall_recall,
                    dense_probe_baseline_recall=dense_base_probe.overall_recall,
                    dense_probe_dense_recall=dense_dense_probe.overall_recall,
                )
            )

    return WikiMemoryDensityTagResult(
        policy=policy.name,
        target_route_coverage=target_route_coverage,
        query_events=first_config.query_events,
        update_events=first_config.update_events,
        summary_banks=first_config.summary_banks,
        summary_width=first_config.summary_width,
        summary_bits=first_config.summary_bits,
        density_tag_bits=density_tag_bits,
        region_directory_cells_per_query=region_directory_cells,
        quality_probe_queries=quality_probe_queries,
        quality_probe_updates=quality_probe_updates,
        quality_probe_min_gain=quality_probe_min_gain,
        points=tuple(points),
    )
