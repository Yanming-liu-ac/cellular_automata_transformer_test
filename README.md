# Cellular Transformer Research

This repository is a research workspace for a CA-first language-model and AI-chip
architecture. The target is not to emulate a Transformer instruction by
instruction, but to discover a trainable cellular automaton architecture that is:

- stable under long recurrent rollout;
- fast at propagating information across long contexts;
- naturally quantizable to low-bit state and low-bit rules;
- directly mappable to a local-memory, local-wire AI accelerator.

## Current Candidate

The first concrete candidate is **HARC-CA**:

> Hierarchical Associative Recurrent Cellular Automaton.

HARC-CA represents text context as a persistent field of low-bit cells. Each cell
updates from local neighbors, but cells are arranged in a multiscale lattice so a
new token can affect local context, block summaries, and long-range associative
routes without reading the entire KV cache.

The near-term goal is to turn this into a sequence-model testbed:

1. prove fast information propagation on the proposed lattice;
2. train a continuous neural CA version on toy language and algorithm tasks;
3. quantize the learned rule to bit-sliced / LUT-style updates;
4. estimate memory movement, local wire cost, and cell throughput against a tiny
   Transformer baseline.

DeepSeek-V4's CSA/HCA design is the closest current external systems anchor:
it supports the two-path memory thesis, while this repository explores whether
the same efficiency logic can be recast as CA-native state, routing, and tiny
low-bit controllers.

## Repository Layout

- `docs/architecture.md` - HARC-CA architecture proposal.
- `docs/research_notes.md` - evidence, hypotheses, and research risks.
- `docs/deepseek_lessons.md` - DeepSeek V3/V4 design lessons translated to
  HARC-CA.
- `docs/wiki_memory_track.md` - CA-native external knowledge and wiki-memory
  track.
- `docs/hardware_metrics.md` - chip-oriented metrics to track from day one.
- `docs/roadmap.md` - staged experiments and kill criteria.
- `src/cellular_transformer/` - NumPy prototypes and measurement utilities.
- `experiments/` - runnable experiments.

## Quick Check

```powershell
python experiments/propagation_demo.py
python experiments/retrieval_demo.py
python experiments/task_benchmark_demo.py
python experiments/overflow_benchmark_demo.py
python experiments/dense_context_demo.py
python experiments/compressed_block_indexer_demo.py
python experiments/dual_path_demo.py
python experiments/candidate_cache_demo.py
python experiments/learned_admission_demo.py
python experiments/candidate_scorer_demo.py
python experiments/candidate_indexer_demo.py
python experiments/synthetic_lm_demo.py
python experiments/synthetic_candidate_demand_sweep_demo.py
python experiments/synthetic_candidate_reducer_demo.py
python experiments/synthetic_hierarchical_candidate_reducer_demo.py
python experiments/synthetic_group_summary_update_demo.py
python experiments/synthetic_lazy_group_summary_demo.py
python experiments/synthetic_triggered_group_summary_demo.py
python experiments/wiki_memory_demo.py
python experiments/wiki_memory_scaling_demo.py
python experiments/wiki_memory_density_demo.py
python experiments/wiki_memory_fanout_demo.py
python experiments/wiki_memory_learned_fanout_grid_demo.py
python experiments/wiki_memory_dense_tile_demo.py
python experiments/wiki_memory_density_aware_tile_demo.py
python experiments/wiki_memory_density_tag_demo.py
python experiments/wiki_memory_mixed_guard_counter_demo.py
python experiments/wiki_memory_mixed_guard_loss_decay_demo.py
python experiments/wiki_memory_mixed_guard_stress_demo.py
python experiments/wiki_memory_mixed_guard_noise_demo.py
python experiments/wiki_memory_learned_guard_sharing_demo.py
python experiments/wiki_memory_learned_guard_audit_demo.py
python experiments/wiki_memory_learned_guard_noise_matrix_demo.py
python experiments/wiki_memory_learned_guard_random_noise_demo.py
python experiments/cellular_moe_demo.py
python experiments/efficiency_profile_demo.py
python experiments/chip_floorplan_demo.py
python experiments/output_head_demo.py
python experiments/hardware_estimate_demo.py
python experiments/lowbit_demo.py
```

The first experiments compare:

- propagation depth for local CA versus HARC-CA;
- low-bit dynamic propagation stability for residual, route, and mHC-style
  grouped CA rules, including 1,000-tick unforced stability checks;
- 1,000-tick content retention for shared mHC carrier versus persistent content
  lane and local refresh policies;
- local content-to-carrier write gates that expose persistent content to the mHC
  carrier only when low-bit mismatch warrants the write;
- an 8-byte learned content-gate LUT over mismatch, route, and envelope buckets;
- a 16-byte demand-weighted content-gate LUT that writes persistent content only
  for route/query-demanded cells;
- rare-directory query-trace demand gates that drive content exposure from
  actual retrieval occurrence positions;
- synthetic exact-query demand gates that wake persistent content only for
  exact-memory lookup events;
- synthetic mixed exact+candidate demand gates that expose the output-side write
  pressure of candidate shortlist scoring;
- candidate-output demand sparsity sweeps showing that content exposure needs
  shortlist pruning before waking candidate rows;
- 9-byte phase/rank/mismatch demand gates that recover exact candidate exposure
  on sparse output demand;
- low-bit candidate reducer traces that turn top-64 candidate demand into
  top-M content exposure before the exact gate;
- hierarchical group-summary candidate reducers that avoid full-pool candidate
  scoring before top-M exposure;
- group-summary maintenance diagnostics that estimate update cost for the
  hierarchical reducer;
- lazy group-summary refresh diagnostics that trade stale summaries for lower
  update traffic;
- triggered group-summary refresh diagnostics that replace fixed refresh
  intervals with local dirty-count and age rules;
- CA wiki-memory diagnostics for mutable page/fact/link storage with triggered
  summary refresh and error-book repair;
- wiki-memory scaling sweeps comparing hierarchical CA routing with flat
  page-summary scans;
- wiki-memory density sweeps that expose summary collision pressure as
  facts/page increases;
- adaptive and learned-LUT wiki-memory fanout sweeps that recover dense-page
  recall without falling back to full flat page-summary scans;
- learned fanout grid sweeps that expose the density boundary where stronger
  summaries or page-internal routing are needed;
- dense routing-tile sweeps that split high-density wiki memory into smaller
  local tiles and restore recall under 32 facts/page pressure;
- density-aware routing-tile sweeps that enable small tiles only when a local
  rolling probe shows enough recall gain to justify the extra state;
- refresh-derived density-tag sweeps showing why low-bit density tags need a
  low-bit online agreement guard before switching routing geometry;
- mixed-stream guard-counter sweeps that feed those low-bit counters from one
  sparse/dense event stream instead of separate region probes;
- event-driven loss-decay sweeps for strict wiki-memory guard counters;
- observation-window stress sweeps for the same shared guard counters;
- update-noise stress sweeps for the shared guard counters;
- learned low-bit sharing-radius/loss-tolerance LUTs for mixed-stream guard counters;
- held-out loss-tolerance audits for learned wiki-memory guard counters;
- update-noise matrix audits for learned wiki-memory guard loss tolerance;
- deterministic randomized-noise audits for learned wiki-memory guard tolerance;
- exact key/value recall through a hash-routed associative CA lane;
- copy, induction, and key/value memory tasks;
- overflow-tier associative memory for exact-recall reliability;
- low-bit compressed dense-context sketching;
- CSA-shaped compressed block indexing plus rare-token block-directory repair;
- combined sparse-exact plus dense-compressed memory path;
- online low-bit candidate-cache generation with admission gating and no
  full-vocabulary scans;
- learned low-bit candidate admission from self-supervised repeat labels;
- learned candidate scorer negative benchmarks, including a future-window
  residual indexer, plus explicit scoring-read budget;
- source/phase/cache candidate scoring sketches for testing dense-state
  contamination and local indexer features;
- trainable low-bit multi-feature candidate indexer benchmark with age/feature
  collision diagnostics;
- synthetic next-token prediction over the dual memory path;
- sparse low-bit Cellular-MoE rule-bank execution;
- unified HARC-CA local-traffic proxy versus Transformer KV-cache read volume;
- tile-level HARC-CA floorplan and local SRAM/bandwidth proxy;
- output-head full-vocabulary versus candidate-shortlist budget;
- rough local-message traffic versus Transformer KV-cache traffic;
- integer-only low-bit state rollout.
