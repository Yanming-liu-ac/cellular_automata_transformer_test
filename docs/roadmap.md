# Roadmap

## Phase 0: Architecture Sanity

Status: started.

Deliverables:

- HARC-CA topology definition.
- Propagation-depth measurement.
- Low-bit integer cell simulator.
- Hardware proxy table against tiny Transformer KV-cache traffic.

Decision:

- Continue only if multiscale routing stays near logarithmic depth and local
  traffic remains plausibly lower than Transformer KV traffic under sparse
  activity.

## Phase 1: Retrieval Before Language

Status: multi-route hash-routed prototype and sequence-memory benchmarks added.

Language modeling will fail if the CA cannot retrieve exact distant information.
The first hard tasks should be:

- copy a symbol from distance `d`;
- induction pattern: `A B ... A -> B`;
- bracket/quote matching;
- key-value recall;
- sparse associative lookup through hierarchy.

Success criterion:

```text
Accuracy remains high as context grows, while active cells per query grow
sublinearly rather than scanning the full context.
```

Initial result:

- A 4-way, 2-route hash-routed associative lane reaches 100% exact recall in the
  first random trial when load factor is 0.5.
- At load factor 1.0 on 16k context, 2-route lookup reaches roughly 92-93%
  exact recall on copy, induction, and key-value tasks while visiting about 32
  cells per query instead of scanning 16,384 token cells.
- At the same load, 4-route and 8-route checks improve recall further but still
  do not guarantee perfect recall; capacity pressure remains a real design
  problem.

Overflow-tier result:

- At 16k context, `buckets=context/4`, `ways=4`, `routes=2`, single-lane recall
  is roughly 92-93% on copy, induction, and key-value tasks.
- Adding a smaller overflow lane with `buckets=context/16`, `ways=4`,
  `routes=2`, and 32-bit tags recovers 100% exact recall in the current
  deterministic full-context trial.
- Average visited cells rise only from about 32 to about 34, because overflow is
  checked only after primary misses; about 8% of queries touch overflow in the
  current trial.

Compressed dense-context result:

- A 4-bit decayed count-sketch with `banks=4`, `width=2048`, and four updates
  per token uses 4KB of state.
- On the current deterministic 65k-vocabulary topic stream, it recovers 100% of
  the exact top-64 decayed topic tokens at 8x lower state than a full 4-bit
  dense counter table.
- This validates only fuzzy dense context compression, not exact recall.

Dual-path result:

- Tiered exact lane plus dense sketch uses about 166.5KB in the current demo.
- Exact lane handles deterministic 16k induction recall.
- Dense sketch handles deterministic 65k-vocabulary topic/recency distribution.

Synthetic next-token result:

- A non-trained dual-path predictor now handles mixed topic events and induction
  key-query events.
- Exact induction next-token accuracy is 100% in the current deterministic run.
- Dense topic candidate top-k hit rate is about 62% using a 512-token candidate
  shortlist.
- The mixed stream touches about 27 local cells per event.
- This is a bridge benchmark, not a language-model quality result.

Online candidate-cache result:

- A 512-entry low-bit set-associative cache now generates candidate shortlists
  without a hot-token oracle or full-vocabulary scan.
- Always-admit standalone topic/noise top-64 hit rate is about 69% after warmup.
- A threshold-1 dense-sketch admission gate raises standalone top-64 hit rate to
  about 70.8% and removes almost all 512-entry cache replacements.
- Plugged into the synthetic LM, gated online topic@64 is about 67.1% versus
  about 62.1% for the static candidate pool and about 61.4% for always-admit
  online cache.
- The gated path adds about 1.31KB of cache state, admits about 60.5% of topic
  observations, touches about 4.0 candidate-cache cells/event, and reads about
  2.7 dense gate cells/event.
- A self-supervised learned admission LUT now recovers the same gate from a
  future-repeat label. The LUT has 16 signed 4-bit entries, uses 8 bytes, reaches
  about 70.8% standalone top-64 hit rate, and keeps synthetic-LM topic@64 at
  about 67.1%.

Cellular-MoE execution result:

- A low-bit CA rule-bank prototype routes active cells to one of six local rules.
- With 20% active cells and top-1 routing, sparse execution uses about 30x fewer
  rule updates than dense all-cell/all-rule execution.
- Bias-controlled routing reduces rule-load CV from about 1.23 to about 0.74 in
  the current deterministic rollout.
- This validates the execution shape, not trained model quality.

Unified efficiency profile:

- The current event-level proxy combines exact memory, dense sketch updates, and
  Cellular-MoE rule execution.
- With 4 rule ticks per event, estimated local on-chip traffic is about
  51.38KB per synthetic event including gated online candidate-cache updates.
- The tiny Transformer KV reference at 16k context reads about 384MB per token.
- This is a design-budget signal, not an energy or quality-equivalence claim.

Tile/floorplan profile:

- The first chip mapping proxy uses 64 cells/tile, 16KB local SRAM/tile, and 32
  local bytes/cycle/tile.
- A 32-tile fabric stores the current prototype state in about 35.9% of local
  SRAM.
- At a 1M synthetic events/s target, aggregate local bandwidth utilization is
  about 5.1% under the proxy assumptions.
- This defines a budget for learned rules and richer output heads; it is not
  physical design closure.

Output-head profile:

- A 65k full-vocabulary output head costs about 4.13MB/event and 8.39M
  MACs/event under the current proxy assumptions.
- A 512-token candidate head costs about 33KB/event.
- A 512-token candidate head with exact-query bypass costs about 22KB/event.
- Candidate generation must be learned and accurate; otherwise the output layer
  becomes the new global bottleneck.

Next retrieval work:

- learned or content-aware routing instead of pure hashing;
- variable-width exact memory for rare names, numbers, and code symbols;
- degradation tests with repeated keys and conflicting induction patterns.
- separate metrics for exact sparse recall versus compressed dense context,
  following the DeepSeek-V4 CSA/HCA split.

## Phase 2: Trainable Continuous HARC-CA

Add PyTorch or JAX when the environment allows dependency installation.

Implement:

- continuous cell state;
- grouped local / summary / route / memory-IO / stability channels;
- shared local update rule;
- Cellular-MoE rule banks with bounded low-cost routing;
- residual bounded updates;
- random asynchronous update masks;
- auxiliary routing losses;
- route-bias control inspired by auxiliary-loss-free load balancing;
- multi-token / multi-tick prediction heads;
- tiny Transformer teacher for distillation experiments.
- optional Muon-style optimizer experiment for the shared recurrent rule.

First trainable target:

```text
Learn the routing decision and candidate scoring currently hand-coded in the
synthetic next-token benchmark.
```

This target now includes learning a candidate output policy that avoids
full-vocabulary scoring for most events.

The first NumPy version of this target is the learned admission LUT. It is not a
neural CA yet, but it proves the hand-set threshold can be replaced by a tiny
trainable low-bit rule.

Second trainable target:

```text
Replace hand-written Cellular-MoE rule banks with learned local rules while
preserving sparse top-k routing and load-bias control.
```

Success criterion:

```text
The model learns algorithmic sequence tasks with fewer global bytes moved than a
same-size Transformer baseline.
```

## Phase 3: Quantized HARC-CA

Move from continuous training to deployment-shaped inference:

- 8-bit training baseline;
- 4-bit state as a primary target, inspired by DeepSeek-V4 FP4 QAT;
- 2-bit / binary ablations;
- per-group scale / offset metadata;
- LUT-style or XNOR-popcount rule kernels;
- integer-only rollout verification.

Success criterion:

```text
Integer-only rollout preserves most task accuracy for at least 1,000 recurrent
ticks without collapse.
```

## Phase 4: Byte-Level Language Model

Build the first language model only after retrieval and stability are working.

Tasks:

- byte-level next-token prediction;
- small character corpora;
- synthetic code-like data;
- perplexity and bits-per-byte metrics;
- latency and proxy energy per generated byte.

Success criterion:

```text
At a small quality target, HARC-CA has lower estimated memory movement per
generated byte than a tiny Transformer.
```

## Phase 5: Chip Mapping

Translate the architecture into a hardware proposal:

- cell tile microarchitecture;
- local SRAM/register layout;
- parent-child routing fabric;
- active-region scheduler;
- low-bit LUT / XNOR-popcount datapath;
- fine-grained quantization metadata and online cast path;
- route-wave / reduction communication offload;
- state-cache hierarchy for active tail, summaries, exact-memory overflow, and
  reusable prompt-prefix states;
- floorplan proxy with tile count, local SRAM, local bandwidth, and state
  utilization budgets;
- area and bandwidth estimates;
- FPGA prototype plan.

Success criterion:

```text
The architecture has a credible path to better tokens-per-watt or
latency-per-watt in a defined deployment class.
```

## Kill Criteria

The project should pivot or stop if:

- retrieval requires dense full-context scans;
- useful quality requires dense global communication every generated token;
- quantized rollout is unstable;
- the output head dominates all savings;
- hardware traffic estimates converge to Transformer-like memory movement.
