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

Compressed block-index result:

- A CSA-shaped context-block index now splits 65k context into 1024 blocks of
  64 tokens and stores a 4-bit count-min summary in each block cell.
- With 4 banks, `summary_width=256`, 8 selected blocks, and 2 exact tail blocks,
  it uses about 512KB of block-summary state and scores about 2KB of summary
  counters per query.
- On the deterministic topic/noise trial it reaches 100% relevant block-hit rate
  while reading about 640 token positions instead of all 65,536, about a 102x
  token-read reduction.
- Occurrence coverage is only about 8.4%, close to the oracle top-block coverage
  at the same block budget. This validates sparse block routing, not full
  attention replacement.
- A repeated-read sweep shows that the compressed index is already close to
  exact block ranking: from 4 to 128 selected blocks, oracle coverage gap stays
  below about 0.3 percentage points. Coverage grows from about 5.6% to 46.1%,
  while read reduction falls from about 170.7x to 7.9x. The next issue is read
  policy and memory-path split, not simply a better block score.
- The first CSA/HCA routing policy uses a 4KB global low-bit summary to skip
  block scoring for frequent queries and reserve CSA reads for rare queries.
  With threshold 8 it routes 100% of measured hot relevant queries to HCA and
  100% of measured cold relevant queries to CSA, reducing average block-score
  reads to about 300B/query and token block reads to about 165/query.
- A first CSA block-state compression sweep finds a compact SRAM point:
  `block_size=128`, `summary_width=256`, and `csa_blocks=4` cut block-summary
  state from 512KB to 256KB while preserving 100% measured CSA-path hit and
  coverage on routed relevant queries. Token block reads rise to about
  331/query, but the full-context token-read reduction remains about 198x.
- A rare-token block directory improves that point again. With `block_size=128`,
  `summary_width=128`, threshold 15, and six exact directory block ids per rare
  token, CSA state is about 158.8KB instead of 256KB and the routed CSA subset
  still reaches 100% measured hit and coverage on the reference stream. The
  directory read cost is only about 0.48B/query in the current average trial.
- The first rare-directory stress sweep shows why the gate and directory fanout
  must be explicit hardware policy knobs. Threshold 8 creates too many
  false-HCA routes for bursty rare tokens; threshold 15 cuts rare false-HCA to
  about 0.8% in the stress set. `dir_k=2` handles burst/split rare tokens, while
  repeated names spread across six blocks need `dir_k=6` to reach about 99.2%
  coverage. Pure rare-query stress reduces token-read savings to about 52x-86x.
- A directory-guard mode gives a higher-recall alternative: probe the rare-token
  directory before HCA admission and force CSA on a hit. On repeated-name stress,
  threshold 8 without the guard gives 75% false-HCA and 25% coverage; threshold
  8 with the guard gives 0% false-HCA and 100% coverage, at the cost of an extra
  directory probe.
- Separating stored fanout from read fanout shows that storing 6 rare block ids
  is not enough by itself. Reading only 2 saves traffic but leaves repeated-name
  coverage around 68%; reading 6 recovers about 99-100% coverage. Fanout needs a
  metadata-driven or learned predictor.
- The first metadata fanout predictor is a span-class LUT: base read fanout is
  2, and the directory expands to 4, 5, or 6 reads when stored block ids span a
  large context region. On repeated-name stress, guarded `span2to4` reaches
  about 93.0% coverage at 13.0B/query, `span2to5` reaches about 98.4% at
  16.25B/query, and full `span2to6` reaches 100.0% at 19.5B/query.
- The first trained fanout LUT uses self-supervised coverage labels and visible
  entry-count/span/CSA-overlap metadata. Its 112-entry 3-bit table is 42B. It
  reaches about 98.4% repeated-name coverage at 12.87B/query, and about 99.7%
  split-rare coverage at 6.50B/query.
- The first joint probe/fanout control adds a 40B HCA-confidence LUT using HCA
  estimate, bank spread, and saturation count. It skips reference hot-token
  probes, keeping reference directory traffic at 0.50B/query instead of
  3.25B/query, while keeping about 97.7% repeated-name coverage at
  12.77B/query.
- Sweeping HCA threshold under joint control rejects threshold 6 and shows that
  thresholds 8-15 keep similar rare coverage. Threshold 15 removes early probes
  in the split/repeated stress cases while keeping about 98.7% split-rare and
  98.3% repeated-name coverage.
- The first trained HCA route LUT is 40B and replaces the explicit threshold at
  inference, but it is not yet the default: it keeps about 99.0% split-rare
  coverage and 97.7% repeated-name coverage, slightly below the threshold-15
  joint policy.
- Adding one rare-directory presence bit gives the first stronger learned
  admission table: the route LUT becomes 80B, pays a modeled 0.125B/query
  sidecar read, preserves 84.7% reference HCA routing, removes rare false-HCA,
  reaches 100.0% split-rare coverage, and reaches 98.4% repeated-name coverage.
- A Bloom-like presence-sidecar false-positive sweep says rare recall is robust
  but HCA hot-path efficiency is sensitive: 1% FPR keeps reference HCA routing
  at 82.1%, 10% keeps it at 80.0%, and 25% drops it to 46.3%. The first sidecar
  target is therefore roughly 1-10% FPR, pending a real layout model.
- The first physical Bloom sidecar instantiates that target. At 8 bits/entry,
  `k=3`, and 8 banks, it uses about 8.8KB on the reference case, reads
  0.375B/query, measures about 1.1% sidecar false positives, keeps reference
  HCA routing at 84.2%, and keeps the rare stress coverage at 100.0% split-rare
  and 98.4% repeated-name.
- Hash-salt robustness is now measured for that candidate. Across 16 salts,
  reference HCA routing averages 82.9%, ranges from 79.7% to 84.6%, and the
  worst salt has about 5.9% hot-token false positives. Salt choice must be a
  compiler/training knob, not a fixed constant.
- Bank mapping is now separated from Bloom false positives. For the same
  sidecar, `by_hash` banking removes same-query bank conflicts, while modulo and
  hash-slot banking average about 36-38% query conflicts. This is a pure layout
  improvement rather than a model-state change.
- Salt selection is now measured under `by_hash`. Scanning 16 salts on a
  reference selection stream picks salt index 14; evaluation keeps reference HCA
  routing at 84.0%, hot-token false positives at 0.9%, query bank conflicts at
  0.0%, and rare stress coverage at 100.0% split-rare / 98.4% repeated-name.
- The first HCA-summary quality check says the same 4KB global summary is good
  enough for threshold routing but not yet for fine dense-topic ranking:
  top-256 recall is about 94.1%, while top-64 recall is only about 42.2%.
  An 8KB version reaches 100% top-256 recall but still only about 51.6% top-64,
  suggesting 4-bit saturation rather than only insufficient width.
- Periodic decay is the first anti-saturation fix. With the same 4KB summary,
  decay intervals from 64 to 512 tokens recover 100% top-64/top-256
  decayed-topic recall and 100% route accuracy when the decayed-state threshold
  is adjusted to 2. Decay every 256 tokens costs about 32 decay-cell touches per
  token if counted synchronously, so the next step is not just quality but
  maintenance scheduling or scale metadata.
- Lazy epoch decay removes that synchronous maintenance traffic. A 4-bit
  `width=2048` summary with 16-bit per-counter epochs uses about 20KB instead
  of 4KB and reads about 10B/query instead of 2B/query, while preserving 100%
  top-64/top-256 decayed-topic recall and 100% route accuracy in the current
  trial.
- The lazy epoch metadata can be reduced to 8 bits for this 65k-token window:
  at decay interval 256, state is about 12KB, read width is about 6B/query, and
  top-64/top-256 recall plus route accuracy remain 100%. Four-bit epoch metadata
  reaches 8KB and 4B/query but loses dense-topic quality at the longer decay
  intervals needed to avoid epoch wrap.

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
- A learned 16x16 candidate-scorer LUT over dense estimate and cache score is a
  negative result: it uses 128 bytes but reduces mixed synthetic topic@64 from
  about 67.1% to about 64.6%. Dense-min scoring remains the baseline.
- Candidate scoring reads are now counted explicitly. The gated synthetic run
  uses about 179.6 score cells/event, raising the unified local profile to about
  51.46KB/event.

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
  51.46KB per synthetic event including gated online candidate-cache updates
  and candidate scoring reads.
- The wide64 CSA/HCA profile raises this to about 52.10KB/event and about
  707.8KB of on-chip state.
- The compact128 CSA/HCA profile raises local traffic to about
  52.28KB/event but lowers on-chip state to about 451.8KB.
- The previous rare128 CSA/HCA profile keeps local traffic about 52.28KB/event
  and lowers on-chip state further to about 354.6KB.
- The current joint128 profile adds learned probe/fanout control metadata to
  rare128. It keeps local traffic about 52.28KB/event and raises on-chip state
  only to about 356.9KB.
- The current retire128c3g3 profile adds the online `count1_retire15` counting
  Bloom sidecar with 3-bit counters and the selective zero-overlap three-entry
  fanout guard. It still keeps normal reference local traffic about
  52.28KB/event, and keeps on-chip state about 392.8KB.
- The tiny Transformer KV reference at 16k context reads about 384MB per token.
- This is a design-budget signal, not an energy or quality-equivalence claim.

Tile/floorplan profile:

- The first chip mapping proxy uses 64 cells/tile, 16KB local SRAM/tile, and 32
  local bytes/cycle/tile.
- With the current retire128c3g3 CSA/HCA-aware state, a 32-tile fabric now fits at
  about 76.7% SRAM utilization and requires 25 16KB state tiles.
- A 64-tile fabric stores the same state in about 38.4% of local SRAM.
- At a 1M synthetic events/s target, aggregate local bandwidth utilization is
  about 5.2% on 32 tiles and about 2.6% on 64 tiles under the proxy assumptions.
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
- within-block and repeated-read scoring after compressed block routing, because
  the first block-index result finds relevant blocks but covers only a small
  fraction of repeated occurrences.
- a learned read-budget policy that decides when to spend additional sparse
  block reads versus when to use the compressed dense/HCA-like state.
- quality tests for the HCA-like summary, because the hand threshold only saves
  reads if the dense recurrent path preserves enough high-frequency evidence.
- anti-saturation HCA summaries: decayed counters, group scale metadata,
  per-topic residual summaries, or selected higher-precision frequency channels.
- replace hand-set HCA decay/threshold with learned or metadata-driven
  scale/decay control and account for maintenance traffic in the unified
  event profile.
- compress lazy-decay metadata, for example 8-bit epochs at longer intervals,
  per-tile shared epochs, or group scale/offset state.
- promote 8-bit lazy epoch HCA summary to the default HCA baseline for the next
  unified event-profile update.
- continue compressing or tiering the CSA block-summary index beyond the current
  rare128 point, because learned rules and richer states still need SRAM
  headroom.
- train or hand-design a delayed-promotion gate with stronger local evidence.
  The first probation diagnostic shows that persistent first-hit presence
  pollutes the hot path, while deletable first-hit probation is promising but
  spends extra state/read bandwidth; the next candidate should use
  directory-probe feedback, short recency, or source phase to approach the
  oracle feedback row without permanent Bloom pollution.
- add recency/query-context features to the trained fanout LUT so the
  zero-overlap guard can distinguish true repeated-key/spread rare misses from
  harmless CSA disagreement; the current `retire128c3g3` budget proves the
  guard fits normal reference traffic.
- improve the trained HCA route table with recency/topic/context metadata or a
  recall-weighted objective after the presence-bit baseline is fixed.
- train a joint admission/probe/fanout policy so HCA threshold, exact-directory
  override, delayed sidecar promotion, and read budget are optimized as one
  hardware control table.

## Phase 2: Trainable Continuous HARC-CA

Add PyTorch or JAX when the environment allows dependency installation.

Implement:

- continuous cell state;
- grouped local / summary / route / memory-IO / stability channels, using the
  measured `mhc_grouped` propagation rule as the first hand-coded scaffold;
- shared local update rule;
- Cellular-MoE rule banks with bounded low-cost routing;
- residual bounded updates;
- random asynchronous update masks;
- auxiliary routing losses;
- route-bias control inspired by auxiliary-loss-free load balancing;
- multi-token / multi-tick prediction heads;
- tiny Transformer teacher for distillation experiments.
- optional Muon-style optimizer experiment for the shared recurrent rule.
- extend the 1,000-tick random-state stability sweep into trained-state sweeps
  that optimize for content entropy preservation, not just avoiding zero
  collapse or saturation.
- train a content-to-carrier write gate. The content-retention diagnostic shows
  that a persistent content lane is necessary, but fixed refresh spends local
  writes inefficiently; the next rule should learn when the mHC carrier needs
  content exposure.
- promote the local `mismatch_ge8` gate into a tiny learned write-gate LUT using
  content-carrier mismatch, route activity, and envelope level as features; use
  the budget_top rows only as an upper-bound target, not as hardware behavior.
- improve the learned write-gate objective. The first 8-byte LUT only recovers a
  threshold-like policy, so the next labels should be task-weighted by active
  route/query demand instead of pure carrier reconstruction error.
- replace synthetic random demand in the learned demand-gate sweep with real
  route/retrieval demand from the dual-path synthetic LM and rare-directory
  workloads. Rare-directory and synthetic exact-query traces now work.
- extend the rare-directory trace-gate result into the dual-path synthetic LM.
  Exact-memory demand now works; mixed exact+candidate demand shows candidate
  output rows dominate write traffic.
- prune candidate-output demand before content exposure. A candidate-row sweep
  shows the useful target is roughly 8-16 demanded candidate rows per topic
  event: 8 rows cost about 0.029 writes/token/tick, 16 rows cost about 0.049,
  while 64 rows costs about 0.178.
- keep the phase/rank/mismatch exact exposure gate as the content-lane rule.
  The 9-byte LUT reaches 100.0% demanded exactness on the candidate sparsity
  sweep and avoids the sparse 2-row/4-row misses from the generic learned LUT.
- build a real low-bit candidate reducer in front of that gate. The first
  topic-phase reducer now produces top-M demand from local scores: top-16 keeps
  82.8% of top-64 topic-hit quality at 28.8 channel writes/event, and top-32
  keeps 91.7% at 58.1 writes/event.
- replace full-pool candidate scoring with hierarchical or bank-local top-k.
  The current reducer still reads all 512 candidate rows, or 2,048 low-bit score
  cells per topic event, before choosing top-M.
- model group-summary update cost. The first hierarchical reducer cuts score
  reads to 256 cells/topic for top-16 and 384 for top-32, but it assumes local
  group max summaries are already maintained.
- move group-summary maintenance from exact recompute to learned/lazy updates.
  Exact maintenance for 16-row groups costs about 234 cells/topic and still
  preserves a 76% net top-16 scoring reduction, but lazy dirty summaries may
  reduce this further.
- replace fixed lazy refresh intervals with a local refresh trigger. Refresh-16
  keeps top-16 at 84.0% of top-64 quality while reducing total score work to
  364 cells/topic; a learned trigger should recover refresh-4 quality with
  closer to refresh-16 cost.
- validate triggered refresh against fixed refresh. `dirty_count_or_age` now
  gets refresh-4-like top-16 quality at about 421 cells/topic, below fixed
  refresh-4 but above fixed refresh-16. Next, test learned trigger thresholds
  and adversarial topic bursts.

First trainable target:

```text
Learn the routing decision and candidate scoring currently hand-coded in the
synthetic next-token benchmark.
```

This target now includes learning a candidate output policy that avoids
full-vocabulary scoring for most events.

Parallel wiki-memory target:

```text
Build a CA-native external knowledge fabric before attempting a full CA-only
general LLM.
```

The first version should store facts as local page/link cells, keep low-bit
summaries per page and block, propagate dirty/version changes by local triggers,
and benchmark update latency plus multi-hop recall against flat RAG and
attention-over-context baselines. This is a better first product-shaped target
than replacing the entire Transformer stack, because mutable knowledge is where
local CA storage, versioning, and sparse routing are naturally strong.

The first synthetic wiki-memory benchmark now includes contradiction clusters.
With truth/memory separated, `trigger16_age16` reaches 94.73% recall at about
356 cells/query and 14,466 cells/update, versus 100.0% recall and about 20,255
cells/update for exact update refresh. Page-local error-book repair raises
recall to 97.66% and repeated failed-probe recall to 98.54% at about 14,739
cells/update. Cluster repair reaches 100.0% checked multi-source consistency at
about 14,914 cells/update. A flat/RAG-style all-page-summary scan matches those
accuracy points but costs about 1,061 cells/query, while the CA hierarchical
route costs about 356-357 cells/query. Scaling to 2,048 pages widens the gap:
CA costs about 804 cells/query while flat scan costs about 8,228, a 90.2%
reduction. The density sweep then finds the boundary: at 1,024 pages and
width-256 summaries, CA recall falls from 98.83% at four facts/page to 19.92%
at 32 facts/page, while flat scan stays near 99.8% but reads more than 4K
summary cells/query. The first adaptive group-fanout version repairs the
16 facts/page stress point: `g4_max32_margin1` reaches 99.80% recall at about
1,991 cells/query, versus 2,445 for fixed 32-group routing and 4,237 for flat
page-summary scan. The learned fanout LUT improves that to the same 99.80%
recall at about 1,560 cells/query with 1.1KB of table state. The
conservative `t100` table is the current default because it is more stable
across checked seeds while still reading only about 1,566 cells/query on the
default stress point. The learned fanout grid now shows where this holds:
8 facts/page works across 512-2,048 pages, 16 facts/page works through 1,024
pages and then hits the 32-group cap, and 32 facts/page needs stronger summaries
or a page-internal second stage. The next step is to add that dense-page second
stage rather than only increasing fanout.

The first dense-page stage is now smaller routing tiles: four-page groups with a
learned max48 fanout LUT. It fixes the 1,024-page, 32 facts/page case at
99.80% recall and about 1,697 cells/query, and the 2,048-page, 32 facts/page
case at 99.22% recall and about 2,897 cells/query. The next step is to make the
tile size itself density-aware and then test mixed sparse/dense wiki regions.

That density-aware version now exists for a two-region mixed wiki. It keeps
16-page tiles for sparse 8 facts/page regions, conditionally enables four-page
tiles for dense 32 facts/page regions, and uses a local quality guard to avoid
the small-region regression at 25% dense pages. At 50-75% dense pages it
recovers flat-level recall while reading 65-73% fewer cells than flat scan. The
next step is to replace the region-level density oracle with per-block density
tags generated during normal summary refresh.

The refresh-derived density tag version now exists. Two-bit tags from summary
refresh identify sparse tag-1 and dense tag-3 regions, but tag-only switching
can still regress recall on small dense regions. The current policy is therefore
tag plus paired online guard. The current NumPy guard uses a 128-query /
64-update probe window, presents the same queries to baseline and dense-tile
routes, requires at least a 2-point recall gain, and blocks any dense loss. It
rejects the 25% dense small-region case, where the probe has 0 wins and 0
losses, and enables the 50% and 75% dense cases, where it sees 20 and 42 dense
wins with zero losses. The next step is to make the counters persistent per
density-tagged block and then learn the tag/guard threshold instead of
hand-setting it.

The first NumPy version of this target is the learned admission LUT. It is not a
neural CA yet, but it proves the hand-set threshold can be replaced by a tiny
trainable low-bit rule.

The first learned candidate scorers did not beat the dense-min baseline. A
future-window teacher with dense-score residual slightly improves the standalone
topic stream, but still fails in the mixed synthetic LM. The next version should
add richer local features such as source phase, recency, contamination counters,
multi-tick state, or distillation from a stronger scorer rather than only
changing the label on `(dense estimate, cache score)`.

The first source-phase feature is now implemented as a separate topic-only
scoring sketch. It improves un-gated candidate ranking but does not beat the
current admission-gated dense baseline. The next scorer should combine source
phase with recency and contamination counters instead of treating phase
separation as a standalone replacement.

The first source/cache combination is also measured. The fixed
`2 * topic_score + cache_score` rule improves the noisy online always-admit
path, but still loses to the admission-gated dense baseline. This makes the next
target more specific: learn a local indexer over `(dense score, topic score,
cache score, gate estimate, recency/contamination counters)` instead of
hand-selecting one score formula.

The first learned versions of that target are now a 3.0-byte signed 4-bit
linear indexer and a 40.5-byte additive feature LUT over `(dense, topic, cache,
contamination, age)`. They are hardware-small but do not beat the fixed
topic-cache formula. The next version should use a richer objective, pairwise
distillation from an oracle scorer, or a less factorized pairwise/tensor LUT
rather than only a linear or additive update.

The feature-collision ceiling narrows that next step. Online always-admit has a
large resident/feature-ceiling gap even after adding 4-bit age, so it needs
finer local state such as shorter-horizon recency, age deltas, or a finer source
phase. Gated mode has almost no resident/feature-ceiling gap, so it needs better
ranking supervision rather than more scalar features.

The full 5D tensor LUT diagnostic rules out a naive dense tensor as the next
move. The table is large, sparse, and does not beat the hand baseline. The next
experiment should use pairwise distillation or a shared/factorized tensor that
can generalize across sparse feature tuples.

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
