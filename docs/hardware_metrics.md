# Hardware Metrics

HARC-CA should be evaluated with chip-facing metrics from the first prototype.

## Model Metrics

- next-token loss / perplexity;
- algorithm-task accuracy;
- context length reached at fixed rollout budget;
- number of recurrent ticks per generated token;
- fraction of active cells per tick;
- retrieval success at different distances.

## Hardware Proxy Metrics

- cell state bits;
- rule table bits or shared rule parameter bits;
- local reads per cell update;
- local writes per cell update;
- global or off-chip reads per generated token;
- estimated bit-meters moved per token;
- output-head bytes moved per token;
- active-cell updates per token;
- integer/LUT operations per token.

## Transformer Baseline Metrics

For a tiny Transformer baseline, track:

- parameter bytes;
- KV-cache bytes per token;
- attention reads per generated token;
- MLP MACs per generated token;
- measured runtime on available hardware;
- estimated memory traffic.

## Initial Target

The first useful win condition is not beating GPT-class models. It is:

```text
At a small fixed quality target on toy language and retrieval tasks,
HARC-CA moves fewer state bits per generated token than a tiny Transformer
with KV cache.
```

If that fails, the chip idea probably needs a different CA topology.

## Retrieval-Lane Metrics

For associative recall, track:

- exact recall rate;
- false-positive rate;
- bucket evictions;
- load factor;
- cells visited per query;
- full-scan avoidance ratio;
- tag bits and value bits per entry;
- overflow traffic;
- active routing steps.

The key metric is not only accuracy. It is whether exact recall can remain high
while query work grows roughly with `log(context)` plus a small number of bucket
ways, instead of `context`.

For multi-route memory, report the whole tradeoff:

```text
visited cells = routes * route_depth + routes * ways
```

More routes can reduce evictions at the same SRAM capacity, but they increase
local query work. This is acceptable only while the scan-avoidance ratio remains
large and exact recall improves enough to justify the extra local activity.

For overflow-tier memory, also report:

- primary evictions;
- overflow insertions;
- overflow evictions;
- fraction of queries that touch overflow;
- total memory bytes across tiers;
- average visited cells including overflow misses;
- tag width needed to avoid collisions.

The first useful overflow gate is:

```text
Adding a small overflow tier should recover exact recall without turning the
query into a full scan or doubling average visited cells.
```

## Dense-Context Metrics

For compressed dense context, track:

- state bytes;
- bits per counter/channel;
- update cells per token;
- decay interval and decay cost;
- top-k topic/recency recall;
- mean absolute count error;
- compression ratio against exact dense state;
- whether the exact-memory lane is still needed for rare facts.

The key distinction is:

```text
Dense context measures distribution preservation.
Associative memory measures exact fact preservation.
```

Do not use top-k dense-topic recall as evidence that exact names, numbers, or
code symbols are preserved.

## Synthetic Next-Token Metrics

For the dual-path next-token interface, track:

- exact induction next-token accuracy;
- topic candidate top-k hit rate;
- candidate pool size;
- exact visited cells per query;
- overflow query rate;
- dense update cells per event;
- candidate-cache update cells per event;
- candidate admission-gate cells per event;
- candidate scoring cells per event;
- candidate admission rate;
- candidate-cache update hit rate;
- candidate-cache replacement count;
- average local cells touched per event;
- total memory bytes;
- whether the benchmark uses full-vocabulary ranking.

The first benchmark intentionally uses a candidate shortlist instead of scanning
the whole vocabulary. A CA-first chip should make candidate generation explicit;
otherwise the output head can erase memory-system savings.

## Candidate-Generation Metrics

For online candidate shortlists, track:

- cache capacity;
- ways and hash routes;
- token bits, score bits, and valid bits per entry;
- decay interval and decay cost;
- top-k hit rate after warmup;
- cache update hit rate;
- replacements;
- admission threshold;
- learned admission LUT bytes;
- admission training label;
- admission precision and recall;
- admission rate;
- gate read cells;
- scoring read cells;
- learned scorer LUT bytes;
- resident token count;
- local cells touched per observed token;
- full-vocabulary scan count.

The first online candidate cache uses 512 entries, 4-bit scores, 2 hash routes,
4 ways, and a 65k vocabulary. It stores about 1.31KB of state and performs zero
full-vocabulary scans. Always-admit mode reaches about 69% standalone top-64 hit
rate and about 61% topic@64 in the mixed synthetic LM. A threshold-1 admission
gate reusing the dense-context sketch reaches about 71% standalone top-64 hit
rate and about 67% mixed synthetic topic@64. In the mixed benchmark, the gated
path admits about 61% of topic observations, reaches about 98% cache-update hit
rate, and costs about 4.0 candidate-cache cells/event plus about 2.7 dense gate
reads/event.

The first learned admission policy trains a 16-entry signed 4-bit LUT from a
self-supervised future-repeat label. The trained LUT uses 8 bytes and recovers
the threshold-1 behavior on the current deterministic stream:

```text
scores: (-8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
standalone top-64 hit: about 70.8%
admission precision / recall against repeat label: about 91% / 92%
synthetic-LM topic@64: about 67.1%
full-vocabulary scans: 0
```

The first learned candidate scorers test 16x16 signed 4-bit LUTs over dense
estimate and cache score. Each uses 128 bytes. The current-token repeat target
is a negative result in the mixed synthetic LM: dense-min scoring reaches about
67.1% topic@64, while the learned LUT reaches about 64.6%. A future-window
teacher plus dense-score residual improves the standalone topic stream from
about 67.7% to about 68.2%, but still falls to about 64.5% topic@64 in the
mixed synthetic LM. Dense-min remains the active baseline. The benchmark now
counts candidate ranking reads explicitly; the gated synthetic LM uses about
179.6 dense-sketch score reads per mixed event.

A source-phase scoring sketch then tests whether the scorer needs a separate
state channel instead of a different LUT label. The sketch is a second 4-bit
dense-context array updated only by topic-output events. It adds about 4KB of
state and about 2.7 local writes per mixed event. This helps when candidate
generation is noisy: static topic@64 rises from about 62.1% to about 66.7%, and
online always-admit topic@64 rises from about 61.4% to about 64.4%. It does not
improve the current gated path: gated dense scoring is about 67.1%, while gated
topic-phase scoring is about 67.0% with extra state and writes.

The first source/cache combination keeps the same topic-phase sketch but adds
candidate-cache score during ranking. `topic_cache` uses
`2 * topic_score + cache_score` and raises online always-admit topic@64 to about
65.8% without increasing score-read cells beyond the single-sketch topic-phase
path. `dense_topic_sum` reaches about 67.0% on the static candidate pool, but
doubles candidate score reads from about 1365 to about 2731 cells/event. In the
current gated path, neither combination beats gated dense scoring.

The first trainable multi-feature indexers use signed 4-bit rules over five
local features: dense score, topic-phase score, candidate-cache score,
contamination, and resident age. The linear rule has only 3.0 bytes including
bias. It learns `(3, 7, 7, 2, -4)` for online always-admit and
`(-1, 7, 6, 2, -5)` for the gated path. A factorized additive LUT uses 40.5
bytes across five 16-bin tables. These are compact rules, but not yet wins:
online topic@64 is about 63.4% for linear and 64.7% for additive, versus about
65.8% for the fixed topic-cache formula. Gated topic@64 is about 66.7% for
linear and 66.6% for additive, versus about 67.1% for gated dense scoring.

The current feature-collision diagnostic reports two additional hardware-facing
numbers: optimistic feature ceiling and positive bucket size. In online
always-admit mode, resident recall is about 79.0%, but the age-augmented feature
tuple only supports about 70.9% optimistic top-k recall because the positive
token still shares its exact low-bit feature bucket with about 47.7 resident
candidates on average. In gated mode, the feature ceiling is about 69.2% and the
average positive bucket size falls to about 3.6. This suggests the noisy online
path needs more local state, while the gated path mostly needs a better ranking
rule.

A full 5D tuple LUT is also measured as a diagnostic, not as a recommended
default. At 4 bits per entry it would use about 512KB of local state. The current
training stream observes only 893 online tuples and 2878 gated tuples, so the
table is extremely sparse. It performs poorly in online mode, about 39.0%
topic@64 with log-odds and 35.0% with rate scoring. In gated mode, rate scoring
reaches about 66.4%, close to topic-cache but still below gated dense scoring.
The metric says dense tensor capacity is not the bottleneck to add next;
generalization and ranking supervision are.

## Compressed Block-Index Metrics

For CSA-shaped context routing, track:

- context length;
- block size and block count;
- selected block count;
- exact tail block count;
- per-block summary banks, width, and bits;
- block-summary state bytes;
- global recurrent-summary bytes;
- score cells and score bytes per query;
- relevant-query rate;
- hot-token and cold-token block-hit rates;
- occurrence coverage versus oracle coverage;
- token reads per query after block selection;
- token-read reduction versus full-context reads.

The first compressed block-index benchmark splits a 65,536-token context into
1024 blocks of 64 tokens. Each block keeps a low-bit count-min summary. With
4-bit summaries, 4 banks, `summary_width=256`, 8 selected blocks, and a 2-block
exact tail, the index uses about 512KB of block-summary state plus a 512-byte
global-summary equivalent. It scores 4096 4-bit cells per query, or about 2KB of
summary reads.

In the deterministic topic/noise trial, relevant-query rate is about 87.2%.
Overall block-hit rate is 100%, hot-token hit rate is 100%, and the measured
cold-token relevant subset is also 100% for `summary_width=256`. The selected
plus tail path reads about 640 token positions per query instead of 65,536, a
roughly 102x token-read reduction.

The limiting metric is not block hit but occurrence coverage. The same setting
covers only about 8.4% of exact token occurrences, close to the oracle top-8
block coverage of about 8.3%. This means the block index is a strong routing
primitive but not a full replacement for attention quality; it must feed a
within-block scorer, exact associative lane, or repeated sparse reads.

## Output-Head Metrics

For output scoring, track:

- vocabulary size;
- candidate pool size;
- exact-bypass fraction;
- hidden channels;
- activation bits;
- output weight bits;
- logit bits;
- resident output weight bytes;
- bytes per event;
- MACs per event;
- reduction versus full-vocabulary scoring.

The current proxy uses a 65k vocabulary, 128 hidden channels, 4-bit activations,
4-bit output weights, and 16-bit logits:

```text
full vocabulary head: about 4.13MB/event and 8.39M MACs/event
512-candidate head: about 33KB/event and 65.5K MACs/event
512-candidate + exact bypass: about 22KB/event and 43.7K MACs/event
```

The output head is a separate bottleneck from KV cache. A CA-first model must
avoid full-vocabulary scoring on every event unless the rest of the architecture
has enough local budget to pay for it.

## Cellular-MoE Metrics

For sparse rule-bank execution, track:

- active cell fraction;
- selected rule banks per active cell;
- sparse rule updates per tick;
- dense-equivalent rule updates per tick;
- update reduction ratio;
- rule-load coefficient of variation;
- routing-bias range;
- saturation fraction;
- checksum or task score after rollout.

The first execution gate is:

```text
Sparse routed rule execution should reduce local rule updates by an order of
magnitude without routing all traffic into one overloaded rule bank.
```

This is a chip metric, not a language quality metric.

## Unified Event Profile

The project now includes a unified per-event proxy that combines:

- exact sparse-memory local reads;
- compressed dense-context counter updates;
- online candidate-cache updates, admission-gate reads, and shortlist scoring
  reads;
- Cellular-MoE sparse rule-bank local reads/writes;
- on-chip state bytes;
- Transformer KV-cache read volume as a reference.

The current deterministic profile uses:

- 16k exact facts;
- a synthetic mixed decode stream;
- 4-bit dense sketch state;
- 512-entry online candidate cache with 4-bit scores and threshold-1 admission;
- Cellular-MoE with 20% active cells, top-1 routing, and 4 rule ticks per event;
- a tiny Transformer KV reference with 12 layers, 8 heads, 64 head dimension,
  and 16-bit KV cache.

Current proxy result:

```text
HARC-CA local bytes/event: about 51.46 KB
Transformer KV read/token: about 384 MB
On-chip HARC-CA state: about 183.8 KB
```

With a 512-token candidate output head and exact-query bypass, output scoring
adds about 22KB/event in the current synthetic setup. A full-vocabulary head
would add about 4.13MB/event, dominating the current HARC-CA local-event budget.

This is not a measured energy claim. Local on-chip bytes and KV-cache read bytes
are physically different costs, and the current HARC-CA prototype is not quality
equivalent to a Transformer. The metric is useful because it makes the design
target explicit:

```text
Keep useful next-token behavior inside a small amount of local low-bit traffic.
```

## Tile/Floorplan Metrics

For chip mapping, track:

- cells per tile;
- local SRAM per tile;
- local bytes per cycle per tile;
- tile count;
- total local SRAM;
- on-chip state bytes;
- state utilization;
- tiles required by state;
- target events/s;
- required aggregate local bandwidth;
- peak aggregate local bandwidth under the proxy assumption;
- local bandwidth utilization;
- proxy maximum events/s.

The first floorplan proxy uses 64 cells/tile, 16KB local SRAM/tile, and 32 local
bytes/cycle/tile. With the current 183.8KB HARC-CA state and 51.46KB local
bytes/event, a 32-tile configuration has:

```text
total local SRAM: 512KB
state utilization: about 35.9%
bandwidth utilization at 1M events/s: about 5.1%
```

These are design-budget numbers. They do not prove timing, routing, area, yield,
or energy.
