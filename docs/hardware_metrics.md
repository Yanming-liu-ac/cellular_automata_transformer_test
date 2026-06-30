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
- average local cells touched per event;
- total memory bytes;
- whether the benchmark uses full-vocabulary ranking.

The first benchmark intentionally uses a candidate shortlist instead of scanning
the whole vocabulary. A CA-first chip should make candidate generation explicit;
otherwise the output head can erase memory-system savings.

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
- Cellular-MoE sparse rule-bank local reads/writes;
- on-chip state bytes;
- Transformer KV-cache read volume as a reference.

The current deterministic profile uses:

- 16k exact facts;
- a synthetic mixed decode stream;
- 4-bit dense sketch state;
- Cellular-MoE with 20% active cells, top-1 routing, and 4 rule ticks per event;
- a tiny Transformer KV reference with 12 layers, 8 heads, 64 head dimension,
  and 16-bit KV cache.

Current proxy result:

```text
HARC-CA local bytes/event: about 51 KB
Transformer KV read/token: about 384 MB
On-chip HARC-CA state: about 182 KB
```

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
bytes/cycle/tile. With the current 182.5KB HARC-CA state and 51.36KB local
bytes/event, a 32-tile configuration has:

```text
total local SRAM: 512KB
state utilization: about 35.6%
bandwidth utilization at 1M events/s: about 5.1%
```

These are design-budget numbers. They do not prove timing, routing, area, yield,
or energy.
