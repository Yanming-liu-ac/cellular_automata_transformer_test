# HARC-CA Architecture

## Objective

Design a CA-native architecture for large language modeling and AI chips. The
architecture should avoid Transformer-style global attention as the primitive.
It should instead use local recurrent updates, persistent state, multiscale
communication, and low-bit rule execution.

## Why A Plain CA Is Not Enough

A one-dimensional radius-1 cellular automaton has a strict light-cone limit:
information moves at most one token per update step. A context of length `N`
therefore needs `O(N)` steps for the last token to interact with the first token.
That is not competitive with attention hardware.

The architecture must keep the CA principle but change the lattice:

- local wires only;
- fixed repeated rule;
- no all-to-all attention bus;
- multiscale cells embedded physically near the token cells they summarize;
- associative routing to recover selected details when a compressed summary is
  not enough.

## Candidate: Hierarchical Associative Recurrent CA

HARC-CA uses a typed but rule-shared lattice:

```text
level L:              [summary of 0..N)
                       /              \
level L-1:      [0..N/2)              [N/2..N)
                 /    \                /     \
level 0:     token token ...      token token ...
```

Each node is a cell or small tile of cells. Parent-child links and same-level
neighbor links are physically local in a folded layout. Cell type is encoded as
state bits, so the silicon can still use one shared update datapath.

DeepSeek's efficient-LLM design suggests a stricter version of this candidate:
HARC-CA should combine compressed latent state, sparse rule execution, bounded
routing, and exact associative memory. It should not activate the whole fabric
with every rule at every tick.

DeepSeek-V4's CSA/HCA split sharpens the HARC-CA memory design:

```text
compressed recurrent CA field  -> dense causal context
sparse associative lane        -> exact rare facts and long-range copy
```

The architecture should not force one memory path to solve both problems.
The V4 report is therefore best treated as the closest external systems
prototype, not as the target architecture itself. HARC-CA keeps the lesson but
changes the primitive: attention/index kernels become local state updates,
bounded route waves, and low-bit learned control tables.

## Cell State

Each cell stores a fixed-width low-bit state:

```text
content bits     semantic / latent features
route bits       query, key, gate, phase, and message tags
memory bits      persistent trace for context compression
clock bits       local update phase and convergence state
health bits      saturation, reset, and uncertainty markers
```

The first software model can use continuous `float32` states for training. The
hardware target is 1-bit, 2-bit, or 4-bit bit-sliced state with saturating update
and lookup-table micro-rules.

V4's grouped compression idea suggests a grouped state layout:

```text
local channels      short-range syntax and immediate token dynamics
summary channels    compressed dense causal context
route channels      query, key, gate, and phase information
memory IO channels  interface to exact associative lanes
stability channels  norms, uncertainty, saturation, and reset signals
```

The deployment target is still low-bit, but routing tags and stability metadata
may need more precision than ordinary latent state channels.

## Update Rule

At each tick, every active cell runs the same local rule:

```text
neighborhood = self + left + right + parent + children + optional diagonal tile links
message      = perceive(neighborhood)
delta        = rule(message, cell_type, phase)
state'       = saturating_residual(state, delta)
```

The continuous training form can be a tiny shared MLP or depthwise-separable
convolution. The deployable form should compile into:

- LUTs for small binary/ternary rules;
- XNOR-popcount for bit-vector similarity;
- saturating add / majority / mux for state updates;
- local SRAM/register-file reads only.

Following the DeepSeekMoE lesson, the shared rule can evolve into a
**Cellular-MoE**:

```text
cell state + phase + local features -> select k local rule banks
```

The rule banks remain local and low-bit. The router must be bounded and
load-balanced, with routing bias separated from content score so load control
does not erase modeling quality.

The first Cellular-MoE prototype uses six hand-written integer rule banks:

- preserve;
- decay;
- diffuse;
- sharpen;
- copy from left;
- copy from right.

Only active cells route to one rule bank per tick. A bias-control loop adjusts
rule scores from observed load, mirroring DeepSeek's auxiliary-loss-free load
balancing idea at CA execution level. In the current deterministic rollout,
updating 20% of cells with one selected rule gives about 30x fewer rule updates
than executing all six rules on every cell. Bias control reduces rule-load
coefficient of variation from about 1.23 to about 0.74.

This is an execution-shape result, not a learned model result. The next step is
to learn the router and rule banks while preserving the sparse low-bit schedule.

## Fast Information Propagation

HARC-CA gets fast propagation from the hierarchy:

- local details spread across nearby tokens in `O(distance)`;
- block summaries move upward in `O(log N)`;
- global context broadcasts downward in `O(log N)`;
- associative requests route through summaries and descend only into promising
  blocks.

This is not equivalent to exact attention over all tokens. It is a different
model class: compressed persistent context plus selective retrieval.

## Associative Retrieval

Language modeling needs exact or near-exact recall for names, numbers, code
symbols, and references. Pure compression will fail on some of these.

HARC-CA adds content-addressed routing:

1. token cells maintain low-bit key traces;
2. summary cells maintain compact sketches of descendant keys;
3. a query wave travels up the hierarchy;
4. routers compare query bits with sketches using XNOR-popcount or Hamming
   distance;
5. only high-match branches receive a descending read wave;
6. retrieved details are merged back into the active generation region.

This keeps communication local while making long-range recall sublinear in the
context length for sparse matches.

The first concrete retrieval lane is a **multi-route set-associative CAM**:

```text
query key -> hash route 0 -> bucket A -> 4 low-bit tag compares
          -> hash route 1 -> bucket B -> 4 low-bit tag compares
          -> optional more routes / overflow tier
```

This is a "power of multiple choices" memory. It keeps the same stored capacity
but gives each key several physically local landing buckets. That reduces bucket
hot spots and evictions, at the cost of more local route waves and more tag
compares per query.

This lane is not enough by itself for an LLM. It is a candidate primitive for the
facts that must be recalled exactly while the recurrent CA state handles fuzzy
context integration.

This is the CA analog of DeepSeek-V4's sparse/dense split: CSA-like sparse
retrieval for exact details, HCA-like compressed recurrence for dense history.

## Compressed Dense Context

The first compressed dense-context prototype is a low-bit decayed count-sketch.
Each token updates a few hash-routed counters, and the counters decay
periodically. This is not a language model and not exact memory; it tests whether
a small CA-local state can preserve coarse topic and recency distribution.

Prototype behavior:

```text
token -> 4 hash-routed low-bit counter updates
periodic integer decay
readout -> approximate topic / recency distribution
```

On a 65k-vocabulary topic stream with 65k context, a 4-bit sketch with
`banks=4` and `width=2048` uses 4KB of state and recovers the exact top-64
decayed topic tokens in the current deterministic trial. A denser exact 4-bit
counter table for the whole vocabulary would use 32KB. This is an 8x state
reduction for this narrow dense-context task.

The correct interpretation is:

```text
compressed sketch = fuzzy dense background
associative lane  = exact rare details
```

The sketch alone cannot preserve names, numbers, or code symbols reliably.

## Compressed Block Indexing

The CSA-like sparse context path is now modeled explicitly. The context is split
into fixed-size blocks, and each block-cell stores only a low-bit count-min
summary. A query token is broadcast to the block cells; each block computes a
local match score from its summary; only the top-scoring blocks are read as
token/KV blocks.

Prototype behavior:

```text
context token -> update 4 local summary counters inside its block
query token   -> every block scores itself from 4 low-bit counters
selector      -> read top-k blocks plus a short exact tail window
```

On the current 65k-context topic/noise stream with 64-token blocks, a 4-bit
block index with `banks=4`, `summary_width=256`, and 8 selected blocks uses
about 512KB of block-summary state. It reaches 100% block-hit rate on relevant
queries in the deterministic trial, including the measured cold-token subset,
while reading about 640 token positions instead of all 65,536 positions. That is
about a 102x token-read reduction before considering Transformer KV width.

This is not an attention-quality result. It shows that a CA fabric can cheaply
route to plausible context blocks. The occurrence coverage is only about 8.4%,
so downstream attention, exact associative recall, or repeated query waves still
need to decide which details matter inside and across the selected blocks.

A follow-up budget sweep keeps the same 512KB compressed block index and varies
the number of selected blocks. At 32 selected blocks plus the 2-block tail, the
path reads about 2176 token positions and covers about 22.1% of exact
occurrences, still a 30x token-read reduction. At 128 selected blocks, coverage
rises to about 46.1% while reduction falls to about 7.9x. The gap to exact
top-block selection stays below about 0.3 percentage points. That means the
current bottleneck is not block ranking; it is the amount of high-frequency
history one can afford to reread.

The first explicit CSA/HCA read policy adds a tiny global 4-bit summary before
block scoring. If the global summary says a query token is frequent, the query
uses the HCA-like recurrent/dense path and skips block scoring; otherwise it
uses a small CSA block read. With a 4KB global summary and threshold 8, the
current deterministic stream routes all measured hot relevant queries to HCA and
all measured cold relevant queries to CSA. Average block-score traffic drops
from 2KB/query for fixed block scoring to about 300B/query, and average token
block reads fall to about 165 token positions per query. This is a 396x
full-context token-read reduction, but it assumes the HCA summary can handle
the high-frequency distributed evidence.

The next block-state sweep tests the SRAM cost directly. Keeping the same HCA
gate and `csa_blocks=4`, `block_size=128` with `summary_width=256` cuts the CSA
block-summary state from 512KB to 256KB. In the current deterministic trial it
preserves 100% measured CSA-path hit and coverage on routed relevant queries,
while block-score traffic falls from about 300B/query to about 150B/query. The
price is larger selected blocks: average token block reads rise from about 165
to about 331 positions/query, still about a 198x full-context token-read
reduction. Smaller 128KB/64KB points begin to lose too much cold exact recall in
this stream.

The rare-token directory sweep uses the exact sparse lane to repair that
low-state point instead of widening every block summary. With `block_size=128`,
`summary_width=128`, threshold 15, and six exact directory block ids per rare
token, CSA block-summary state is only 128KB and the directory adds about
30.8KB. The combined 158.8KB CSA state restores the routed CSA subset from
68.9% hit and coverage to 100% hit and 100% coverage in the current trial.
Directory reads add only about 0.48B/query on the reference stream, while
selected token reads stay about 332 positions/query. This is a cleaner CA split:
HCA handles frequent distributed context, the low-width CSA summary proposes
blocks, and a tiny exact directory repairs rare block ids.

The stress sweep is the caution. At the older threshold 8, bursty or repeated
rare tokens can be falsely routed to HCA before the directory is consulted.
Raising the HCA gate to threshold 15 reduces rare false-HCA routes to about
0.8% in the current synthetic stress cases. `dir_k=2` is enough for burst and
three-way split rare tokens, but repeated-name tokens spread across six blocks
need `dir_k=6` to recover about 99.2% measured coverage. In pure rare-query
stress, token-read reduction falls to roughly 52x-86x because more exact blocks
are intentionally read; that is the expected worst-case cost of preserving rare
details.

There is also a conservative guard mode. If the exact rare-token directory is
probed before HCA admission, a directory hit overrides the HCA route and forces
CSA. On the repeated-name stress case, `threshold=8` without the guard has 75%
rare false-HCA and only 25% coverage. `threshold=8` with the guard removes those
false-HCA routes and recovers 100% coverage, at the cost of one small directory
probe per query. The cheaper default remains `threshold=15` without the guard;
the guard is the higher-recall mode for exact-sensitive workloads.

The final hand-policy diagnostic separates stored fanout from read fanout. A
directory can store up to six block ids for a repeated rare name but read only
two when the token metadata says it is compact. This saves reads, but if applied
blindly to repeated names it drops coverage to about 68%. Reading all six
recovers about 99-100% coverage. So the real policy should not be a fixed
`dir_k`: it should choose HCA threshold, guard, and directory read fanout from
small per-token metadata.

The first metadata-driven fanout proxy uses a two-level rule: start with a
base read fanout of two, then expand when the stored rare-token block ids span a
large part of the context. On repeated-name stress, `guard_t8_span2to4` lifts
coverage to about 93.0% at 13.0B/query of directory reads, while
`guard_t8_span2to5` reaches about 98.4% at 16.25B/query. Full
`guard_t8_span2to6` still reaches 100.0% at 19.5B/query. This is not yet a
trained router, but it proves the control signal can be compact directory
metadata rather than a transformer-like dense attention pass.

The next sweep trains that control signal. A 42B low-bit LUT indexed by
directory entry count, span class, and CSA-overlap is trained from
self-supervised coverage labels. With guarded threshold-8 routing it reaches
about 98.4% repeated-name coverage at 12.87B/query, while split rare tokens stay
at about 99.7% coverage with only 6.50B/query. This is the first concrete
trainable control-plane block for the exact sparse lane: training changes a tiny
metadata table, not a dense attention mechanism.

The first joint control sweep adds a second 40B HCA-confidence probe LUT. It
uses the HCA bank counter pattern, not token identity, to decide whether an
HCA-routed query needs a rare-directory probe. Strong reference hot tokens have
all HCA banks saturated and no spread, so `confidence_probe` skips the directory
and keeps reference traffic at 0.50B/query instead of 3.25B/query. On
repeated-name stress it still probes about 74.2% of queries, gets about 97.7%
coverage, and spends 12.77B/query. The remaining 0.8% false-HCA rate is now an
explicit probe-LUT recall/traffic tradeoff rather than hidden behavior.

Sweeping the HCA threshold after joint control is available changes the
recommendation. Threshold 6 is too permissive: split-rare coverage collapses in
the current stress generator. Thresholds 8, 10, 12, and 15 keep almost the same
rare coverage, but higher thresholds reduce early probes. At threshold 15,
`confidence_probe` has 0.0% early probe rate on split-rare and repeated-name
stress, keeps about 98.7% split-rare and 98.3% repeated-name coverage, and keeps
reference directory traffic at 0.50B/query. So the current exact-recall mode is
again threshold 15, but now with learned probe/fanout control rather than a
fixed no-guard rule.

The first trained HCA route LUT removes the explicit threshold from inference.
It is a 40B table over the same HCA bank metadata and activates only one HCA
route bucket in the current stress set. It preserves reference HCA routing and
keeps reference directory traffic at 0.50B/query, while getting about 99.0%
split-rare coverage and 97.7% repeated-name coverage. That is close, but still
slightly weaker than threshold-15 plus learned fanout. The conclusion is useful:
the hand threshold can be represented as a tiny CA-local table, but the next
route LUT needs richer metadata or a recall-weighted training objective before
it should replace the current joint policy.

Adding one rare-directory presence bit to that route table is the first useful
fix. The directory-aware route LUT is still only 80B, keeps the same 84.7%
reference HCA route rate, and uses a 0.125B/query sidecar read. In the current
stress sweep it removes the remaining rare false-HCA routes, reaches 100.0%
split-rare coverage, and reaches 98.4% repeated-name coverage at about
13.00B/query directory traffic. This is the better CA control-plane shape:
HCA admission should see a tiny exact-memory sidecar, not only the compressed
HCA counters.

A Bloom-like sidecar false-positive sweep makes the hardware tradeoff explicit.
At 1% target false-positive rate, the sidecar is about 10.8KB in the reference
case and reference HCA routing drops modestly from 84.7% to 82.1%. At 10%, the
sidecar is about 5.4KB and reference HCA routing is still about 80.0%. At 25%,
reference HCA routing falls to 46.3%, so the hot path starts losing its point.
Rare recall remains safe in this sweep because false positives route extra
queries to CSA, not HCA.

The concrete Bloom-sidecar sweep turns that into an SRAM/read-port candidate.
With `8 bits/entry`, `k=3` hashes, and 8 banks, the sidecar is about 8.8KB on the
reference case, reads 3 bits/query, writes 3 bits per rare-directory insertion,
and keeps reference HCA routing at 84.2%. The same setting keeps split-rare
coverage at 100.0% and repeated-name coverage at 98.4%. Increasing `k` reduces
false positives but raises read traffic and bank conflicts, so the CA control
plane now has a concrete layout knob instead of an abstract "presence bit."

The first hash-salt robustness check shows why the sidecar cannot be treated as
a passive data structure. For the `8 bits/entry, k=3, 8 banks` candidate, 16
salts on the reference stream average 82.9% HCA routing, but range from 79.7%
to 84.6%. The worst salt has about 5.9% hot-token sidecar false positives. The
CA compiler or training loop should therefore pick hash salts and bank mappings
with the HCA hot path in the objective.

Bank mapping gives a layout-side fix for one part of that problem. Keeping the
same `8 bits/entry, k=3` Bloom sidecar and 16 salts, modulo banking has about
36.3% average query bank conflict, while assigning each hash function to its own
bank (`by_hash`) removes same-query bank conflicts without changing false
positives or HCA routing. This is a CA-chip-friendly result: some efficiency
comes from the memory fabric layout, not from adding model state.

The first HCA-summary quality check weakens that assumption in a useful way. A
4KB global 4-bit summary is good enough for the threshold-8 routing decision in
the deterministic query stream: query route accuracy is 100%, with no false HCA
routes or missed HCA routes. But it is not yet a strong dense semantic state.
Its top-256 frequency recall is about 94.1%, while top-64 recall is only about
42.2%; an 8KB version reaches 100% top-256 recall but still only about 51.6%
top-64 recall. The likely culprit is 4-bit saturation among very frequent
tokens. The next HCA path needs decay, scaling, grouped summaries, or slightly
higher-precision metadata if it must preserve fine dense-topic order.

The first anti-saturation fix is simple periodic decay. Keeping the same 4KB
global summary but decaying counters every 256 tokens removes saturation in the
current stream, recovers 100% top-64 and top-256 decayed-topic recall, and keeps
the threshold route accurate when the decayed-state threshold is lowered to 2.
The cost is about 32 decay-cell touches per token if counted synchronously. This
looks like the right HCA direction: low-bit recurrent state should be decayed or
scaled, but the decay interval and routing threshold should become learned or
metadata-driven rather than hand fixed.

The next implementation removes the synchronous sweep. A lazy-decay HCA summary
stores a small epoch next to each low-bit counter and applies the right shift
only when that counter is read or updated. On the same 4KB-counter, 256-token
decay setting, 16-bit epoch metadata raises summary state to about 20KB and read
traffic to about 10B/query. In exchange it removes the 32 decay-cell touches per
token while preserving 100% top-64/top-256 decayed-topic recall and 100% route
accuracy in the current deterministic trial. This is a plausible CA-chip
tradeoff: spend local SRAM metadata to avoid global maintenance waves.

The metadata can be compressed. An 8-bit epoch is enough for the current
65k-token window at decay interval 256, reducing the lazy HCA state from 20KB to
12KB and read traffic from 10B/query to 6B/query while preserving 100%
top-64/top-256 decayed-topic recall and 100% route accuracy. A 4-bit epoch can
cut the state to 8KB, but it requires longer decay intervals and starts losing
dense-topic quality. The current best hand point is therefore 4-bit counters
plus 8-bit lazy epoch metadata, not the earlier 16-bit metadata baseline.

## Training Stability

A recurrent CA can become chaotic, die out, or converge too early. The software
training rule should therefore include:

- residual updates with bounded step size;
- state norm or entropy regularization;
- random asynchronous update masks during training;
- variable rollout lengths;
- auxiliary losses for memory, routing, and algorithmic tasks;
- curriculum from short contexts to long contexts;
- distillation from a small Transformer only as a teacher signal, not as the
  hardware primitive.

DeepSeek's multi-token prediction result suggests an additional CA-native loss:
train the state field to predict multiple future tokens and multiple future
state slices. This makes the recurrent dynamics plan ahead instead of only
reacting to the next symbol.

The low-bit rule should be trained with quantization-aware training and
straight-through estimators, then verified with integer-only rollout.

## Language Model Interface

Initial experiments should avoid a huge vocabulary projection:

1. byte-level or character-level prediction;
2. then small BPE vocabularies;
3. then candidate shortlist generation through CA routing plus a compact output
   head.

During autoregressive inference, new tokens are injected into the level-0 tail.
The chip runs a fixed number of relaxation ticks, then reads logits from the
active output tile.

## Hardware Shape

The target chip is a cellular fabric:

- dense grid of identical cell processing elements;
- small local register file or SRAM per cell/tile;
- nearest-neighbor and parent-child wires;
- no global attention crossbar;
- event-driven or wavefront clocking to update only active regions;
- bit-sliced arithmetic instead of large FP matrix units;
- optional nonvolatile or SRAM-based rule tables.

The key hardware bet is that moving a few low-bit messages over short wires is
cheaper than repeatedly streaming large KV-cache vectors from HBM.

DeepSeek's hardware notes also imply that the chip should support:

- fine-grained per-tile/per-channel quantization metadata;
- fused online quantization during local memory transfer;
- higher-precision accumulation or counters only on sensitive paths;
- communication offload for route waves and reductions;
- separate prefill, decode, and background-refresh schedules.

## Memory-Lane Tradeoff

The retrieval lane exposes a basic chip tradeoff:

- low load factor gives near-perfect recall but spends more SRAM cells;
- more routes improve recall at the same storage capacity but spend more local
  query work;
- overflow tiers can preserve exact facts but add routing complexity;
- tag bits reduce false positives but increase cell width.

Early experiments show that 2-route lookup at load factor 1.0 improves recall
substantially over single-route lookup, but still leaves too many misses for a
general language model. The likely useful region is lower load factor plus a
small overflow tier, unless learned routing can reduce bucket imbalance.

The first overflow-tier experiment supports this. At 16k context, a primary lane
with `buckets=context/4`, `ways=4`, and `routes=2` reaches only about 92-93%
exact recall because of primary bucket evictions. Adding a smaller overflow lane
with `buckets=context/16`, `ways=4`, and `routes=2` recovers full recall in the
current deterministic full-context trial with 32-bit tags. Average query work
increases only from about 32 visited cells to about 34 visited cells because
only about 8% of queries touch overflow.

This is a CA-native cache hierarchy: no full-context scan fallback is used.

Combined with the 4KB compressed dense-context sketch, the current dual-path
prototype uses about 166.5KB for:

- 100% exact induction recall on the deterministic 16k trial;
- 100% top-64 dense-topic recall on the deterministic 65k-vocabulary trial;
- about 34 visited cells per exact query;
- four low-bit counter updates per dense-context token.

## Synthetic Next-Token Interface

The first next-token-style prototype connects the two memory paths to a simple
prediction interface:

```text
if input is a key query:
    exact sparse lane predicts the next value token
else:
    compressed dense sketch ranks a small candidate pool for topic-like tokens
```

This is deliberately non-neural and non-trained. Its purpose is to check whether
the memory system can serve next-token behavior without falling back to full
attention or full-vocabulary dense projection.

Current deterministic trial:

- 16k exact facts in a tiered associative lane;
- 8k topic events and 4k key-query events;
- 65k vocabulary and a 512-token candidate shortlist for dense prediction;
- exact induction next-token accuracy: 100%;
- static-oracle topic candidate top-k hit rate: about 62%;
- online-cache topic candidate top-k hit rate: about 61%;
- gated online-cache topic candidate top-k hit rate: about 67%;
- online candidate-cache update hit rate: about 79%;
- average local cells touched per mixed event after counting candidate ranking
  reads: about 1393 with static candidates and about 214 with gated online
  candidates;
- combined memory: about 166KB with the static shortlist and about 168KB with
  the online candidate cache.

The correct conclusion is narrow:

```text
The dual-path memory system can be wired into a next-token interface.
It is not yet a trainable language model.
```

## Output Head Constraint

The output head can erase CA-local savings. With a 65k vocabulary, 128 hidden
channels, 4-bit weights, 4-bit activations, and 16-bit logits, a full-vocabulary
projection reads/writes about 4.13MB per event and performs about 8.39M MACs.
That is far larger than the current 51KB HARC-CA local event profile.

The first output-head proxy shows:

```text
full vocab head:               about 4.13MB/event
512-token candidate head:      about 33KB/event
512-token head + exact bypass: about 22KB/event
```

The architecture therefore needs candidate generation and exact-memory bypass as
first-class hardware paths. Exact associative hits should directly produce value
tokens when possible; dense context should rank a small candidate pool rather
than scan the whole vocabulary.

This is not free. The candidate generator must be accurate enough that quality
does not collapse, and the shortlist machinery itself must remain local.

## Online Candidate Cache

The first candidate generator removes the static hot-token oracle. It uses a
fixed-size set-associative cache:

```text
observed token -> 2 hash routes -> 4 ways each
resident entry -> token id + low-bit score + valid bit
periodic decay -> integer right shift
admission gate -> optional dense-sketch threshold before cache write
top-k readout  -> scan resident cache entries, not the full vocabulary
```

With 512 entries, 4-bit scores, 2 routes, 4 ways, and a 65k vocabulary, the
cache uses about 1.31KB. On the current topic/noise stream it reaches about 69%
top-64 hit rate after warmup while scanning zero full-vocabulary entries. When
plugged into the synthetic next-token benchmark, it keeps topic@64 close to the
static candidate pool, but adds about 6.6 local cache-cell touches per mixed
event.

Adding a threshold-1 dense-sketch admission gate improves the current synthetic
LM topic@64 to about 67%. The gate reuses the existing dense-context sketch,
admits about 61% of topic observations, raises cache-update hit rate to about
98%, and reduces candidate-cache touches to about 4.0 cells/event plus about
2.7 dense gate reads/event. This is a better chip shape: fewer noisy writes,
fewer replacements, and no full-vocabulary scan.

The first learned admission policy replaces the hand-set threshold with a
16-entry signed 4-bit LUT indexed by the dense-sketch estimate. It is trained
from a self-supervised repeat label: admit a token when it is likely to reappear
within a future horizon. In the current deterministic trial, the learned LUT is:

```text
(-8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
```

It uses 8 bytes of state and recovers the same behavior as the threshold-1 gate:
about 70.8% standalone top-64 hit rate and about 67.1% synthetic-LM topic@64.
This is not yet a full language-model router, but it shows the candidate policy
can be represented as a tiny trainable low-bit rule instead of a hand-written
constant.

The first learned candidate scorers are negative results. A 16x16 signed 4-bit
LUT over `(dense estimate, cache score)` uses 128 bytes and matches the dense-min
baseline on the standalone topic stream, but drops synthetic-LM topic@64 from
about 67.1% to about 64.6%. A second future-window teacher with a dense-score
residual improves standalone topic scoring from about 67.7% to about 68.2%, but
still drops the mixed synthetic LM to about 64.5%. The current baseline
therefore keeps dense-min candidate scoring. The important accounting correction
is that shortlist ranking now explicitly counts dense-sketch reads: the gated
synthetic run needs about 179.6 candidate score cells/event.

The lesson is specific: two scalar local features are not enough for a CSA-like
indexer once query/fact traffic pollutes the recurrent dense sketch. The next
candidate scorer needs phase/source features, multi-tick state, or distillation
from a stronger teacher rather than just a different label on the same feature
pair.

The next source-phase experiment added a separate topic-phase scoring sketch.
It is another 4-bit dense-context sketch, but it is updated only by topic-output
events and is used only for candidate ranking. This isolates the output indexer
from exact-memory query/fact traffic. It improves static candidate scoring from
about 62.1% to about 66.7% topic@64 and online always-admit scoring from about
61.4% to about 64.4%.

The follow-up combination sweep tested whether the topic-phase score should be
combined with other local signals. `dense_topic_sum` raises static topic@64 to
about 67.0%, but it doubles candidate score reads. `topic_cache` uses
`2 * topic_score + cache_score`, keeps the same single-sketch read cost as
`topic_phase`, and raises online always-admit topic@64 to about 65.8%. However,
with the current admission gate these combinations still do not beat the
default: gated dense scoring is about 67.1%, gated topic-phase scoring is about
67.0%, and gated topic-cache scoring is about 66.7%. The current default
therefore remains gated dense scoring, but source/phase/cache signals are now
measured local features for the next learned indexer.

The first trainable multi-feature indexers use signed 4-bit rules over `dense`,
`topic`, `cache`, `contamination=max(dense-topic, 0)`, and a 4-bit resident
`age` bucket. The linear rule has only 3.0 bytes of state including bias. In the
current deterministic trial it learns weights `(3, 7, 7, 2, -4)` for online
always-admit and `(-1, 7, 6, 2, -5)` for the gated path. A factorized additive
LUT uses 40.5 bytes across five 16-bin feature tables. These rules are close to
the best hand formula but not better: online linear topic@64 is about 63.4% and
additive is about 64.7%, versus about 65.8% for `topic_cache`; gated linear is
about 66.7% and additive is about 66.6%, versus about 67.1% for the current
gated dense baseline. This is still useful because it confirms that tiny local
rules can absorb source/cache/age features, but the learner needs a better
objective or a less factorized state before it can replace the hand-written
gate/scorer.

The feature-collision diagnostic shows where the next capacity should go. In
online always-admit mode, the resident-token ceiling is about 79.0%, but the
current feature tuple has an optimistic top-k ceiling of only about 70.9%; the
positive candidate shares its exact feature bucket with about 47.7 candidates on
average. This is better than the no-age tuple, but still leaves a large gap. In
gated mode, the feature ceiling is about 69.2%, essentially the same as the
resident-token ceiling, and the mean positive bucket falls to about 3.6. This
means admission gating is already doing most of the noise separation; the next
scorer should add finer recency or pairwise state mainly for the noisy online
path, while the gated path needs a better ranking objective.

The full tuple tensor diagnostic tests the opposite extreme: a dense 5D LUT over
all five 4-bit features. This table would use about 512KB at 4 bits per entry,
but the current training stream observes only 893 feature tuples in online mode
and 2878 in gated mode. It does not solve ranking. Online tensor scoring drops
to about 39.0% with log-odds and about 35.0% with rate scoring; gated tensor
rate scoring reaches about 66.4%, still below gated dense scoring. The next
indexer should therefore not be a naive dense tensor. It needs either a smaller
pairwise/tensor factorization with better sharing, or pairwise distillation from
a stronger oracle.

## Event-Level Efficiency Profile

The current prototype can be profiled as a decode event:

```text
event traffic =
    exact sparse-memory reads
  + dense sketch counter updates
  + sparse Cellular-MoE rule-bank local reads/writes
  + online candidate-cache updates, admission-gate reads, and shortlist scoring reads
  + CSA/HCA context-summary reads and updates
  + candidate output-head scoring
```

With gated online candidate generation and 4 Cellular-MoE ticks per synthetic
decode event, the current deterministic profile estimates about 51.46KB of local
on-chip byte movement per event. The paired tiny Transformer KV-cache reference
reads about 384MB per token at 16k context.

The earlier wide64 CSA/HCA context profile raises event traffic only slightly to
about 52.10KB/event, but its 512KB block summary plus 12KB lazy-epoch HCA
summary raise on-chip state from about 183.8KB to about 707.8KB. The compact128
profile uses 256KB block summaries instead. It raises local traffic to about
52.28KB/event because selected token block reads double, but it lowers on-chip
state to about 451.8KB. The rare128 profile replaces half of that block summary
with a small exact directory: context traffic remains about 52.28KB/event, while
on-chip state falls to about 354.6KB. The current joint128 profile adds the
learned probe/fanout control state to rare128 and still keeps local traffic about
52.28KB/event, with on-chip state about 356.9KB.

This is a proxy comparison, not a performance claim. It ignores model quality,
full vocabulary output cost, real SRAM/HBM energy, clocking, routing contention,
and learned-rule overhead. Its value is that it gives the chip design a concrete
budget to protect as the model becomes more capable.

## Tile-Level Mapping

The first floorplan proxy maps the current HARC-CA event profile onto repeated
local-SRAM tiles:

```text
tile = 64 low-bit cells + 16KB local SRAM + 32 local bytes/cycle
```

At 4 Cellular-MoE ticks per synthetic event, the joint-control rare-directory
CSA/HCA-aware profile needs about 52.28KB of local traffic and about 356.9KB of
on-chip state. With a 32-tile fabric under the proxy assumptions, the state now
fits in about 69.7% of available SRAM and requires 23 state tiles. A 64-tile
fabric stores the same state at about 34.9% utilization, while a 1M events/s target consumes
about 2.6% of aggregate local byte bandwidth.

This is not area/timing closure. It is the first explicit chip budget:

```text
future learned rules, richer state, and better output heads must fit inside
local SRAM and local bandwidth without falling back to global KV-style traffic.
```

## Immediate Falsification Tests

HARC-CA should be rejected or redesigned if:

- multiscale routing does not materially reduce propagation steps;
- low-bit rollout is unstable for hundreds or thousands of ticks;
- algorithmic tasks require dense global updates every token;
- associative retrieval degenerates into scanning all memory cells;
- the output head dominates energy and bandwidth;
- tiny Transformer baselines win at the same memory-movement budget.
