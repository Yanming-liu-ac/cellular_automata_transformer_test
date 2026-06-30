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

The first learned candidate scorer is a negative result. A 16x16 signed 4-bit
LUT over `(dense estimate, cache score)` uses 128 bytes and matches the dense-min
baseline on the standalone topic stream, but drops synthetic-LM topic@64 from
about 67.1% to about 64.6%. The current baseline therefore keeps dense-min
candidate scoring. The important accounting correction is that shortlist ranking
now explicitly counts dense-sketch reads: the gated synthetic run needs about
179.6 candidate score cells/event.

## Event-Level Efficiency Profile

The current prototype can be profiled as a decode event:

```text
event traffic =
    exact sparse-memory reads
  + dense sketch counter updates
  + sparse Cellular-MoE rule-bank local reads/writes
  + online candidate-cache updates, admission-gate reads, and shortlist scoring reads
  + candidate output-head scoring
```

With gated online candidate generation and 4 Cellular-MoE ticks per synthetic
decode event, the current deterministic profile estimates about 51.46KB of local
on-chip byte movement per event. The paired tiny Transformer KV-cache reference
reads about 384MB per token at 16k context.

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

At 4 Cellular-MoE ticks per synthetic event, the current event profile needs
about 51.46KB of local traffic and about 183.8KB of on-chip state. With a
32-tile fabric under the proxy assumptions, this state occupies about 35.9% of
local SRAM and a 1M events/s target consumes about 5.1% of aggregate local byte
bandwidth.

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
