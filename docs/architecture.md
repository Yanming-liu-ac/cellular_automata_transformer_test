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

## Immediate Falsification Tests

HARC-CA should be rejected or redesigned if:

- multiscale routing does not materially reduce propagation steps;
- low-bit rollout is unstable for hundreds or thousands of ticks;
- algorithmic tasks require dense global updates every token;
- associative retrieval degenerates into scanning all memory cells;
- the output head dominates energy and bandwidth;
- tiny Transformer baselines win at the same memory-movement budget.
