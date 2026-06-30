# DeepSeek Lessons For CA-First AI Chips

This note translates DeepSeek V3/V4 design ideas into constraints for HARC-CA.
The point is not to copy Transformer internals. The point is to absorb the
engineering logic behind strong efficient LLMs.

## Source Status

Reliable public sources used here:

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-V3 GitHub repository: https://github.com/deepseek-ai/DeepSeek-V3
- DeepSeek-V2 Technical Report: https://arxiv.org/abs/2405.04434
- DeepSeek-V3.1 model card: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- DeepSeek-V3.2-Exp model card: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek-V4 Technical Report: https://arxiv.org/pdf/2606.19348
- TransMLA as a secondary MLA migration reference:
  https://arxiv.org/abs/2502.07864

Correction: the DeepSeek-V4 Technical Report is available on arXiv. Earlier
notes that no official V4 report was found are obsolete.

## What DeepSeek Actually Optimizes

DeepSeek-V3 is not just a larger Transformer. Its important pattern is:

```text
compress memory + activate sparsely + route carefully + train stably +
co-design with hardware
```

Concrete examples:

- **MLA:** compresses attention KV cache into latent vectors.
- **DeepSeekMoE:** uses sparse expert activation so total parameters can grow
  while per-token compute stays much smaller.
- **Auxiliary-loss-free load balancing:** adjusts routing bias directly, instead
  of relying only on an auxiliary loss that can hurt model quality.
- **Node-limited routing:** limits the communication domain for each token.
- **No token dropping:** load balancing is engineered strongly enough that tokens
  are not discarded.
- **MTP:** predicts additional future tokens during training, improving training
  signal and potentially enabling speculative decoding.
- **FP8 / fine-grained quantization:** low precision works only because scaling,
  accumulation, and sensitive operators are carefully separated.
- **Deployment split:** prefill and decode have different bottlenecks, so they
  are scheduled differently.
- **DSA in V3.2-Exp:** sparse long-context attention is introduced as an
  efficiency experiment while aiming to preserve output quality.

DeepSeek-V4 continues the same systems pattern but changes the attention and
optimization stack more aggressively:

- **Compressed Sparse Attention (CSA):** uses Lightning Indexer to select a
  smaller set of key/value entries before attention.
- **Hybrid Compression Attention (HCA):** keeps dense causal context through a
  shared-head, recurrently updated compressed KV state.
- **mHead Compression (mHC):** expands GQA into HCA by grouping heads and
  combining local attention with compressed recurrent attention.
- **Muon optimizer:** targets better scaling and training efficiency than a
  conventional AdamW-only recipe.
- **FP4 QAT:** pushes training and inference toward very low precision with
  group-wise scale and offset metadata.
- **TileLang and fused kernels:** custom kernels are used for CSA/HCA, FP4, and
  MoE kernels rather than relying on generic operators.
- **Heterogeneous cache management:** KV/cache data is staged across GPU memory,
  CPU memory, and SSD, including prefix sharing and on-disk prompt cache.

## Translation To HARC-CA

### 1. MLA -> Latent State Compression

MLA says: do not store full attention keys and values if a compressed latent
state can preserve enough information.

HARC-CA translation:

- token cells should not retain full high-dimensional hidden states forever;
- each block should maintain compressed latent summaries;
- exact details should move into associative lanes only when they are likely to
  be needed;
- CA state width must be justified by retrieval and prediction quality, not by
  copying Transformer hidden size.

Design gate:

```text
State bytes per context token must be measured against KV-cache bytes per token.
```

### 2. MoE -> Sparse Active Cell Experts

DeepSeek grows total model capacity through sparse expert activation.

HARC-CA translation:

- each cell should not run every rule every tick;
- the fabric should contain specialized local rule banks;
- a small router selects a few rule banks per active region;
- inactive regions should sleep or perform cheap decay/maintenance ticks.

This suggests a **Cellular MoE**:

```text
cell state + phase + local features -> route to k local rule banks
```

Hardware implication:

- rule banks are local LUT/bit-sliced kernels;
- routing must be low-bit and bounded to a small physical neighborhood;
- active cell fraction becomes as important as parameter count.

The first HARC-CA Cellular-MoE prototype now tests this execution pattern. It
routes only active cells to one low-bit local rule bank per tick and uses a
DeepSeek-style bias control loop to reduce load imbalance without a loss term.
The current rollout shows about 30x fewer rule updates than dense all-rule
execution at 20% active cells.

### 3. Auxiliary-Loss-Free Balancing -> Bias-Controlled CA Routing

DeepSeek avoids letting load-balancing loss dominate modeling quality by
adjusting routing biases based on observed expert load.

HARC-CA translation:

- do not rely only on a loss term to make CA routing balanced;
- maintain per-lane or per-rule bias counters in hardware/software;
- increase bias for underused lanes and decrease it for overloaded lanes;
- keep the actual content score separate from the load-balancing bias.

For our hash-routed associative lane, this points to:

- redundant hot buckets;
- overflow tiers;
- adaptive route bias;
- learned routing only after the fixed hash baseline is understood.

### 4. Node-Limited Routing -> Wire-Limited CA Communication

DeepSeek explicitly limits how many nodes a token can route to.

HARC-CA translation:

- every query wave must have a hard route budget;
- every generated token must have a maximum active-cell budget;
- no fallback path may silently become a full-context scan;
- routing should expose a physical locality constraint from the beginning.

Design gate:

```text
visited cells per query <= O(log context + small constant)
```

If exact recall requires dense global communication, the CA-first chip thesis is
not working.

### 5. MTP -> Multi-Tick Prediction

MTP improves training by asking the model to predict beyond the next token.

HARC-CA translation:

- train the CA to predict token `t+1`, `t+2`, and local future state;
- use rollout consistency losses, not just final-token loss;
- train routing waves to prepare future retrieval before the next token is
  requested;
- use multi-tick auxiliary heads during training and discard or compress them at
  inference.

This is especially natural for CA because the model is already a dynamical
system evolving over ticks.

### 6. FP8 Lessons -> Low-Bit CA Is A Training Problem

DeepSeek's FP8 result does not mean "just use low precision." It means low
precision works when sensitive paths, scaling granularity, and accumulation are
engineered carefully.

HARC-CA translation:

- start with continuous training but design the integer path from day one;
- use per-tile or per-channel scale / threshold metadata;
- keep sensitive routing scores and stability counters at higher precision;
- verify integer-only rollout for long tick counts;
- measure quantization-induced recall loss, not just average loss.

For chips, CA has a strong advantage here: bit-sliced local state and LUT rules
are more natural than large FP GEMMs. But the training method must respect this.

### 7. Prefill vs Decode -> Different CA Schedules

DeepSeek separates prefill and decode because their bottlenecks differ.

HARC-CA translation:

- context ingestion can run broad wavefront summarization;
- autoregressive decode should run a narrow active region plus sparse retrieval;
- the same silicon can support both with different schedules;
- benchmark them separately.

Suggested schedules:

```text
prefill:  dense-ish bottom-up summary + memory insertion
decode:   sparse active tail + query waves + local relaxation
refresh:  background maintenance / rebalance / compression
```

### 8. DSA -> Learned Sparse Retrieval, Not Blind Attention

DeepSeek-V3.2-Exp introduces sparse attention to improve long-context
efficiency while preserving output quality.

HARC-CA translation:

- our associative lane should eventually become a trainable sparse indexer;
- fixed hashing is only a baseline;
- index reuse across layers/ticks may matter;
- retrieval quality must be measured against exact attention-like tasks.

This supports our current Phase 1 focus on copy, induction, and key-value recall.

### 9. V4 CSA/HCA -> Two Memory Paths

DeepSeek-V4 separates sparse exact retrieval from compressed dense context. CSA
uses an indexer to reduce the candidates for attention, while HCA preserves a
compressed recurrent representation of causal history.

HARC-CA translation:

- use the associative lane for exact sparse recall;
- use the recurrent CA field for compressed dense context;
- do not force the same cell state to store all rare facts exactly;
- do not force the exact memory lane to carry all fuzzy semantic context.

This is a direct match for the HARC-CA split:

```text
CA state field     -> compressed dense context / fuzzy integration
associative lane   -> exact sparse facts / copy / induction / code symbols
```

Design gate:

```text
Exact-memory misses and compressed-context loss must be measured separately.
```

### 10. V4 mHC -> Grouped Rule Channels

V4's mHC idea groups attention heads and introduces compressed recurrent
attention while preserving local attention behavior.

HARC-CA translation:

- divide cell channels into groups with different roles;
- keep some groups for short-range local dynamics;
- reserve some groups for compressed long-range state;
- let retrieval lanes write into selected channel groups rather than the whole
  cell state.

This suggests a grouped state layout:

```text
local channels | recurrent summary channels | routing channels | exact-memory IO
```

### 11. V4 Muon -> Optimizer Matters

DeepSeek-V4 reports a custom optimizer stack. This matters for HARC-CA because
recurrent CA training may be more sensitive than Transformer training.

HARC-CA translation:

- do not assume AdamW is enough;
- test Muon-style or orthogonalized updates for the shared CA rule;
- separate optimizer settings for rule banks, routing, and quantization
  metadata;
- measure rollout stability, not only training loss.

### 12. V4 FP4 QAT -> 4-Bit Is A First-Class Target

V4 pushes quantization further with FP4-aware training, group-wise scaling, and
offsets. This strongly supports our low-bit CA chip thesis, but also raises the
bar: low bit-width must be trained into the model, not bolted on after training.

HARC-CA translation:

- make 4-bit state and 4-bit message lanes a primary target;
- keep per-group scale/offset metadata for continuous-to-integer training;
- test whether exact retrieval needs higher tag precision than the state field;
- report integer-only rollout separately from floating training rollout.

### 13. V4 Cache System -> CA Memory Hierarchy

V4's inference system treats cache placement as part of the model system:
hot data stays close, colder data moves to CPU/SSD, and shared prefixes are
cached.

HARC-CA translation:

- active tail cells should live in fast local SRAM/registers;
- block summaries can live in slower on-chip SRAM;
- exact associative entries can spill to overflow tiers;
- repeated prompt prefixes can be stored as pre-relaxed CA states;
- prefill, decode, and background refresh need different schedules.

The first HARC-CA overflow experiment is the minimal version of this idea:
evicted primary associative entries spill into a smaller hash-routed overflow
lane. Queries touch overflow only after a primary miss or tag collision. This
recovers exact recall in the current 16k full-context trial with a small average
visited-cell increase.

The first compressed dense-context experiment is the minimal HCA-like analog:
a low-bit decayed sketch tracks topic/recency distribution with a few local
counter updates per token. It is intentionally separate from exact sparse recall.
This reinforces the V4 lesson that sparse and dense memory paths should be
optimized independently.

The first synthetic next-token experiment adds the output-interface lesson:
exact sparse recall can produce value tokens directly, but dense context still
needs candidate generation and ranking. This mirrors the broader DeepSeek lesson
that model architecture and inference system cannot be separated.

The first unified efficiency profile follows the same discipline: do not judge a
model component in isolation. Exact memory, dense context, sparse rule execution,
and output candidates must fit one decode-event budget. The current profile is
still a toy, but it gives HARC-CA the same kind of systems accounting mindset
that makes DeepSeek's architecture reports useful.

The first floorplan proxy extends that mindset to tile budgets. It asks whether
the current state and event traffic fit into local SRAM and local byte bandwidth
before adding learned rules. This is the CA-chip analog of treating cache,
communication, and kernels as part of the model design rather than deployment
afterthoughts.

The first output-head budget adds another systems lesson: solving attention/KV
traffic is not enough if the logits path becomes the dominant kernel. HARC-CA
needs exact-token bypass and candidate scoring so the output head stays within
the same local budget as the CA memory and rule fabric.

The first online candidate-cache experiment removes the hot-token oracle from
that path. A 512-entry low-bit set-associative cache now generates the dense
candidate shortlist from observed tokens with zero full-vocabulary scans. This
is the CA analog of making the indexer/cache part of the model system rather
than assuming the expensive scoring kernel receives a perfect shortlist for
free.

The follow-up admission gate reuses the compressed dense-context sketch to
reject low-evidence noise tokens before they write into the candidate cache.
This mirrors the CSA/HCA systems lesson more closely: a cheap recurrent/index
path narrows the work before the more expensive candidate scoring path runs.
The first learned version trains an 8-byte signed 4-bit LUT from future-repeat
labels and recovers the same behavior as the hand-set threshold gate. That is
the first concrete step from manual indexing policy toward learned low-bit CA
routing.

## Revised HARC-CA Design Principle

The CA chip should not be "a big CA that tries to think everywhere." It should be:

```text
compressed persistent state
+ sparse local rule execution
+ bounded trainable routing
+ exact associative memory for rare details
+ multi-tick prediction training
+ grouped local/recurrent/routing channels
+ cache-aware state hierarchy
+ online candidate cache plus exact-token bypass
+ hardware-native low-bit arithmetic
```

This is the CA analog of DeepSeek's efficient-LLM recipe.

## Immediate Changes To Our Research Plan

1. Add a Cellular-MoE rule-bank concept before building the trainable CA.
2. Add route-bias / overflow experiments to the associative lane.
3. Treat active-cell fraction as a first-class model metric.
4. Add multi-token / multi-tick prediction heads to the first trainable model.
5. Separate prefill and decode benchmarks.
6. Preserve a small set of higher-precision stability variables even in low-bit
   rollout.
7. Split CA memory into compressed dense context and exact sparse recall paths.
8. Add grouped channel layout before the first trainable HARC-CA model.
9. Add a state-cache hierarchy design, including prefix-state reuse.
10. Treat online candidate generation as a learned/cacheable hardware path, not
    an oracle shortlist.

## Strong Warning

DeepSeek's lesson is not that any exotic architecture wins. The lesson is that
every efficiency claim must be tied to a bottleneck and validated at scale.

For HARC-CA, the bottleneck claim is:

```text
Reduce global memory movement and global communication during inference while
preserving enough exact recall and language modeling quality.
```

If experiments do not prove that claim, the CA chip idea must pivot.
