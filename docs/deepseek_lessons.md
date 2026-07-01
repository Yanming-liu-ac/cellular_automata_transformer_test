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

## Why The V4 Report Matters

DeepSeek-V4 is the closest public external anchor for this project so far. It
does not prove that a CA-native LLM already exists, and it should not be copied
as a Transformer clone. What it does prove is more important for hardware: a
frontier-style LLM can be reorganized around compressed memory, sparse context
selection, recurrent compressed state, very low precision, and custom kernels.

For HARC-CA, the report turns the design question into a sharper one:

```text
Can CSA/HCA/FP4-style systems logic be made into local CA state, local routing,
and small learned LUT controllers instead of attention kernels?
```

That is why the current experiments focus on a CSA-like sparse block path, an
HCA-like compressed recurrent path, rare exact directories, and tiny route/probe
LUTs before attempting a full language model.

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

The first dynamic propagation diagnostic supports this translation. A scalar
4-bit residual-average rule is stable but too slow to move a source pulse across
the HARC graph within 128 ticks. A max-route wave reaches all tokens quickly,
but it saturates the route plane. The mHC-inspired grouped rule separates local
residual state, a fast route state, and a stability envelope. It reaches all
tokens in 16/20/24 ticks for 128/512/2048-token HARC contexts while holding
saturation near one third of low-bit entries. The lesson is practical: grouped
state is not just a modeling flourish; it is how a low-bit CA can get both fast
propagation and bounded rollout behavior.

The 1,000-tick unforced check adds the training lesson. A hand-written damping
constant is not enough: the `mhc_damped` variant simply erases state. The
ungated max-route rule is fast but saturates or collapses. The grouped rule is
the only current hand-coded scaffold that survives sparse-random, dense-random,
and structured-pulse starts without zero collapse or global saturation, but it
settles to a low-entropy attractor. This is exactly where V4's optimizer lesson
matters: the CA rule needs trained, constrained dynamics, not just a manually
tuned leak.

The content-retention diagnostic adds the memory-layout lesson. The mHC carrier
is a computation carrier, not a reliable content store: direct shared-carrier
content retention is only about 5% after 1,000 ticks. A persistent content lane
keeps content exact, while the route/local/envelope carrier can remain a
low-bit dynamic workspace. Periodic refresh from content into carrier is a crude
upper bound and quickly spends local writes. The DeepSeek-style translation is
to keep paths specialized: persistent content, fast route state, and stability
metadata should be separate paths with learned gates between them.

The first gate diagnostic makes that last clause concrete. A simple local
`mismatch_ge8` comparator already beats fixed refresh16 on write traffic and
carrier error, because it spends writes only where the carrier has drifted far
from persistent content. This is the CA analog of path-aware gating: do not
collapse all residual paths every tick; expose the persistent path to the active
carrier only when the local state says it is useful.

The first learned version of that gate is only an 8-byte LUT. It learns a
threshold-like two-state policy, matching `mismatch_ge8` rather than surpassing
it. That is still useful: the control format is small enough for hardware, and
the failure mode is now precise. Better gates need task-weighted labels and
additional local demand features, not a larger controller.

The demand-weighted version confirms this. Adding one route/query demand bit
turns the gate into a 16-byte table and shifts the objective from "reconstruct
the whole carrier" to "make demanded content exact now." The result is a large
gain on the relevant path: 96.6% demanded exactness at about 0.134 channel
writes/token/tick. This is the CA form of DeepSeek-style path specialization:
move data on the active path, not everywhere.

Using rare-directory query traces sharpens the lesson. When demand is produced
by actual retrieval queries, the learned 16-byte gate reaches essentially exact
demanded content with only about 0.03 channel writes/token/tick. The systems
lesson is the same as CSA/HCA: compression and routing are useful only when the
controller knows which path is active.

The dual-path synthetic LM exact-query trace pushes the same point further:
topic events leave the exact-content lane idle, while query events demand one
fact row. The learned gate reaches about 99.6% demanded exactness at roughly
0.0019 channel writes/token/tick. This is path specialization as a CA rule:
idle content stays persistent, active content is exposed locally.

The mixed exact+candidate trace adds the uncomfortable but useful constraint:
when topic events demand all 64 output candidate rows, the learned gate rises to
about 0.178 writes/token/tick for 95.0% demanded exactness. DeepSeek-style
sparsity has to be copied at the output side too. HARC-CA needs a candidate
selection rule that makes output demand sparse before content is exposed.

The candidate sparsity sweep suggests the right scale: 8-16 demanded candidate
rows keep learned content-gate writes around 0.029-0.049, while 64 demanded rows
nearly erases the write advantage. That matches the DeepSeek lesson at the
systems level: routing and compression must happen before expensive state is
materialized.

The phase/rank/mismatch gate refines the lesson. A smaller 9-byte exactness LUT
beats the larger generic route/envelope-aware learned gate on sparse candidate
demand because it uses the right control signal. For CA hardware, tiny tables
can work if their inputs match the active path.

The first reducer diagnostic shows the DeepSeek-style tradeoff in miniature:
top-16 candidate exposure keeps about 82.8% of top-64 topic quality at about a
quarter of the content-gate writes, while top-32 keeps 91.7% at about half the
writes. The missing piece is not the exact exposure rule; it is a more local
candidate-ranking fabric that avoids scoring all rows.

The group-summary reducer is closer to that fabric. It gets top-32 demand to
93.6% of top-64 quality while cutting score reads by 81.25%. This mirrors the
DeepSeek theme more directly: do not materialize every candidate path; summarize
locally, route a small subset, then spend exact work only there.

The update-cost estimate keeps the same conclusion honest. Maintaining exact
16-row summaries costs about 234 score-equivalent cells/topic, but the
hierarchical top-16 path still cuts total score work by about 76%. Compression
has to include the update path, not only the read path.

Lazy refresh pushes the same point further: refreshing dirty 16-row summaries
every 16 topic steps still keeps top-16 at about 84% of top-64 quality while
cutting score work by about 82%. The useful CA rule is not "always exact"; it is
"exact only where stale state would change the route."

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

The CSA/HCA-aware profile tightens that lesson. The lazy HCA summary barely
changes event traffic, while CSA block-summary geometry controls whether a small
tile fabric fits. The earlier 512KB wide64 index pushed the 32-tile proxy over
its SRAM budget. The compact128 point brings current state down to about
451.8KB. The rare128 point goes further by replacing half of the block-summary
state with a small exact rare-token directory, bringing current state to about
354.6KB and 32-tile utilization to about 69.3%. This is the CA-chip version of
cache hierarchy pressure: sparse attention-style indexing saves reads, but the
index itself must be tiered, compressed, and co-designed with the routing
policy and exact-memory lane.

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
The learned candidate-scorer LUT did not beat dense-min scoring. A future-window
teacher is better aligned with sparse-attention indexing and slightly improves
the isolated topic stream, but still fails in the mixed synthetic LM. This is a
useful DeepSeek-style lesson: efficient indexing and scoring need the right
state interface, not merely a smaller kernel or a better label.
The first source-phase scoring sketch confirms the same lesson from another
angle: separating topic-output state from query/fact traffic helps noisy
candidate paths, but the current admission gate still dominates the best
synthetic result. CSA-like indexing needs a coordinated state interface, not
one isolated feature.
Combining topic-phase score with candidate-cache score through a fixed
`2 * topic_score + cache_score` rule improves the noisy online path, which is
the first positive sign for a multi-feature local indexer. It still does not
beat the gated dense baseline, so the DeepSeek-style takeaway remains
systems-level: compression, cache state, routing gate, and scorer state must be
trained and evaluated together.
The first learned linear and additive-LUT versions are small enough for hardware,
even after adding a 4-bit age feature, but they only match the hand-written
direction and do not beat it. That mirrors the DeepSeek lesson again: the win
comes from the full training/inference system, not from replacing a formula with
any learned low-bit rule.
The feature-collision diagnostic makes that concrete: before adding more LUT
capacity, the system must know whether its local state actually separates the
right candidates. Age helps but does not close the online gap. This is the
CA-chip analog of measuring sparse-index recall before optimizing the attention
kernel.
The full tuple LUT diagnostic reinforces the point: a large sparse table is not
a system design. Like sparse attention, the indexer needs sharing and a good
teacher signal, not just a bigger lookup.

The first compressed block-index experiment is a cleaner CSA analog than the
candidate scorer. It makes context blocks into CA cells, stores a 4-bit
compressed summary per block, and routes a query to only a few blocks. At
65k context, the `summary_width=256`, 8-block setting reaches 100% relevant
block-hit rate in the deterministic topic/noise trial while reducing token reads
from 65,536 to about 640 positions. The caution is equally important:
occurrence coverage is only about 8.4%, so CSA-like block routing is a front-end
for later scoring and exact memory, not a complete substitute for attention.
The repeated-read sweep sharpens the HCA side of the lesson: the compressed
block score is already within about 0.3 percentage points of exact block ranking
across the measured read budgets. More indexer cleverness is not the immediate
answer. High-frequency evidence should move through an HCA-like recurrent
summary, while extra CSA-style block reads should be spent selectively on
queries that need exact local detail.
The first hand-written CSA/HCA policy confirms this split in a toy setting. A
4KB global low-bit summary decides whether to skip block scoring. At threshold
8, the policy sends hot queries to HCA and cold relevant queries to CSA in the
current deterministic trial, reducing average block-score reads to about
300B/query. The next DeepSeek-style step is to learn that policy and verify the
compressed HCA path with task loss, not just with routing labels.
The block-state sweep adds the hardware half of the same lesson. A 64-token
block index with `summary_width=256` is reliable but costs 512KB of SRAM. Moving
to 128-token blocks with the same width halves CSA state to 256KB and halves
average block-score reads, while preserving measured CSA-path hit and coverage
in the deterministic stream. The cost shifts into larger selected token blocks,
about 331 positions/query instead of 165. That is the right kind of trade:
SRAM, bandwidth, and route quality are co-designed rather than optimized in
isolation.
The rare-directory sweep sharpens the split. A lower-width 128-token CSA summary
uses only 128KB but misses too many cold blocks by itself. Adding about 30.8KB
of exact rare-token block ids restores measured routed-CSA hit and coverage to
100% on the reference stream. The stress sweep adds the missing systems lesson:
the HCA gate must not falsely swallow rare tokens, and directory fanout must
grow when rare names are spread across many blocks. This mirrors the CSA/HCA
systems lesson more closely than a single larger index: dense recurrent state,
sparse compressed routing, exact rare-detail state, and learned admission/fanout
policy should be separate cooperating structures.
The directory-guard experiment makes that cooperation active rather than
passive. The exact rare directory can override HCA admission when a token is
known to be rare and exact-sensitive. That costs a tiny directory probe, but it
turns the exact lane into a control signal for the memory hierarchy, not just a
fallback store.
The fanout policy sweep adds the next systems detail: storing enough rare block
ids is not the same as reading them every time. Metadata should control fanout
so compact rare tokens stay cheap and spread-out repeated names spend more
local reads. This is the same style of design discipline as CSA/HCA/FP4 systems
work: the metadata is part of the model architecture.
The first span-class fanout LUT makes that concrete. With guarded threshold-8
routing, a 2->4 expansion reaches about 93.0% repeated-name coverage at
13.0B/query, 2->5 reaches about 98.4% at 16.25B/query, and 2->6 reaches full
coverage at 19.5B/query. The lesson for the CA chip is that exact sparse recall
needs a learned control plane, but that control plane can be a few low-bit
metadata classes rather than a dense attention module.
The trained fanout LUT strengthens that point: a 42B table using entry-count,
span, and CSA-overlap metadata reaches about 98.4% repeated-name coverage at
12.87B/query, matching the hand 2->5 coverage with less directory traffic. This
is close to the CSA/HCA design style we want: large behavior changes come from
small learned routing tables attached to structured memory, not from making
every token attend to every prior token.
The joint probe sweep adds another useful systems lesson: HCA can expose its own
confidence through low-bit bank statistics. A 40B probe LUT uses estimate,
bank-spread, and saturation count to skip rare-directory probes for strong hot
tokens, cutting reference directory traffic from 3.25B/query to 0.50B/query
while retaining about 97.7% repeated-name coverage. The CA chip should therefore
treat HCA uncertainty as a routing signal for exact sparse memory.
When this control plane is folded into the unified event profile, the current
joint128 budget still uses about 52.28KB/event and about 356.9KB of on-chip
state. That is the right direction for a CA-first chip: learned routing improves
recall behavior without turning into a large dense state or a global read path.
The threshold sweep adds the matching policy lesson: once exact-memory probe and
fanout are learned, the HCA threshold should be chosen jointly with them. In the
current stress set, threshold 15 is cheaper than threshold 8 because it removes
early probes while preserving the same rare coverage under the learned fanout
path.
The first route-LUT sweep is the useful caution: replacing the explicit
threshold with a 40B table is possible, but not automatically better. It
preserves reference HCA routing yet loses a little repeated-name coverage. For a
CA chip, trainable control tables need recall-weighted objectives, not just
smaller control state.

The directory-aware route sweep is the constructive version of that lesson.
Adding one rare-directory presence bit gives an 80B route table that preserves
reference HCA routing, removes rare false-HCA in the stress set, and slightly
beats the threshold-15 repeated-name coverage. This mirrors the useful V4
pattern: the win is not an exotic rule alone, but the right small metadata made
visible at the routing point.
The sidecar false-positive sweep adds the matching hardware discipline: low-bit
metadata needs an error budget. In the current stress set, 1-10% presence
false positives mostly cost hot-path HCA efficiency, while 25% is too loose and
collapses reference HCA routing. CA routing metadata should therefore be trained
and budgeted as part of the model, not treated as free bookkeeping.
The physical Bloom-sidecar sweep is the next systems step: hash count, bits per
entry, read bits, update bits, and bank conflicts become model-facing design
parameters. This is exactly the DeepSeek-style lesson at CA scale: efficient
models come from the joint design of routing policy, memory format, and kernel
or fabric access pattern.
The hash-salt robustness sweep makes that even more concrete: the same sidecar
geometry can be good or mediocre depending on which hot tokens collide with the
rare-directory Bloom state. A CA chip needs compiler-selected or trained
metadata layouts, not fixed hashes chosen outside the model loop.
The bank-mapping sweep separates a layout problem from a model problem:
`by_hash` banking removes same-query sidecar bank conflicts without changing the
Bloom contents or routing labels. This is the CA version of a fused-kernel
lesson: the data layout is part of the model system.
The salt-selection sweep completes the same point from the compiler side:
selecting the hash salt against hot-token false positives recovers most of the
ideal HCA path without increasing state. For CA hardware, routing metadata is
not static bookkeeping; it is part of the learned or compiled model artifact.
The streaming-update sweep adds the time dimension to that lesson. A static
Bloom sidecar can be excellent after salt selection, but a naive count-threshold
online insert rule writes future hot tokens into the rare-token sidecar and
collapses the HCA fast path. The DeepSeek-style takeaway is that routing state
updates, retirement, and load protection belong in the architecture, not in a
post-hoc runtime patch.
The counting-retirement sweep is the constructive follow-up. It restores the
HCA fast path by making sidecar metadata deletable when a token becomes hot,
while preserving a 1-bit query plane for fast routing. The cost, about 44-45KB
of local state for this context geometry, is the CA equivalent of paying for
expert-routing or FP4 scale metadata: it is worthwhile only if the compiler or
training loop can account for it and then compress it.
The retire128 event profile closes that accounting loop. Adding the conservative
online sidecar raises state from about 356.9KB to about 401.8KB but adds less
than 1B/event of local traffic. This is a useful CA-chip result because it
separates the bottleneck: the next problem is sidecar SRAM compression, not
sidecar bandwidth.
The compression and adversarial-collision sweeps then find the next
DeepSeek-like step: metadata should be as narrow as the measured routing task
allows, but no narrower. Two-bit sidecar counters look sufficient on normal
streams, but targeted hot-token deletions reduce visible rare-token rate to
about 97.7%-98.4% with one collider and 62.5% with eight colliders per rare
token in the repeated-key collision stress. Three-bit counters restore the
robust 100.0% visible rare-token point through that multi-collider stress and
keep the total event-profile state about 392.8KB. One-bit counters are tempting,
but the adversarial sweep effectively rejects them for exact sidecar use. The
remaining repeated-key coverage gap at c3 is a fanout/directory objective, not a
counter-width objective. The next fanout-budget sweep is the hardware version of
the same lesson: raising the abstract coverage target from 95% to 100% does not
move the repeated-key corner, but raising the minimum directory read guard from
two entries to three restores 100.0% coverage. The cost is local and explicit:
directory traffic rises from 6.88B/query to 10.12B/query, and token-read
reduction moves from 78.2x to 76.6x. The next refinement is even more
DeepSeek-like: price the exact failure mode, not the whole path. A
zero-overlap guard restores the same repeated-key 100.0% coverage at
7.33B/query, because it triggers only when CSA selected none of the exact
rare-directory entries. Under the normal threshold-15 fanout profile it keeps
reference, rare-burst, and repeated-name traffic unchanged; split-rare moves
only from 6.50B/query to 6.53B/query while coverage reaches 100.0%. This is the
right hardware pattern: make the robust case explicit, then verify that the hot
reference path does not pay for it.

The delayed-promotion diagnostic adds the matching negative lesson. A pure
count gate is too blunt: `count2_retire15` and `count3_retire15` save update
traffic, but they drop visible rare-token sidecar coverage to single digits on
normal stress because one-hit facts never get admitted. That is exactly the kind
of low-bit optimization DeepSeek-style engineering should reject. The next gate
needs another local signal, not just a higher threshold.

The probation-promotion follow-up gives the constructive version. A persistent
first-hit presence plane repairs rare visibility but pollutes the hot path, so
it fails the same "common path must stay cheap" test. A deletable first-hit
probation plane is better: at 8 probation bits/entry it keeps about 99% rare
visibility and lowers update traffic from about 0.22B/token to about
0.14B/token, but it spends roughly 53KB of sidecar state and 0.75B/query of
sidecar read bandwidth. At 4 bits/entry it is smaller, about 45KB, but takes
more false-positive risk. The DeepSeek-style lesson is that promotion must be a
joint control problem: admission, deletion, local probe feedback, and sidecar
bit-width have to be optimized together.

The first HCA-summary quality check is the cautionary half of the lesson. The
4KB 4-bit global summary is good enough for the current threshold gate, but not
for fine ranking of the hottest topic tokens. Even an 8KB version has only about
51.6% top-64 frequency recall because the hottest counters saturate. This is
exactly why V4-style low-bit design needs scales, offsets, grouped state, and
sensitive-path exceptions rather than simply using 4-bit counters everywhere.
The decay sweep shows the constructive side: low-bit state can work when the
state dynamics keep it out of saturation. A 4KB decayed global summary recovers
100% top-64/top-256 decayed-topic recall in the current stream. The hardware
lesson is that decay or scale metadata is part of the model, not an
implementation detail; otherwise maintenance traffic can erase the gain.
Lazy epoch decay makes that lesson concrete. It preserves the decayed HCA target
without global sweeps by storing per-counter epoch metadata. The cost moves from
periodic maintenance traffic into local SRAM and slightly wider reads. This is
the CA-chip analog of treating FP4 scales and offsets as first-class state
rather than post-hoc quantization.
The metadata-width sweep improves the design point: 8-bit epochs keep the same
measured quality as 16-bit epochs in the current 65k-token window while reducing
state from about 20KB to 12KB. Four-bit epochs are smaller but start losing
dense-topic quality because they require slower decay. This is the practical
shape of low-bit design: counters, epochs, scales, and update schedule must be
co-optimized.

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
