# Research Notes

## Evidence We Can Build On

- Neural cellular automata are learnable local dynamical systems, and recent
  work surveys NCA notation and reference implementations.
- Universal NCA work has shown trained CA-style systems performing matrix
  multiplication, transposition, and neural-network emulation inside the CA
  state.
- NCA-generated data has recently been used as synthetic pre-pre-training data
  for language models, with reported improvements in language modeling and
  convergence. That does not make the NCA itself an LLM, but it shows CA
  dynamics can generate statistics that interact usefully with language-model
  training.
- TPU history supports the broader hardware lesson: large gains can come from
  domain-specific execution and software-managed local memory.
- DNN accelerator surveys consistently identify data movement and memory
  hierarchy as central energy/performance concerns.

## Working Hypothesis

The winning CA language architecture is unlikely to be a single elementary CA
rule. It is more likely to be a learned, quantized, multiscale CA with:

- persistent context state;
- local recurrent update;
- hierarchical summaries;
- associative sparse retrieval;
- bit-sliced execution;
- a small output interface that avoids full-vocabulary dense projection where
  possible.

DeepSeek-V3/V4 sharpen this hypothesis. Their system-level pattern is
compressed memory, sparse activation, careful routing, low precision with
selected high-precision paths, and hardware-aware scheduling. HARC-CA should
mirror that pattern in CA-native form: compressed cell summaries, sparse active
rule banks, bounded route waves, associative exact memory, and multi-tick
prediction training.

DeepSeek-V4 adds an especially relevant lesson: efficient inference can combine
two memory paths. Its CSA path compresses the retrieval set before attention,
while HCA preserves a dense causal view through compressed recurrent state. The
CA analog is to pair a sparse exact associative lane with a compressed recurrent
state field, instead of forcing one mechanism to solve both exact recall and
fuzzy context integration.

## What Must Be Proven

The project should prove or disprove these claims experimentally:

1. **Fast propagation:** useful long-range information can travel in `O(log N)`
   ticks over a physically local multiscale layout.
2. **Stable recurrence:** the same rule can run for long rollouts without
   exploding, collapsing, or accumulating unrecoverable quantization error.
3. **Language capability:** the architecture can learn next-token prediction,
   induction, copy, bracket matching, arithmetic, and retrieval tasks.
4. **Low-bit viability:** 1-bit to 4-bit state retains enough capacity after
   quantization-aware training.
5. **Hardware advantage:** measured bytes moved per generated token and local
   update counts plausibly beat a Transformer baseline at useful quality.

## Important Risks

- The hierarchy may compress away exact facts needed for language tasks.
- Associative routing may become too expensive or too hard to train.
- CA rollout depth may erase the energy advantage if too many cells update per
  generated token.
- A byte-level output space is hardware-friendly but may require more generated
  steps than subword tokenization.
- Matching Transformer quality may require a hybrid architecture, at least
  during early versions.

## Could This Be Worse Than A Transformer?

Yes. A naive CA language model will almost certainly be worse than a Transformer.
The failure modes are clear:

- **Quality gap:** global attention is an extremely strong primitive for exact
  token-token interaction. Local CA rules must learn routing, compression, and
  retrieval before they can compete.
- **Training gap:** Transformer optimization is mature. Recurrent CA training can
  be unstable, slow, or sensitive to rollout length.
- **Latency gap:** if a token needs hundreds of CA ticks, local low-bit hardware
  may still lose to dense matrix hardware.
- **Memory gap:** if the CA needs too many active cells, the state field becomes
  just another large cache.
- **Output gap:** a full vocabulary projection can dominate energy if not
  redesigned.

Therefore the project should not claim "CA beats Transformer" until measured.
The correct bet is narrower:

```text
At a chosen quality target, a CA-first architecture may reduce global data
movement and off-chip bandwidth enough to win in tokens-per-watt or
latency-per-watt for specific model sizes and deployment regimes.
```

If that narrower claim fails, the architecture should be changed or abandoned.
The most likely practical path is hybrid at first: CA fabric for local memory,
state evolution, compression, and retrieval; small attention or expert modules
only where exact global binding is truly needed.

## Early Decision Gates

The next prototypes should be judged by gates instead of intuition:

1. HARC-CA propagation must stay near `O(log N)` in context length.
2. Integer or quantized rollout must stay stable for at least 1,000 ticks.
3. CA retrieval must solve copy/induction tasks without scanning every token.
4. At equal toy-task accuracy, CA proxy memory movement must be lower than a
   tiny Transformer baseline.
5. If quality requires dense global communication every token, the CA-first chip
   hypothesis is not working.

## First Retrieval Prototype

The first non-neural retrieval component is a multi-route hash-routed
associative CA lane. It is closer to a hardware primitive than to a trained
model:

- a query routes through logarithmic local trees using hash bits;
- it lands in one or more set-associative buckets;
- a small number of low-bit tags are compared in parallel;
- the value returns without scanning all sequence cells.

This is deliberately not exact Transformer attention. It tests whether a CA
fabric can provide sparse exact recall for copy and induction tasks.

The first single-route sweep showed:

| Context | Buckets | Ways | Correct Recall | Cells Visited | Full Scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,024 | 4 | 99.3% | 14 | 1,024 |
| 4,096 | 4,096 | 4 | 99.2% | 16 | 4,096 |
| 16,384 | 16,384 | 4 | 99.6% | 18 | 16,384 |

The warning sign is capacity pressure: at load factor 1.0, the same 4-way design
falls to roughly 80% recall due to bucket evictions. The chip architecture
therefore needs either more ways, better hashing, learned routing, overflow
lanes, or tiered memory for rare exact facts.

The second sweep added multi-route lookup and explicit copy / induction /
key-value tasks. At fixed capacity (`buckets = context / 4`, `ways = 4`) and
16k context, 2-route lookup reached roughly 92-93% recall while visiting about
32 cells per query. That is still far below a full scan of 16,384 token cells,
but it is not yet reliable enough for LLM-grade exact memory.

The third sweep added a hash-routed overflow tier, following the DeepSeek-V4
cache-hierarchy lesson. At 16k context, the primary lane used
`buckets=context/4`, `ways=4`, `routes=2`; the overflow lane used
`buckets=context/16`, `ways=4`, `routes=2`; tags were widened to 32 bits. On the
deterministic full-context copy, induction, and key-value trials, this recovered
100% exact recall while average query work stayed around 34 visited cells. The
single-lane baseline stayed around 92-93% recall at about 32 visited cells.
Only about 8% of tiered queries touched overflow, so the overflow tier behaved
like a cache hierarchy rather than a scan fallback.

This suggests a promising memory hierarchy:

```text
primary associative lane   -> common hot facts
overflow associative lane  -> bucket-pressure victims
compressed CA field        -> fuzzy dense context
```

The fourth sweep added the compressed dense context path. A 4-bit decayed
count-sketch with `banks=4` and `width=2048` used 4KB of state for a
65k-vocabulary topic stream and recovered the exact top-64 decayed topic tokens
in the current deterministic 65k-context trial. The exact 4-bit dense counter
table would use 32KB. This supports the dense-path role as a compact context
distribution tracker, not as exact memory.

The current dual-path demo combines:

- tiered associative memory: 162.5KB, 100% induction recall on the deterministic
  16k full-context trial;
- compressed dense sketch: 4KB, 100% top-64 topic recall on the deterministic
  65k-vocabulary trial.

This is still a memory-system prototype, not an LLM. The next hard step is to
connect these paths to a trainable recurrent CA rule and a prediction head.

The fifth sweep added a non-trained synthetic next-token benchmark. It combines
topic-like events and induction key-query events in one stream. The exact sparse
lane predicts key values; the compressed dense sketch ranks a candidate pool for
topic tokens. In the current deterministic run, induction accuracy is 100%,
topic top-k hit rate is about 62%, overflow is touched by about 6.7% of exact
queries, and the mixed stream touches about 27 local cells per event.

This is a useful bridge because it exposes the output-interface problem:

- exact facts can bypass a dense output head;
- fuzzy topic tokens still need candidate generation/ranking;
- a future trainable CA must learn when to use each path.

The sixth sweep added a Cellular-MoE execution prototype. It addresses a
different bottleneck: if every cell runs every rule every tick, CA loses its chip
advantage. The prototype routes only active cells to one of six local low-bit
rule banks and adjusts routing bias from observed load. In the current rollout,
20% active cells and top-1 routing reduce rule-update count by about 30x versus
dense all-rule execution. Bias control reduces load imbalance but does not fully
solve it, which means learned routing or stronger hardware scheduling will still
be needed.

The seventh sweep added a unified event-level efficiency proxy. It combines the
synthetic dual-path next-token benchmark with Cellular-MoE execution and compares
the resulting local byte movement with a tiny Transformer KV-cache read volume.
With 4 Cellular-MoE ticks per event and gated online candidate generation, the
legacy HARC-CA profile moves about 51.46KB of local on-chip bytes per event and
keeps about 183.8KB of on-chip state. The tiny Transformer KV reference reads
about 384MB per token at 16k context.

This ratio is intentionally not treated as a win. The HARC-CA prototype is not a
quality-equivalent model, and local SRAM/register traffic is not the same as
HBM/cache traffic. The useful conclusion is narrower: the current architecture
has a measurable path to keeping its toy next-token behavior inside local
low-bit traffic, which is the right bottleneck direction for a CA-first chip.

The eighth sweep added a tile-level floorplan proxy. The legacy event profile
is mapped onto repeated tiles with 64 cells, 16KB local SRAM, and 32 local
bytes/cycle. Under these assumptions, a 32-tile fabric has 512KB local SRAM,
stores the current 183.8KB state at about 35.9% utilization, and reaches about
5.1% aggregate local bandwidth utilization at 1M synthetic events/s. This gives
the project a first SRAM/bandwidth budget for future learned rules.

The warning remains important: this is not area, power, timing, or model quality.
It is a bookkeeping tool to prevent future experiments from silently consuming
the locality advantage.

The ninth sweep added an output-head budget. This is a critical risk: the
current HARC-CA event profile is about 51KB/event before output scoring, while a
65k full-vocabulary output head with 128 hidden channels costs about
4.13MB/event and 8.39M MACs/event. A 512-token candidate head with exact-query
bypass costs about 22KB/event. This makes candidate generation and exact bypass
mandatory parts of the architecture, not optional optimizations.

The tenth sweep removed the static hot-token oracle from candidate generation.
The new online candidate cache is set-associative, low-bit, and decayed. With
512 entries, 4-bit scores, 2 routes, and 4 ways, it uses about 1.31KB of state
and performs zero full-vocabulary scans. On the standalone topic/noise stream it
reaches about 69% top-64 hit rate after warmup. Plugged into the synthetic LM,
it gives about 61.4% topic@64 versus about 62.1% for the static candidate pool,
with about 6.6 extra candidate-cache cell touches per mixed event.

The same sweep then added a threshold-1 admission gate that reuses the dense
context sketch. This gate prevents low-evidence noise tokens from writing into
the candidate cache. In the standalone topic/noise stream, top-64 hit rate rises
to about 70.8% and replacements drop to zero for a 512-entry cache. In the mixed
synthetic LM, gated online candidates reach about 67.1% topic@64, admit about
60.5% of topic observations, and raise cache-update hit rate to about 98%. The
cache-write cost falls to about 4.0 cells/event, with about 2.7 dense gate
reads/event.

The next step replaced the hand-set threshold with a learned low-bit LUT. The
training label is self-supervised: whether the token repeats within a future
horizon of 256 events. The deployed LUT only reads the dense-sketch estimate and
does not inspect token identity or future labels. In the current deterministic
trial it learns `(-8, 7, ..., 7)`, an 8-byte policy equivalent to threshold-1.
It reaches about 70.8% standalone top-64 hit rate and about 67.1% synthetic-LM
topic@64, with zero full-vocabulary scans.

This is a small but important transition: candidate admission is now a trained
low-bit rule in the prototype, not only a hand-selected constant. It still does
not prove real language-model routing quality.

This is an important correction to the research accounting: candidate
generation is no longer assumed to be free. The first learned version is only a
tiny LUT and is not sufficient for real LLM quality.

The eleventh sweep tested learned candidate scoring. The scorer is a 16x16
signed 4-bit LUT over dense estimate and candidate-cache score, trained from the
same future-repeat signal. It uses 128 bytes and matches dense-min scoring on
the standalone topic stream, but it does not generalize to the mixed synthetic
LM: topic@64 drops from about 67.1% with dense-min scoring to about 64.6% with
the learned LUT. This is a useful negative result. Admission can be learned with
the current feature, but candidate scoring needs a richer state.

The twelfth sweep changed the scorer objective rather than the hardware shape.
It trains the same 128-byte LUT from a future-window teacher and applies it as a
local residual on top of the dense score. This is closer to a CSA-style indexer:
the rule is asked which resident candidates will matter soon, not only whether
they match the current token. It improves the standalone topic stream from about
67.7% to about 68.2%, but it still fails in the mixed synthetic LM, reaching
only about 64.5% topic@64. The likely cause is feature insufficiency: query/fact
traffic contaminates the dense sketch, and the 2D LUT has no source, phase,
recency, or multi-tick stability feature to distinguish useful context from
noise.

The thirteenth sweep added the first explicit source/phase feature. Candidate
ranking can now read a separate topic-phase dense sketch that is updated only
after topic-output events. This isolates output scoring from exact-memory query
and fact updates. The effect is conditional rather than universally positive:
static topic@64 improves from about 62.1% to about 66.7%, and online always-admit
topic@64 improves from about 61.4% to about 64.4%. Once the admission gate is
enabled, the same source-phase sketch is redundant: gated dense scoring reaches
about 67.1%, while gated topic-phase scoring reaches about 67.0% and adds about
4KB state plus about 2.7 score-sketch writes per mixed event.

The fourteenth sweep combined local scorer signals. `topic_cache` uses
`2 * topic_score + cache_score` and raises online always-admit topic@64 to about
65.8% without increasing score-read cells beyond the topic-phase path.
`dense_topic_sum` combines both dense sketches and reaches about 67.0% on the
static candidate pool, but doubles candidate score reads. The admission-gated
path remains the strongest current setting: gated dense scoring is about 67.1%,
gated topic-phase is about 67.0%, and gated topic-cache is about 66.7%. This
suggests source/phase/cache signals should feed a learned local indexer, but
they are not yet a better hand-written replacement for the current gate.

The fifteenth sweep trained the first multi-feature local indexers. The first
rule is a signed 4-bit linear scorer over `(dense, topic, cache, contamination,
age)` and has only 3.0 bytes of parameter state including bias. A top-k
perceptron-style training pass learns `(3, 7, 7, 2, -4)` for online always-admit
and `(-1, 7, 6, 2, -5)` for the gated path. The second rule is a factorized
additive LUT with five 16-bin feature tables and 40.5 bytes of state. Both are
small enough for local hardware, but they still do not beat the hand-written
topic-cache rule: online topic@64 is about 63.4% for linear and 64.7% for
additive, versus about 65.8% for topic-cache. Gated topic@64 is about 66.7% for
linear and 66.6% for additive, versus about 67.1% for gated dense scoring. This
is a useful boundary result. The feature interface is reasonable, but the
current learners are too weak, too factorized, or trained against the wrong
objective.

The sixteenth sweep added a feature-collision ceiling. This asks whether a
perfect scorer over the existing low-bit feature tuple could separate the true
resident token from other residents. Adding 4-bit resident age helps but does
not close the online gap: in online always-admit mode, resident recall is about
79.0%, but the optimistic feature ceiling is only about 70.9%; the true token
shares its exact feature bucket with about 47.7 candidates on average. In gated
mode, resident recall is about 69.2% and the feature ceiling is also about
69.2%, with an average positive bucket of about 3.6 candidates. This separates
two problems: online mode needs additional local state to reduce feature
collisions, while gated mode mostly needs a stronger ranking/training objective.

The seventeenth sweep tested a full 5D tuple LUT over the same age-augmented
feature tuple. This is intentionally the high-capacity extreme: a dense 4-bit
table over five 16-bin features would use about 512KB. It is not useful yet.
The training stream observes only 893 online tuples and 2878 gated tuples, so
the table is mostly empty. Online tensor scoring collapses to about 39.0%
topic@64 with log-odds and about 35.0% with rate scoring. Gated tensor rate
scoring reaches about 66.4%, close to topic-cache but still below gated dense
scoring. This says the next indexer should not simply add dense tensor capacity;
it needs better sharing or distillation.

The eighteenth sweep shifted from output-candidate ranking to CSA-shaped
context-block routing. The new compressed block index splits a 65,536-token
context into 1024 blocks of 64 tokens. Each block stores a 4-bit count-min
summary; a query token is scored locally by every block, then only the top
blocks plus a short exact tail are read. With `summary_width=256`, 4 banks,
8 selected blocks, and 2 tail blocks, the block summaries use about 512KB and
the query scoring pass reads about 2KB of 4-bit summary counters. The selected
path reads about 640 token positions instead of 65,536, about a 102x token-read
reduction.

This is the first direct CA analog of DeepSeek-V4 CSA in the repo. It is a
positive routing result: relevant-query rate is about 87.2%, overall block-hit
rate is 100%, and the measured cold-token relevant subset also reaches 100% in
the deterministic trial at `summary_width=256`. It is not a full attention
replacement. Occurrence coverage is only about 8.4%, close to the oracle
top-block coverage for the same block budget, so selected blocks must feed
within-block scoring, exact associative recall, or repeated sparse reads.

The nineteenth sweep measured the repeated-read budget curve for that same
512KB block index. The result is useful because it separates two possible
failure modes. With 4, 8, 16, 32, 64, and 128 selected blocks plus a 2-block
tail, occurrence coverage rises from about 5.6% to 46.1%, while token-read
reduction falls from about 170.7x to 7.9x. The gap to an exact top-block oracle
stays tiny, about 0.04 to 0.26 percentage points. The current compressed
block-index rule is therefore already close to the exact block ranking on this
task; the hard problem is deciding when to spend more sparse reads versus when
to trust a compressed dense summary.

The twentieth sweep added the first explicit CSA/HCA read-policy diagnostic.
A 4KB global 4-bit summary estimates query frequency before the block index is
scored. Frequent queries are delegated to the HCA-like dense/recurrent path;
low-frequency queries use 4 CSA blocks plus a 2-block tail. At threshold 8, the
policy routes about 85.4% of all queries to HCA and 14.7% to CSA. In the current
deterministic trial it sends 100% of measured hot relevant queries to HCA and
100% of measured cold relevant queries to CSA. CSA-routed relevant queries have
100% block hit and 100% occurrence coverage because they are mostly rare tokens.
Average block-score traffic drops to about 300B/query and average token block
reads to about 165/query, about a 396x full-context token-read reduction. This
is a policy result, not a language result: it assumes the HCA path can summarize
high-frequency distributed evidence well enough.

The twenty-first sweep tested that assumption directly. The global HCA-like
summary is a low-bit count-min sketch over the whole context. At threshold 8,
1KB and 2KB summaries are too collision-heavy for routing: query route accuracy
is about 85.4% and 94.6%, with many false HCA routes. A 4KB summary reaches
100% route accuracy on the deterministic query stream and a 100% threshold
recall, so it is enough for the hand CSA/HCA policy. But dense-state quality is
not solved. The 4KB summary has only about 42.2% top-64 frequency recall, and
the 8KB summary still reaches only about 51.6% top-64 recall despite 100%
top-256 recall. This points to 4-bit saturation among hot tokens. The next HCA
experiment should add decay, scaling, or grouped higher-precision metadata
rather than only increasing width.

The twenty-second sweep tested the simplest anti-saturation mechanism: periodic
integer decay. Keeping the HCA-like global summary at 4KB (`width=2048`) and
using a decayed-state threshold of 2, decay intervals from 64 to 512 tokens
recover 100% top-64 and top-256 decayed-topic recall with 100% route accuracy
on the deterministic query stream. A 1024-token decay interval still reaches
about 98.4% top-64 and 99.6% top-256 recall. The no-decay baseline at the same
threshold falls back to about 42.2% top-64 and 88.2% route accuracy because
saturation produces false HCA routes. The tradeoff is maintenance traffic:
decay every 256 tokens costs about 32 decay-cell touches per token if counted
synchronously. This should become a scheduled/background tile operation or a
learned scale/threshold mechanism.

The twenty-third sweep replaced synchronous decay with lazy epoch metadata. Each
global-summary counter stores a small epoch and is shifted only when read or
updated. With `width=2048`, 4-bit counters, 16-bit epochs, decay interval 256,
and threshold 2, the lazy summary matches the decayed target in the current
trial: 100% top-64/top-256 recall, 100% route accuracy, and no false HCA routes.
It removes the 32 decay-cell touches per token from the explicit decay sweep,
but increases state from 4KB to about 20KB and read traffic from 2B/query to
10B/query. This is a cleaner HCA hardware tradeoff: local SRAM metadata versus
global maintenance waves.

The twenty-fourth sweep compressed that metadata. Eight-bit per-counter epochs
are enough for the current 65k-token window at decay interval 256: state falls
from 20KB to 12KB, read traffic falls from 10B/query to 6B/query, and top-64,
top-256, and route accuracy remain 100%. The same 8-bit metadata also works at
decay 512, while decay 1024 gives about 98.4% top-64 and 99.6% top-256 recall.
Four-bit epochs reduce state to 8KB and reads to 4B/query, but require longer
decay intervals and begin to damage dense-topic quality: at decay 4096 top-256
recall falls to about 81.2%, and at decay 8192 top-64 recall falls to about
84.4%. The current best hand setting is therefore 4-bit counters plus 8-bit lazy
epoch metadata.

The twenty-fifth sweep added the wide64 CSA/HCA context summaries to the
unified event profile. The event traffic barely changes: with 4 Cellular-MoE
ticks, local traffic rises from the legacy 51.46KB/event to about
52.10KB/event. The added context path contributes about 648B/event: 6B HCA
summary read, 12B HCA update, 300B CSA block-summary scoring, and 330B selected
token-cell reads. The state budget changes much more. The 512KB block-summary
index plus 12KB 8-bit lazy HCA summary raise on-chip state from about 183.8KB
to about 707.8KB. Under the current 16KB/tile SRAM proxy, 32 tiles no longer
fit the state; 64 tiles fit at about 69.1% state utilization. This shifted the
next hardware pressure from event bandwidth to local SRAM capacity and
block-summary compression.

The twenty-sixth sweep compressed that CSA block-summary state. Holding the HCA
gate fixed at `width=2048`, threshold 8, and `csa_blocks=4`, the useful point is
`block_size=128`, `summary_width=256`: CSA block-summary state falls from 512KB
to 256KB, block-score traffic falls from about 300B/query to 150B/query, and
the routed CSA relevant subset still has 100% measured hit and 100% coverage in
the deterministic stream. The cost is larger selected blocks: average selected
token reads rise from about 165.5 positions/query to about 330.9, reducing the
full-context read reduction from about 396x to about 198x. In the unified
event-level profile this compact128 point raises context traffic to about
829.8B/event and total local traffic to about 52.28KB/event, but on-chip state
falls to about 451.8KB. The 32-tile, 16KB/tile floorplan now fits at about
88.2% SRAM utilization, requiring 29 state tiles. This is the first concrete
SRAM/bandwidth tradeoff that moves the design back toward a smaller CA chip
without breaking the current synthetic routing quality.

The twenty-seventh sweep adds a tiny exact rare-token block directory. This uses
the exact sparse lane to repair low-width CSA misses instead of widening every
block summary. With `block_size=128`, `summary_width=128`, and no directory, the
routed CSA subset has only about 68.9% hit and coverage. Adding one exact block
id per rare token raises hit to 100% and coverage to about 99.3%, using about
28.6KB of directory state. Adding two block ids raises reference-stream coverage
to 100% while using about 30.7KB. The combined CSA state is then 128KB of block
summaries plus about 30.7KB of directory entries, about 158.7KB total. Average
selected token reads stay about 331.6 positions/query, directory read traffic is
only about 0.48B/query, and the full-context token-read reduction stays about
198x.

The twenty-eighth sweep stress-tested that directory. The important failure was
not directory capacity at first; it was HCA admission. At the older threshold 8,
bursty rare tokens, split rare tokens, and repeated names are often falsely
routed to HCA before the directory can help. Raising the gate to threshold 15
keeps the reference-stream policy unchanged but cuts rare false-HCA routes in
the stress scenarios to about 0.8%. Under that safer gate, `dir_k=2` handles
burst and three-way split rare tokens, but repeated names spread across six
blocks still have only about 67.5% coverage. `dir_k=6` raises repeated-name
coverage to about 99.2%, with pure rare-query token-read reduction around 52x.
In the unified event profile, this rare128 point keeps local traffic about
52.28KB/event but lowers on-chip state to about 354.6KB. Under the same
32-tile, 16KB/tile floorplan, state utilization falls from compact128's 88.2%
to about 69.3%, requiring 23 state tiles. This is a better CA-chip split:
frequent distributed evidence belongs in HCA, fuzzy block proposals belong in
CSA, and rare exact location hints belong in a small associative directory.

The twenty-ninth sweep tested an exact-directory guard. Instead of raising the
global HCA threshold, the guard probes the rare-token directory before HCA
admission; a directory hit forces the query into CSA. This fixes the original
threshold-8 failure mode directly. On the repeated-name stress case, threshold 8
without the guard has 75% rare false-HCA and only 25% coverage. Threshold 8 with
the guard removes those false-HCA routes and recovers 100% coverage. The cost is
one small directory probe per query, about 3.25B/query on the reference stream,
and about 19.5B/query on repeated-name stress when all six block ids are read.
The current default remains the cheaper threshold-15 no-guard policy because it
gets about 99.2% repeated-name coverage with less average directory traffic, but
the guard is a useful exact-recall mode for names, code symbols, or other
sensitive rare tokens.

A related accounting correction remains important: candidate shortlist ranking
reads dense-sketch counters. In the gated synthetic LM this adds about 179.6
score cells per mixed event. Because these are 4-bit local reads, the unified
event-level profile rises only from about 51.38KB/event to about 51.46KB/event,
but the cells/event metric is now honest.

Current interpretation:

```text
Multi-route associative CA memory is a credible primitive, but the architecture
needs overflow or learned routing before it can replace attention for facts
that must be recalled exactly.
```

## Primary References

- Spitznagel and Keuper, "A New Kind of Network? Review and Reference
  Implementation of Neural Cellular Automata", arXiv:2604.24990.
- Bena et al., "A Path to Universal Neural Cellular Automata", arXiv:2505.13058.
- Lee et al., "Training Language Models via Neural Cellular Automata",
  arXiv:2603.10055.
- Pajouheshgar et al., "Neural Cellular Automata: From Cells to Pixels",
  arXiv:2506.22899.
- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing
  Unit", arXiv:1704.04760.
- Sze et al., "Efficient Processing of Deep Neural Networks: A Tutorial and
  Survey", arXiv:1703.09039.
