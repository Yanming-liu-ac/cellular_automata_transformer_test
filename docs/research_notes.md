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

The arXiv report `2606.19348` is therefore not merely adjacent literature; it is
the strongest current external blueprint for the memory-system side of this
project. The missing step is still the hard one: replace attention-centric
kernels with trainable CA dynamics and tiny local controllers without losing the
quality benefits that CSA/HCA provide.

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

The dynamic propagation sweep adds the first low-bit rollout check for the
first gate. Static shortest paths already show the HARC graph reaching the
oldest token in about 13, 17, and 21 ticks for 128, 512, and 2048 tokens,
instead of 127, 511, and 2047 ticks for a radius-1 line. The new rollout test
injects a 4-bit signal at the newest token and asks whether integer update rules
can actually carry it to the oldest token in 128 ticks. Plain `residual_avg`
stays stable but misses the far token even on the HARC graph because integer
amplitude diffuses too slowly. `route_max` reaches all tokens on HARC in 19,
23, and 27 ticks, proving the hierarchy can carry a low-bit wave, but it
saturates the whole route plane. The mHC-inspired `mhc_grouped` rule reaches all
tokens faster, in 16, 20, and 24 ticks, while keeping saturation around one
third of low-bit entries rather than 100%. This does not prove language
modeling, but it gives the first constructive evidence that grouped
route/local/envelope channels are a better CA primitive than a single scalar
diffusion state.

The next stability sweep removes the forced source and runs 4-bit HARC dynamics
for 1,000 ticks from sparse-random, dense-random, and structured-pulse initial
states. This separates "can propagate a pulse" from "can remain a usable
recurrent state." The scalar residual rule collapses into low-entropy rest
states. `route_max` is confirmed to be a propagation primitive only: dense
random inputs saturate to 100%, while other inputs still collapse to a uniform
high-level state. A naive `mhc_damped` leak prevents saturation but collapses to
zero, so simple damping is not the fix. The original `mhc_grouped` rule is the
only hand-coded rule in this sweep that avoids both collapse and global
saturation across all three initial states. It ends near a low-entropy
multi-channel attractor, about 1.58 bits of level entropy, which is stable but
not yet information-rich. The next trainable rule should preserve this grouped
stability scaffold while learning content-preserving perturbations and memory
injection.

The content-retention sweep makes that requirement measurable. It initializes
random 4-bit token content and rolls the HARC/mHC carrier for 1,000 ticks. If
the content is stored directly in the shared mHC carrier, exact retention is
only about 5.3% and mean absolute content error is about 45.0%; a stable carrier
alone is not a memory. Adding one explicit content latch lane raises token
content retention to 100.0% at 16 state bits/token, but the dynamic carrier
still remembers only about 5.7% of exact content on average. Periodic refresh
from the content lane into the carrier gives the expected traffic/visibility
tradeoff: refresh64 costs about 0.045 low-bit channel writes/token/tick and
raises average carrier exactness to 6.9%; refresh16 costs about 0.186 writes and
raises it to 12.0%; refresh8 costs about 0.375 writes and raises it to 19.1%
while reducing average carrier error to 17.3%. This is a constructive boundary:
HARC-CA needs a separate persistent content lane, and the trainable rule should
learn when to expose that content to the route/local carrier rather than
refreshing everything on a fixed schedule.

The first content-to-carrier gate sweep tests that idea with local mismatch
comparators. The `mismatch_ge8` gate writes only when the carrier content
differs from the persistent content lane by at least eight 4-bit levels. It
beats fixed refresh16 on the main hardware tradeoff: write traffic falls from
about 0.186 to 0.137 low-bit channel writes/token/tick, and average carrier
error drops from 28.5% to 21.9%, although average exact carrier matches fall
from 12.3% to 10.2%. Lowering the threshold to `mismatch_ge6` spends more
writes, about 0.250, but cuts average carrier error to 15.6%; `mismatch_ge4` reaches
21.4% average exactness and 10.0% average error at about 0.467 writes. The
budgeted top-error rows are useful upper bounds, but they use a global top-k
selection in the simulator. The hardware candidate is therefore a local
mismatch or learned local gate, not global budget sorting.

The first learned write-gate LUT makes that candidate concrete. The LUT has
only 64 states, 8 bytes, indexed by local content-carrier mismatch, route
level, and envelope level. With write cost 0.55 it enables two states. On an
independent rollout it matches the `mismatch_ge8` tradeoff: about
0.146 channel writes/token/tick, 11.2% average exact carrier matches, and
21.3% average carrier error. That beats fixed refresh16 on write traffic and
error, but it does not beat the hand threshold or the global budget-top upper
bound. The useful result is therefore not "learning solved content exposure";
it is that the gate fits in a tiny LUT and the next training objective must use
richer features or task-weighted demand labels to move beyond threshold-like
behavior.

The demand-weighted gate sweep gives that better label. It adds one local
route/query demand bit to the write-gate LUT, expanding the controller from
8 bytes to 16 bytes. With 5% demanded token cells per tick, the learned demand
LUT enables 7 of 128 states. It spends about 0.134 channel writes/token/tick and
delivers 96.6% exact content on demanded cells with only 0.4% demand error. For
comparison, fixed refresh16 spends 0.186 writes and reaches only 12.9% demanded
exactness, while the global `mismatch_ge8` gate spends 0.154 writes and reaches
11.5%. The demand-only hand upper bound, `demand_mismatch_ge1`, reaches 100.0%
at about 0.139 writes. This is the first genuinely better content-gate result:
the controller should not reconstruct the whole carrier; it should expose
persistent content only where the route/query lane asks for it.

The next sweep replaces random demand with real rare-directory query traces.
For each rare-directory query token, the demanded cells are the token occurrence
positions in the generated stress context, capped at six positions per query.
Training the same 16-byte LUT on split-rare traces and evaluating on
rare-burst, split-rare, and repeated-name traces gives the first workload-shaped
content gate result. Fixed refresh16 spends 0.187 channel writes/token/tick and
only reaches 10.3%-12.7% demanded exactness. The global `mismatch_ge8` gate
spends 0.140-0.152 writes and reaches 10.7%-11.4%. The learned trace LUT spends
only 0.028-0.034 writes and reaches 99.8%-100.0% demanded exactness. This is the
right direction: once demand comes from actual retrieval queries, the gate can
move content only at queried occurrence cells instead of reconstructing the
whole carrier.

The same trace-demand interface now works on the dual-path synthetic LM's exact
query events. Topic events demand no exact fact row; query events demand the
selected fact row. On the small 512-fact, 768-event diagnostic, fixed refresh16
still spends 0.187 channel writes/token/tick and reaches only 13.3% demanded
exactness. The learned exact-query trace LUT spends about 0.0019 writes and
reaches 99.6% demanded exactness. This is an even clearer systems result than
the rare-directory trace: when demand is truly sparse, content exposure should
be event-routed, not maintained continuously.

Adding candidate-output demand makes the bottleneck more honest. In the mixed
trace, query events still demand one exact fact row, but topic events demand the
64 candidate rows used for shortlist scoring. The demand fraction rises to
7.47%. The learned 16-byte LUT reaches 95.0% demanded exactness with only 0.3%
mean demand error, but it spends 0.178 channel writes/token/tick, close to
fixed refresh16's 0.187. This does not invalidate the CA path; it says the next
output-side rule must reduce candidate demand before content exposure, rather
than waking all top-k candidate rows every topic step.

A candidate-output sparsity sweep turns that warning into a design target. With
512 fact rows and 768 mixed events, a 16-byte learned gate costs 0.0057,
0.0287, 0.0489, 0.0937, and 0.1783 channel writes/token/tick when topic events
demand 1, 8, 16, 32, and 64 candidate rows respectively. Demanded exactness is
100.0%, 100.0%, 97.6%, 97.5%, and 95.0% on those points. The hand
`demand_mismatch_ge1` upper bound keeps 100.0% demanded exactness, confirming
that sparse candidate exposure is physically compatible with the CA rule.

The follow-up phase/rank gate closes the learned-control gap. A 9-byte LUT using
only demand phase, candidate-rank bucket, and content mismatch reaches 100.0%
demanded exactness at every tested candidate count. Its writes match the
`demand_mismatch_ge1` upper bound: 0.0057, 0.0287, 0.0502, 0.0964, and 0.1892
for 1, 8, 16, 32, and 64 demanded candidate rows. The earlier 16-byte
route/envelope-aware LUT missed sparse 2-row and 4-row candidate demand because
its dynamic state buckets were too fragmented. Hardware target: keep output
content demand near 8-16 rows, and use phase/rank/mismatch for the exact content
exposure gate.

The next diagnostic replaces oracle candidate-row counts with an actual low-bit
candidate reducer. It ranks a 512-row static candidate pool with the
topic-phase dense sketch, then exposes only top-M rows to the content gate. On
the small 512-fact, 768-event trace, top-64 has a 61.1% topic hit rate. Reducing
to 8, 16, and 32 rows gives 41.8%, 50.6%, and 56.1% hit rate, retaining 68.4%,
82.8%, and 91.7% of top-64 quality. The 9-byte content gate stays exact in all
cases. Channel writes per mixed event are 16.1, 28.8, 58.1, and 115.4 for 8,
16, 32, and 64 rows. This is the first real reducer result: top-16 is a useful
energy/quality point, but the current reducer still scores all 512 candidates
with 2,048 low-bit score reads per topic event.

A hierarchical reducer removes most of that scoring traffic in the model. It
groups the 512 candidate rows into 32 local groups of 16 rows, reads one
max-score summary per group, then fine-scores only candidates inside selected
groups. With top-16 demand and two selected groups, topic hit is 52.2%, or
85.3% of the top-64 baseline, while score reads fall from 2,048 to 256 per topic
event. With top-32 demand and four selected groups, topic hit is 57.2%, or 93.6%
of top-64, while score reads are 384 per topic event. The content gate remains
100.0% exact. This is the first evidence that candidate reduction can be
hierarchical rather than full-pool.

The group-summary maintenance diagnostic checks whether those summaries are
too expensive to keep exact. For 16-row groups, each topic update touches about
3.6 candidate rows and 3.4 groups on average. Exact summary maintenance costs
about 234 score-equivalent cells/topic, including group recompute, summary
writes, and decay shifts. Adding that to the hierarchical reducer gives about
490 cells/topic for top-16 and 618 for top-32, still 76.1% and 69.8% below the
2,048-cell full-pool scorer. Group size 16 is the current best balance; group
size 8 scans more summaries, while group size 32 makes each recompute too wide.

Lazy summary refresh reduces the update cost further. With 16-row groups and
top-16 demand, refreshing dirty summaries every 4 topic steps keeps 85.6% of
top-64 quality and drops total score work to 446 cells/topic. Refreshing every
16 topic steps still keeps 84.0% of top-64 quality and drops total score work
to 364 cells/topic, an 82.2% reduction from full-pool scoring. For top-32,
refresh-16 keeps 91.1% of top-64 quality at 492 cells/topic. Stale summaries
therefore look tolerable on this trace; the next step is to learn when to force
an early refresh rather than using a fixed interval.

Triggered summary refresh is the first local rule for that decision. The
`dirty_count_or_age` policy refreshes all dirty groups only when the local dirty
set reaches 16 groups or the summary age reaches 16 topic events. On the same
trace, top-16 keeps 85.6% of top-64 quality at about 421 cells/topic, slightly
cheaper than fixed refresh-4 and higher quality than fixed refresh-16. Top-32
keeps 93.3% at about 549 cells/topic. The more aggressive `top_dirty_or_age`
policy is cheaper, but it lets top-32 exact content exposure fall to 97.8%, so
the current lesson is conservative: refresh pressure should be driven by
dirty-count and age first, with learned top-dirty shortcuts treated as a later
risk/reward optimization.

## First Wiki-Memory Prototype

The CA wiki-memory prototype turns the mutable-knowledge idea into a measured
task. It builds a 256-page synthetic wiki with local facts, page links, 16-page
groups, low-bit page/group summaries, and 32 three-source contradiction
clusters. A query first scores group summaries, then page summaries, then reads
exact facts from selected pages; a multi-hop query also follows page links. A
fact update changes source-of-truth first, marks local page/group state dirty,
and relies on refresh or error-book repair to merge the new key, revised value,
or replicated claim value into routable memory cells. The benchmark now
separates route misses, value-stale misses, and multi-source cluster
consistency.

The exact-update policy reaches 100.0% recall but writes about 20,255
score-equivalent cells/update because it refreshes summaries after every fact
edit. The `trigger16_age16` policy reaches 94.73% overall recall and 92.08%
recent-update recall, while cutting writes to about 14,466 cells/update. Adding
page-local error-book repair raises overall recall to 97.66% and repeated
failed-probe recall to 98.54% at about 14,739 cells/update. Adding cluster
repair costs about 14,914 cells/update and forces 100.0% checked cluster
consistency, versus 93.06% for page-local repair. Query work is about 356-357
cells/query versus 1,024 for a flat exact fact scan, a 65% read reduction. The
flat/RAG-style page-summary baseline reaches the same accuracy under the same
refresh policies, but scans all page summaries and costs about 1,061 cells/query.
The no-refresh control reaches only 50.39% recall and 49.61% stale misses. This
is the strongest evidence so far that CA is naturally aligned with external
mutable knowledge: local state can remain mostly stale, as long as dirty/version
pressure, error-book probes, and cluster-local consolidation fire before route
quality collapses; hierarchy then gives the read advantage over flat page scans.

The follow-up scaling sweep checks that this is not a 256-page accident. Holding
the `trigger16_age16_clusterbook` policy fixed, 256, 512, 1,024, and 2,048-page
wikis produce CA reads/query of about 357, 420, 548, and 804 cells. The flat
page-summary scan reads about 1,061, 2,084, 4,132, and 8,228 cells/query on the
same workloads. CA read reduction versus flat scan grows from 66.3% to 90.2%,
while recall remains matched by construction. This supports the architectural
claim that wiki memory should be routed through local summaries rather than
global page-summary scans.

The density sweep adds the first failure mode. At 1,024 pages with 4x256x4-bit
summaries, increasing facts/page from 4 to 8, 16, and 32 drops CA recall from
98.83% to 77.93%, 30.47%, and 19.92%. Flat page-summary scan remains near
99.8% at width 256 because it scores every page, but still reads about 4.1K
page-summary cells/query. Narrowing summaries to width 128 makes collisions
visible even for flat scan: flat recall falls to 59.96% at 32 facts/page, while
CA recall is only 11.72%. This is useful evidence against a too-small fixed
fanout: dense pages need adaptive group selection or stronger summaries before
this becomes a credible memory chip path.

The adaptive group-fanout follow-up is the first repair for that failure mode.
On the same 1,024-page, 16 facts/page, width-256 stress point, fixed four-group
routing reaches only 30.47% recall at about 644 cells/query. Fixed 32-group
routing recovers 99.80% recall at about 2,445 cells/query. The adaptive rule
starts at four groups, expands only when group-summary scores are within a
low-bit margin, and caps at 32 groups; with margin 1 it also reaches 99.80%
recall at about 1,991 cells/query. This is still a hand rule, but it is the
right hardware shape: a small local comparator spends extra reads only when
summary evidence is ambiguous.

The learned fanout LUT makes the same idea trainable. Features are local and
low-bit: base group-summary score, top/base score gap, exact-tie group bucket,
and near-tie group bucket. A 1.1KB table trained on 32,737 self-supervised route
labels reaches 99.80% recall on the 16 facts/page stress point at about 1,566
cells/query. This is better than both fixed 32-group routing and the first hand
adaptive rule, so fanout control is now a real learned CA-chip primitive rather
than only a heuristic. A smaller/less conservative target can under-route on
some seeds, so the current default records the conservative `t100` table.
Across checked evaluation seeds 91, 123, 211, 307, 401, 503, and 607, `t100`
matches the hand adaptive recall while reading about 1,526-1,674 cells/query
instead of the hand adaptive 1,991-2,309 cells/query range.

The learned fanout grid turns this into a boundary map. For 8 facts/page,
learned fanout matches flat recall across 512, 1,024, and 2,048 pages while
cutting flat reads by 78.35%, 85.52%, and 87.95%. For 16 facts/page it still
matches flat at 512 and 1,024 pages, but at 2,048 pages the 32-group cap holds
recall to the hand adaptive level, 89.84%, while flat scan reaches 99.02%.
For 32 facts/page, learned fanout either becomes too expensive at small page
count or cannot recover enough candidate pages at large page count. This is a
useful failure: fanout control is not sufficient once page summaries saturate.

The dense routing-tile sweep resolves most of that failure by changing the
geometry rather than the score rule. Reducing group size from 16 pages to four
pages gives each group summary less page mixture, and a learned max48 fanout LUT
keeps the read path sparse. At 1,024 pages and 32 facts/page it recovers 99.80%
recall at about 1,697 cells/query, versus 59.38% and 2,581 cells/query for the
old learned max32 route. At 2,048 pages and 32 facts/page it reaches 99.22%
recall at about 2,897 cells/query, while flat scan reaches 95.12% at about
8,474 cells/query. The cost is roughly 96.6KB-192.6KB of extra summary state
over the tested page counts, which is a plausible CA-chip trade.

The density-aware tile sweep adds a quality guard to avoid blindly applying the
dense geometry everywhere. In a 2,048-page mixed wiki with 8 facts/page sparse
regions and 32 facts/page dense regions, the guard keeps 16-page tiles when the
dense region is only 25% of pages because the four-page tile lowers recall on
that small region. At 50% and 75% dense pages it turns the dense tile on and
recovers 99.22% and 99.32% recall, compared with 79.00% and 64.60% for uniform
16-page tiles. The read path stays far below flat scan: 1,158.5 versus 4,280.6
cells/query at 50% dense, and 1,842.6 versus 5,359.6 at 75% dense. This is the
first real density-aware routing policy in the wiki-memory track.

The refresh-derived density-tag sweep makes the policy more hardware-realistic.
The tag is a 2-bit value computed from fact density seen during summary refresh:
8 facts/page maps to tag 1 and 32 facts/page maps to tag 3. A tag threshold of
2 or 3 turns dense tiles on for the 32 facts/page region, but density alone
repeats the 25% dense failure: tag-only recall is 97.71% while the guarded
policy keeps 99.02%. The guard is no longer an offline region comparison: it
runs a 128-query / 64-update paired online probe, compares baseline and
dense-tile answers on the same queries, and compresses the decision into two
4-bit saturating counters. The hand rule is `c_win >= 3` and `c_loss == 0`.
The probe rejects 25% dense because the dense route stays at `0/0`, but enables
50% and 75% dense because 20 and 42 raw dense wins saturate to `15/0`. The
conclusion is precise: density tags are good wake signals, but geometry changes
need a local low-bit quality guard.

The mixed-stream version is the first less-oracular version of that guard. It
feeds sparse and dense regions from one 512-query / 256-update stream and stores
two 4-bit counters per 512-page guard block. Sparse false-enable remains 0.00%,
which is the safety property we need. Purely local counters under-cover the
50% dense case, enabling only one of two dense blocks even though aggregate
dense evidence is strong at 58/0 wins/losses. Same-tag one-hop sharing repairs
that without opening sparse blocks: 50% dense becomes 2/2 dense blocks enabled,
75% dense stays 3/3, and sparse false-enable remains 0.00%. This says the next
CA rule should learn when to share local evidence, not just when to flip a tile.

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

The thirtieth sweep separated directory storage fanout from directory read
fanout. Keeping six stored block ids is cheap in SRAM because most rare tokens
do not use all six slots. The question is how many to read per query. On
repeated-name stress, storing six but reading only two drops coverage to about
68%, whether the gate is threshold-15 no-guard or threshold-8 guard. Reading all
six recovers about 99.2% coverage for the cheap threshold-15 policy and 100%
coverage for the guarded threshold-8 policy. This says the architecture needs a
small learned or metadata-driven fanout predictor: compact rare tokens should
read one or two block ids, while repeated names and code symbols spread across
many blocks should read more.

The thirty-first sweep implemented the first metadata-driven fanout proxy. The
rule is deliberately hardware-shaped: each rare-directory row carries a tiny
spread class, the read path starts at fanout two, and a small LUT expands the
read only when stored block ids span at least 128 blocks. On repeated-name
stress, the guarded threshold-8 `span2to4` point reaches about 93.0% coverage at
13.0B/query of directory reads; `span2to5` reaches about 98.4% at
16.25B/query; and `span2to6` reaches 100.0% at 19.5B/query. On split rare
tokens, the same rule reads three blocks per hit and keeps 100.0% coverage. This
turns fanout into a low-bit control problem instead of a fixed `dir_k` choice.

The thirty-second sweep trains the fanout control table from self-supervised
coverage labels. Training uses exact block counts to choose the smallest fanout
that reaches the target coverage for a metadata bucket; inference uses only
entry count, span class, and overlap with the CSA-selected blocks. The resulting
112-entry 3-bit LUT is 42B. With guarded threshold-8 routing, it gets 100.0%
coverage on rare-burst stress at 3.25B/query, 99.7% on split-rare stress at
6.50B/query, and 98.4% on repeated-name stress at 12.87B/query. This matches
the hand `span2to5` repeated-name coverage with materially lower directory
traffic because the LUT can see when CSA already selected one of the rare
blocks.

The thirty-third sweep adds the first joint guard/probe/fanout diagnostic. A
40B probe LUT reads only HCA bank-counter metadata: count-min estimate,
counter spread across banks, and saturation count. It learns that strong
reference hot tokens are saturated in all banks with zero spread, so they do not
need a rare-directory probe. `confidence_probe` therefore keeps reference
directory traffic at 0.50B/query, versus 3.25B/query for `hca_probe` or
`always_probe`. On repeated-name stress it probes 74.2% of queries, keeps
coverage at about 97.7%, and spends 12.77B/query. On split-rare stress it gets
about 99.0% coverage at 6.45B/query. The remaining 0.8% false-HCA rate is a
real recall/traffic knob for the probe LUT, and should be jointly trained with
fanout and HCA threshold next.

The thirty-fourth sweep closed the accounting loop by promoting the joint
control state into the unified event profile. The current `joint128` budget keeps
the rare128 block-summary and exact-directory geometry, adds the 42B fanout LUT,
40B probe LUT, and about 2.25KB of per-row spread metadata, and accounts for
about 0.17B/event of control-LUT reads. Local traffic stays about 52.28KB/event,
while on-chip state rises only from 354.6KB to about 356.9KB. In the 32-tile
floorplan proxy, state utilization moves from 69.3% to about 69.7%, still
requiring 23 state tiles. This means learned probe/fanout control fits inside
the existing CA-chip budget instead of invalidating it.

The thirty-fifth sweep varies the HCA threshold under the same learned
probe/fanout control. Threshold 6 is rejected because split-rare coverage
collapses in the stress generator. Thresholds 8, 10, 12, and 15 have nearly the
same split/repeated coverage, but the early probe rate falls as the threshold
rises. At threshold 15, `confidence_probe` needs no early probe on split-rare or
repeated-name stress, keeps about 98.7% split-rare and 98.3% repeated-name
coverage, and spends the same 6.45B/query and 12.54B/query directory traffic as
the lower thresholds. This moves the current recommendation back to threshold
15, now with learned probe/fanout control instead of the old hand no-guard rule.

The thirty-sixth sweep trains a 40B HCA route LUT to replace the explicit
threshold at inference. The route table uses only HCA estimate, bank spread, and
saturation count, and it activates one HCA bucket in the current stress set. It
preserves reference HCA routing and keeps reference directory traffic at
0.50B/query. On split-rare stress it gets about 99.0% coverage at
6.47B/query; on repeated-name stress it gets about 97.7% coverage at
12.77B/query. This is close but not better than threshold-15 plus learned
fanout, so the current recommendation stays with the threshold-15 joint policy.
The value of the result is that it exposes the next missing feature: route
selection needs richer low-bit metadata or a training objective that prices rare
false-HCA routes more heavily.

The thirty-seventh sweep adds the smallest useful metadata feature: a
rare-directory presence bit visible to HCA admission. The route table doubles
from 40B to 80B and pays a modeled 0.125B/query sidecar read. That is enough to
keep the reference stream at 84.7% HCA routing, remove the remaining rare
false-HCA routes, reach 100.0% split-rare coverage at 6.65B/query, and reach
98.4% repeated-name coverage at 13.00B/query. This is a stronger CA-chip
primitive than the HCA-only route table because it exposes exactly the metadata
the compressed counters cannot infer from collisions.

The thirty-eighth sweep turns that presence bit into a Bloom-like sidecar with
false positives. The important failure mode is not rare recall: false positives
send extra queries to CSA, so split-rare coverage stays at 100.0% and
repeated-name coverage stays at 98.4% in the measured stress cases. The cost is
hot-path efficiency. On the reference stream, a 1% target sidecar has about
10.5KB state and lowers HCA routing from 84.7% to 82.1%; a 10% target uses about
5.3KB and keeps HCA routing at 80.0%; a 25% target drops HCA routing to 46.3%.
This gives the first concrete sidecar design target: 1-10% false positives may
be acceptable, but loose presence summaries destroy the dense hot path.

The thirty-ninth sweep instantiates the sidecar as an actual Bloom-style bit
array. The best first physical point is `8 bits/entry, k=3, 8 banks`: it uses
about 8.8KB on the reference case, reads 0.375B/query, has about 1.1% measured
sidecar false positives, keeps 84.2% of reference queries on HCA, and keeps the
rare stress cases at 100.0% split-rare and 98.4% repeated-name coverage. Moving
to `k=4` lowers false positives further but raises read traffic to 0.5B/query
and creates many same-bank read conflicts. This is the first result that looks
like a real CA-chip control SRAM rather than only an abstract routing label.

The fortieth sweep checks whether that physical sidecar is robust to hash-salt
choice. It is not automatically robust enough to ignore. With `8 bits/entry,
k=3, 8 banks`, 16 salts average 82.9% reference HCA routing and 2.1% hot-token
false positives. The best salt keeps HCA routing at 84.6% with only 0.2%
hot-token false positives; the worst salt drops HCA routing to 79.7% with 5.9%
hot-token false positives and 53.4% query bank conflict. Hash choice and bank
mapping therefore belong inside the control-plane optimization loop.

The forty-first sweep isolates bank mapping from false positives. With the same
Bloom geometry and salts, modulo banking averages 36.3% query bank conflict,
hashing the slot to a bank averages 37.6%, and assigning hash functions to banks
(`by_hash`) removes same-query bank conflicts in this model. HCA routing and
hot-token false positives are unchanged because the bit array is unchanged. This
is the first sidecar result that is purely a memory-layout win.

The forty-second sweep adds the first compiler-like salt selection objective.
Under `by_hash`, the sweep scans 16 salts on a reference selection stream and
chooses the salt with lowest mean hot-token false positives. The selected salt
has index 14. On the evaluation reference stream it gets about 1.1% sidecar
false positives, 0.9% hot-token false positives, 84.0% HCA routing, and no query
bank conflicts. The rare stress cases remain unchanged at 100.0% split-rare and
98.4% repeated-name coverage. This turns the sidecar from a fixed hash table
into a small compiled control memory.

The forty-third sweep tests the same selected sidecar under streaming inserts.
It shows that the static result is not enough. `final_oracle`, which inserts
only tokens that are rare at the end of the context, keeps the reference stream
at 84.0% HCA routing with about 0.052B/token of sidecar update traffic. But
simple count-threshold policies insert future hot tokens before they are known
hot. On the reference stream, `count1`, `count2`, `count4`, `count8`, and
`count14` all pollute 100.0% of the final hot-token set and collapse HCA routing
to 0.0%. Rare stress coverage remains high because the pollution routes more
queries to CSA, but the hot path loses its purpose. The next sidecar must be
delayed, counting/deletable, or paired with hot-token retirement.

The forty-fourth sweep adds that retirement path. The sidecar becomes a counting
Bloom filter with a fast 1-bit query plane and 4-bit update counters. When a
token reaches the HCA threshold, the sidecar deletes it from rare-token
presence. `count1_retire15` restores the useful oracle shape: reference HCA
routing returns to 84.0%, hot-token pollution falls to 0.0%, split-rare coverage
is 99.5%, and repeated-name coverage is 99.1%. The cost is visible but still
local: about 44-45KB of sidecar state and about 0.27B/context-token of update
traffic. Later insert thresholds such as `count2_retire15` reduce update traffic
to about 0.03-0.04B/token in the stress cases, but they leave most final rare
tokens out of the sidecar, so they should be treated as learned promotion
candidates rather than the conservative exact-recall baseline.

The forty-fifth sweep folds that conservative retirement sidecar into the
unified event and floorplan profiles. The new `retire128` profile keeps the
joint128 rare directory, learned probe/fanout control, lazy HCA summary, and
adds the measured `count1_retire15` sidecar budget. Local traffic stays about
52.28KB/event because the sidecar adds only about 0.38B/event of presence reads
and 0.28B/event of updates. State rises from about 356.9KB to about 401.8KB. In
the 32-tile, 16KB/tile floorplan proxy, SRAM utilization rises from about 69.7%
to about 78.5% and state tiles from 23 to 26. This keeps the online sidecar
inside the current local-SRAM budget, but it also makes sidecar compression the
next concrete hardware target.

The forty-sixth sweep compresses the retirement sidecar geometry. The decisive
new metric is visible rare-token rate after hot-token deletions, because the
logical active set can hide Bloom-counter false negatives. At `8 bits/entry`,
2-bit counters keep visible rare-token rate at 100.0% on the measured reference,
split-rare, and repeated-name streams, while preserving the 84.0% reference HCA
rate and 99.5%/99.1% rare-stress coverage. Sidecar state falls from about
44.9KB to about 26.9KB, and update traffic falls from about 0.27B/token to about
0.16B/token. One-bit counters reduce state further to about 18KB, but visible
rare-token rate falls to about 98.9-99.3%, so that point needs harder collision
tests before it can become the baseline. The current event profile therefore
moves from `retire128c4` to `retire128c2`: total on-chip state falls from about
401.8KB to about 383.8KB, and the 32-tile state utilization falls from about
78.5% to about 75.0%.

The forty-seventh sweep targets the failure mode that the forty-sixth sweep only
hinted at: hot-token deletions that share Bloom slots with rare tokens. The
adversarial generator first inserts rare tokens, then selects hot colliders with
shared Bloom slots and retires them. It now varies rare-token occurrences from 1
to 3 and colliders per rare token from 1 to 8. At `8 bits/entry`, 1-bit counters
collapse visible rare-token rate to 0.8% with one single-occurrence collider and
0.0% with eight, so they are rejected for exact sidecar use. Two-bit counters
keep 97.7%-98.4% visible rare-token rate with one collider, but fall to 62.5%
under the 8-collider chosen-deletion stress and no longer meet the robust
exact-memory contract. Three-bit counters restore 100.0% visible rare-token rate
through the repeated-key 8-collider stress. The current budget therefore moves
again, from `retire128c2` to `retire128c3`: total on-chip state is about 392.8KB,
32-tile utilization is about 76.7%, and state tiles are 25. This is the right
kind of regression: the sidecar remains inside the tile budget, while the
adversarial test prevents an over-compressed metadata format from becoming the
default. The repeated-key c3/c4 coverage ceiling is 95.3%, so the next failure
mode is directory/fanout read coverage rather than Bloom-sidecar deletion.

The forty-eighth sweep attacks that remaining repeated-key gap directly. It
fixes the sidecar at `retire128c3`, keeps the repeated-key 8-collider stress,
and sweeps the low-bit fanout LUT's minimum directory read count. Raising the
coverage target from 95% to 100% while keeping `min_read=2` does not move the
result: coverage remains 95.3%, with 2.00 directory entries and 6.88B of
directory traffic per query. A global `min_read=3` guard restores 100.0%
coverage at 10.12B/query. The better guard uses an existing LUT feature: when
CSA-selected blocks overlap zero directory entries, floor the read at three
entries. This zero-overlap guard also restores 100.0% coverage but needs only
7.33B/query, so the robust candidate becomes `retire128c3` plus a zero-overlap
three-entry fanout guard. It is a clean CA-chip trade: a tiny local comparator
closes the adversarial coverage hole without changing the sidecar state format
or expanding the LUT dimensions.

The forty-ninth sweep checks that guard against the normal threshold-15 fanout
profile before promoting it into the unified event budget. The zero-overlap
guard leaves reference traffic unchanged at 3.25B/query, rare-burst traffic
unchanged at 3.25B/query, and repeated-name traffic unchanged at 12.87B/query.
It changes only the split-rare zero-overlap corner in this sweep: coverage moves
from 99.7% to 100.0%, average directory entries per hit move from 2.00 to 2.01,
directory traffic moves from 6.50B/query to 6.53B/query, and token-read
reduction barely moves from 84.8x to 84.7x. The unified profile therefore keeps
the same measured normal reference event traffic and state as `retire128c3`:
about 52.28KB/event and about 392.8KB on-chip state. The next policy target is
not more SRAM; it is richer context metadata for deciding when zero-overlap is a
real rare-token miss rather than harmless CSA disagreement.

The fiftieth sweep tests the obvious delayed-promotion shortcut and rejects it.
Keeping the robust `8 bits/entry, 3-bit counter` sidecar, a pure
`count2_retire15` gate cuts normal update traffic from about 0.22B/token to
about 0.03B/token, and `count3_retire15` cuts it to about 0.015B/token. But both
break the exact sidecar contract: visible rare-token rate falls to 7%-9% for
`count2` and about 0%-2% for `count3` across reference, split-rare, and
repeated-name stress. In the repeated-key collision case, insert thresholds 1,
2, and 3 all keep 100.0% sidecar visibility because the constructed rare token
appears three times, but they save no meaningful update traffic; insert 4 fails
immediately. The result is a useful negative: delayed promotion cannot be a
plain count threshold. It needs extra local evidence, such as directory-probe
feedback, short recency, source phase, or a tiny probation state, while keeping
count1-style protection for one-hit rare facts.

The fifty-first sweep tests that probation idea directly. It keeps the full
counting sidecar on delayed `count2_retire15` promotion and adds optional
first-hit visibility. A persistent first-hit presence Bloom restores 100.0%
rare-token visibility, but it is not usable as a default because it also makes
100.0% of final hot tokens look like rare-sidecar hits and reaches 85.2%
sidecar false positives on the reference query stream. A deletable
first-hit-retiring probation plane is much closer to hardware-useful: with
8 bits/entry and 1-bit counters it keeps 99.1%-99.4% rare-token visibility,
limits hot pollution to at most 2.0%, and cuts update traffic to about
0.136-0.141B/token instead of the count1 baseline's about 0.22B/token. The cost
is real: total sidecar state rises to about 53KB and every query reads both the
full and probation sidecars, 0.75B/query. A `directory_feedback_oracle` row gives
the upper bound: if a local exact directory signal could expose one-hit rare
facts, count2 would keep 100.0% visibility with the low 0.027-0.030B/token
update traffic and only about 0.5B/query sidecar read. Therefore the next
promotion target is not a bigger Bloom filter; it is a local feedback/probe rule
that approaches the oracle without the persistent-probation hot-path pollution.

A small probation bit-budget check sharpens the tradeoff. Two bits/entry is too
collision-prone, with up to 27.7% hot pollution and 48.8% false positives.
Four bits/entry is the aggressive low-state point: about 44.9KB maximum sidecar
state, at least 98.5% rare visibility, up to 6.6% hot pollution, and the same
about 0.139B/token update traffic. Eight bits/entry is the cleaner robust point:
about 53.9KB maximum state, at least 99.1% rare visibility, and at most 2.0% hot
pollution. This is a useful candidate family, but not yet the default robust
architecture.

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
