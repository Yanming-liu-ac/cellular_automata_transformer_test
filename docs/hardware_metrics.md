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

## Propagation Rollout Metrics

Static graph distance is necessary but not sufficient. For low-bit CA dynamics,
also track:

- source-to-farthest-token reach tick;
- all-token reach tick;
- final token reach fraction;
- active low-bit entry fraction;
- final and peak saturation fraction;
- mean low-bit level after rollout.

The current 4-bit dynamic sweep uses 128 ticks and injects the newest token as a
source pulse. On a 2048-token line, neither scalar residual diffusion nor the
faster route wave reaches the oldest token in time. On the HARC graph,
`route_max` reaches all token cells in 27 ticks but saturates the route state.
The mHC-inspired grouped rule reaches all token cells in 24 ticks while limiting
saturation to about 33.3% of entries. This is the first measured reason to keep
separate local, route, and stability-envelope channels in the CA cell format.

The 1,000-tick unforced stability check uses the same 4-bit HARC graph at
512 tokens and starts from sparse random, dense random, and structured pulses.
`residual_avg` and `route_max` collapse to low-entropy fixed states; `route_max`
also reaches 100% saturation from dense random input. A leaky `mhc_damped`
variant avoids saturation by erasing the state, so it is rejected. `mhc_grouped`
keeps active state without global saturation across all three starts, but its
final entropy is only about 1.58 bits. The next metric target is therefore not
just "stable for 1,000 ticks"; it is stable with content entropy preserved under
trained local rules.

Content-retention adds another hardware-facing gate:

- exact token-content retention after rollout;
- average carrier exactness, not only final carrier exactness;
- normalized carrier content error;
- persistent state bits/token;
- refresh channel writes/token/tick.

In the current 512-token, 1,000-tick HARC sweep, shared mHC state stores only
about 5.3% of random 4-bit content exactly. Adding one persistent content lane
keeps content retention at 100.0% and raises per-token state from 12 to
16 bits, but the carrier still averages only about 5.7% exactness. Refreshing
the carrier from the content lane improves carrier visibility at a direct local
write cost: refresh64 is 0.045 writes/token/tick for 6.9% average carrier
exactness, refresh16 is 0.186 writes/token/tick for 12.0%, and refresh8 is
0.375 writes/token/tick for 19.1%. This makes content exposure a learned
write-gating problem, not a fixed always-refresh rule.

The first local write-gate point improves that budget. `mismatch_ge8` compares
the persistent content lane with the mHC carrier and writes only when their
4-bit difference is at least eight levels. It needs about 0.137 channel
writes/token/tick, lower than refresh16, and lowers average carrier error from
28.5% to 21.9%, while average exact carrier matches fall from 12.3% to 10.2%.
The more aggressive `mismatch_ge6` point costs about 0.250 writes/token/tick
and lowers average error to 15.6%; `mismatch_ge4` costs about 0.467 and lowers
average error to 10.0%. The budgeted top-error rows should be tracked only as upper bounds
because global top-error selection is not a natural local CA primitive.

The first learned write-gate LUT is 8 bytes: 64 one-bit actions over mismatch,
route, and envelope buckets. With write cost 0.55 it enables two actions and,
on an independent seed, lands on the same tradeoff as `mismatch_ge8`: about
0.146 channel writes/token/tick and 21.3% average carrier error. Fixed refresh16
on the same seed needs 0.186 writes/token/tick and has 28.9% average error. The
hardware conclusion is that the controller size is negligible; the remaining
problem is better labels/features, not table storage.

Demand-weighted labels change the result. Adding one route/query demand bit
makes the LUT 128 bits, or 16 bytes. With 5% demand rate, the learned demand LUT
uses about 0.134 channel writes/token/tick, below fixed refresh16's 0.186 and
below global `mismatch_ge8`'s 0.154. It reaches 96.6% exact content on demanded
cells with about 0.4% demand error. The global carrier still has high average
error, about 33.1%, but that is now acceptable for this control objective: the
chip should move persistent content into the carrier only where computation
needs it, not refresh the full context field.

On rare-directory query traces, the same controller shape is much stronger
because demand is sparse and structured. With six demanded occurrence cells per
query, a 16-byte trace-trained LUT spends about 0.033 channel
writes/token/tick on rare-burst and split-rare, and about 0.028 on repeated-name.
Demanded exact content is 99.8%-100.0%. Fixed refresh16 costs 0.187
writes/token/tick and leaves demanded exactness around 10%-13%. This is the
current best evidence that demand-routed content exposure can keep local write
traffic low while preserving exact queried content.

On the dual-path synthetic exact-query trace, demand is even sparser: one fact
row on query events and no exact row on topic events. The learned trace LUT
spends about 0.0019 channel writes/token/tick and reaches 99.6% demanded
exactness. Fixed refresh16 still costs 0.187 writes/token/tick. This suggests
the content lane can remain mostly idle during topic-only decode steps and wake
only for exact-memory demand.

The mixed exact+candidate trace is the first warning that output-side demand can
eat the savings. With 512 fact rows plus 64 candidate rows, topic events demand
all candidate rows used for shortlist scoring, raising demand to 7.47% of
token-cells per tick. The learned 16-byte LUT reaches 95.0% demanded exactness
with 0.178 writes/token/tick, only slightly below fixed refresh16. The hardware
target therefore needs candidate pruning, hierarchical candidate routing, or a
separate low-bit scorer path that avoids exposing every candidate row's content.

The candidate-demand sparsity sweep gives a practical budget line. At 8
candidate rows, the learned LUT writes 0.0287 channels/token/tick with 100.0%
demanded exactness. At 16 rows, it writes 0.0489 with 97.6% demanded exactness.
At 32 rows, writes rise to 0.0937, still about half of fixed refresh16. At 64
rows, writes reach 0.1783, nearly the fixed-refresh cost. The output hardware
therefore needs an early low-bit candidate reducer that keeps content-demanded
rows near the 8-16 range for normal topic steps.

A smaller exactness-oriented candidate gate fixes the sparse-row misses. The
phase/rank/mismatch LUT is only 72 bits, or 9 bytes. It reaches 100.0% demanded
exactness for 1-64 candidate rows, with writes equal to the local
`demand_mismatch_ge1` upper bound. The useful budget points are 0.0287 writes at
8 rows and 0.0502 writes at 16 rows. At 64 rows the exact gate costs 0.1892
writes, so the chip still needs candidate reduction before exact content
exposure.

The first low-bit reducer trace uses the topic-phase dense sketch to rank all
512 candidate rows before the content gate. It confirms the expected budget:
top-16 content demand costs 28.8 channel writes per mixed event and preserves
82.8% of top-64 topic-hit quality; top-32 costs 58.1 writes/event and preserves
91.7%. The top-64 baseline costs 115.4 writes/event. The remaining hardware
cost is candidate scoring: this first reducer still reads 2,048 low-bit score
cells per topic event, so the next reducer needs hierarchical or bank-local
top-k selection rather than scoring every candidate row every topic step.

The group-summary reducer is the first answer to that cost. With 32 summaries
over 16-row groups, top-16 with two selected groups reads 128 summary cells plus
128 fine-score cells per topic event, or 256 total. It keeps 85.3% of top-64
topic-hit quality and costs 28.0 content-gate channel writes per mixed event.
Top-32 with four selected groups reads 384 score cells per topic event, keeps
93.6% of top-64 quality, and costs 57.4 content-gate writes/event. This reduces
candidate-score reads by 81%-88% before exact exposure. The unmodeled hardware
question is how cheaply each group max summary can be maintained during updates.

The exact maintenance estimate says the summary path still has margin. For
16-row groups, summary state is about 64 bytes. A topic update impacts about
3.4 groups on average, so exact recompute plus summary writes and decay shifts
costs about 234 score-equivalent cells/topic. Including this maintenance,
top-16 hierarchical scoring costs about 490 cells/topic and top-32 costs about
618, versus 2,048 for full-pool scoring. The net reduction remains 76% and 70%,
respectively.

Lazy refresh improves that budget. With 16-row groups, top-16 and refresh-16
costs about 364 score cells/topic while preserving 84.0% of top-64 topic-hit
quality. Top-32 with refresh-16 costs about 492 cells/topic while preserving
91.1%. Both keep the exact content gate at 100.0% demanded exactness. This
suggests group summaries can be maintained with dirty/lazy local updates rather
than exact recompute every topic step.

Triggered refresh adds a local controller to the same path. The current best
conservative rule, `dirty_count_or_age`, refreshes dirty groups when either 16
groups are dirty or the summary is 16 topic steps old. It gives top-16 about
421 score cells/topic with 85.6% top-64 quality retention, and top-32 about 549
score cells/topic with 93.3% retention. This is not a dramatic new minimum, but
it proves the maintenance decision can be made from local dirty bits and an age
counter rather than from a global scheduler.

## Wiki-Memory Metrics

For the CA wiki-memory path, track separate read and update costs because the
goal is not only retrieval speed. Mutable knowledge must be cheap to edit. The
first prototype uses 256 pages, four facts per page, four links per page,
16-page groups, and 4x256x4-bit summaries. It keeps about 146.7KB of page,
group, fact, link, version, and dirty metadata.

The exact-update policy reaches perfect recall but writes about 20,255
score-equivalent cells per fact edit. The `trigger16_age16` local policy keeps
94.73% recall with 5.27% stale misses and writes about 14,466 cells/update.
Adding page-local error-book repair raises recall to 97.66% and repeated-probe
recall to 98.54% at about 14,739 cells/update. Adding cluster repair costs about
14,914 cells/update and raises checked multi-source cluster consistency to
100.0%, versus 93.06% for page-local repair. Reads stay about 356-357
cells/query, compared with 1,024 exact fact cells for a flat scan and about
1,061 cells/query for a flat/RAG-style all-page-summary scan. The no-refresh
control proves the failure mode: writes fall to about seven metadata
cells/update, but stale misses climb to 49.61%. This makes stale miss rate,
value-miss rate, error-probe recall, cluster consistency, and page-summary scan
traffic the main safety metrics for any more aggressive write-saving policy.

Scaling makes the read metric sharper. With the same 4x256x4-bit summaries and
clusterbook repair, page counts 256, 512, 1,024, and 2,048 give hierarchical CA
reads of about 357, 420, 548, and 804 cells/query. The flat page-summary scan
costs about 1,061, 2,084, 4,132, and 8,228 cells/query. The CA route therefore
cuts flat-scan reads by 66.3%, 79.9%, 86.7%, and 90.2%, respectively. Writes
are the same under the same repair policy, so this sweep isolates routing
traffic.

Density pressure changes the story. At 1,024 pages, facts/page 4, 8, 16, and 32
with width-256 summaries give CA reads of about 548, 582, 644, and 773
cells/query, but recall drops from 98.83% to 77.93%, 30.47%, and 19.92%. Flat
scan reads about 4.1K-4.4K cells/query and keeps high recall at width 256. With
width-128 summaries, collision pressure hurts both paths, and flat recall at
32 facts/page falls to 59.96%. Hardware implication: page-count scaling favors
hierarchical CA routing, but page-density scaling requires adaptive fanout or
wider/more structured summaries.

Adaptive fanout is the first concrete density repair. At 1,024 pages and 16
facts/page, fixed four-group routing reads about 644 cells/query but only
answers 30.47% of queries. Fixed 32-group routing reaches 99.80% recall at
about 2,445 cells/query. Adaptive `g4_max32_margin1` reaches the same 99.80%
recall at about 1,991 cells/query, while the flat page-summary scan reads about
4,237 cells/query. The hardware target is therefore not a larger fixed fanout;
it is a low-bit ambiguity detector that increases local group reads only when
summary scores are tied or near-tied.

The learned version of that detector is small enough for local SRAM. The first
LUT uses 1,152 bytes of fanout-control state, trained from 32,737 minimal-route
labels. Its conservative `t100` setting reaches 99.80% recall at about 1,566
cells/query, cutting flat page-summary reads by 63.0% and hand adaptive reads by
about 21.3%. This makes group fanout a concrete low-bit control table rather
than a fixed architecture constant.

The grid sweep adds the hardware boundary. At 8 facts/page, learned fanout keeps
flat-level recall while reading 459, 604, and 996 cells/query for 512, 1,024,
and 2,048 pages. At 16 facts/page it still wins at 512 and 1,024 pages, but at
2,048 pages the 32-group cap limits recall to 89.84%. At 32 facts/page, larger
fanout alone is no longer the right primitive: reading all 32 groups at 512
pages costs slightly more than flat page-summary scan, while 1,024 and 2,048
pages need more than 32 groups for flat-level recall. The next hardware target
is a denser page summary or a second-stage page-local index, not only a bigger
fanout cap.

Smaller routing tiles are the first dense-page repair. Moving from 16-page
groups to four-page groups increases summary state by about 96.6KB at 1,024
pages and 192.6KB at 2,048 pages, plus a 1.69KB LUT, but it restores dense
recall without flat scans. At 1,024 pages and 32 facts/page, dense tiles read
about 1,697 cells/query and match flat's 99.80% recall; flat reads about 4,378
cells/query. At 2,048 pages and 32 facts/page, dense tiles read about 2,897
cells/query and reach 99.22% recall, versus flat at 8,474 cells/query and
95.12% recall. This is a better hardware move than simply raising fanout on
16-page groups, because it improves the signal before the fanout decision.

Density-aware tile sizing turns that into a conditional hardware policy. The
mixed-region sweep stores a 1-bit density tag per page, 256B for 2,048 pages,
and enables four-page tiles only when a local quality probe says dense recall
does not drop. At 25% dense pages the guard rejects the small tile and saves
12.62% state versus all-four-page tiling. At 50% dense pages it spends 96.84KB
extra state over the uniform 16-page baseline, but improves recall from 79.00%
to 99.22% and cuts flat reads by 72.94%. At 75% dense pages it spends 144.85KB
extra state, improves recall from 64.60% to 99.32%, and cuts flat reads by
65.62%. This is the current best CA-chip trade for mixed wiki density.

The refresh-derived tag version keeps that state tiny. A 2-bit tag per page is
512B for a 2,048-page wiki and can be generated while refreshing page summaries
by counting local fact slots. The tag alone is not sufficient: at 25% dense
pages it would enable the small tile and reduce recall to 97.71%. With the
local 128-query / 64-update paired online guard, recall remains 99.02%; at 50%
and 75% dense pages the same tag enables the dense path and keeps the 72.94%
and 65.62% flat-read reductions. The guard uses two 4-bit saturating counters
per 16-page guard block, so the 2,048-page sweep adds only 128B of counter
state. The hand rule is `c_win >= 3` and `c_loss == 0`, making the guard a
small canary workload or online agreement counter rather than a global oracle.
This separates the hardware roles cleanly: density tag wakes the alternative
geometry, quality guard commits it.

The mixed-stream counter diagnostic changes the state accounting. Using two
4-bit counters per 512-page guard block, the 2,048-page mixed stream needs only
4B of counter state. The locality sweep keeps sparse false-enable at 0.00%
across 256, 512, and 1,024-page guard blocks. At 50% dense, local dense
coverage is 2/4, 1/2, and 1/1 for those block sizes; same-tag sharing raises
the 512-page case to 2/2 at radius 1 and the 256-page case to 4/4 at radius 2.
At 75% dense, all three block sizes are already full locally. In the 50% dense,
512-page, radius-1 stress, shared dense coverage stays 2/2 across 128/64 to
1,024/512 query/update windows. Raising revision updates to 80%, cluster
updates to 60%, or both together keeps shared dense coverage at 2/2 and sparse
false-enable at 0.00%. The state is cheap enough; the hardware problem is now
controlled evidence sharing across local blocks.

The learned guard LUT is smaller than the counters. With three guard block
geometries, radii 0-2, decay modes none/win/nonloss, and loss tolerances 0-1,
plus win-threshold deltas -1/0/+1, the table is 21 bits, or 2.625B. It chooses
radius 2/decay win/loss 0/dwin +1 for 256-page blocks, radius 1/decay
win/loss 0/dwin +1 for 512-page blocks, and radius 0/decay win/loss 0/dwin +1
for 1,024-page blocks. That restores 50% dense coverage to 100% for the finer
blocks while keeping sparse false-enable at 0.00% on the training stream. The
former held-out seed 1501 failure is now a positive stress case: a 99/1 dense
wins/losses trace at 75% dense no longer trips the guard because decay-on-win
restores 100% learned dense coverage with strict `loss == 0`, a one-count
higher win threshold, and 0.00% sparse false-enable. The next hardware state
should test a learned base win-threshold scale and a larger held-out seed/noise
audit.

The first loss-tolerance audit fixes the 512-page/radius-1 geometry and tests
seeds 1201, 1301, 1401, and 1501. Strict `loss=0` has one dense-on failure,
87.50% mean dense-on coverage, zero off-region enables, and 0.00% max sparse
shared false-enable. Tolerant `loss=1` has zero dense-on failures, 100.00% mean
dense-on coverage, zero off-region enables, and the same 0.00% max sparse
false-enable. In a seed1501 high-update-noise case with revision updates at 80%
and cluster updates at 60%, both gates keep 100.00% dense-on coverage and
0.00% sparse false-enable. The hardware implication is that one extra low-bit
comparison state can remove the observed brittle gate without increasing
measured false-enable on this audit.

The first event-driven loss-decay check uses no extra counter bits. It keeps the
strict `loss=0` gate and updates only the existing 4-bit loss counter. On the
seed1501 75% dense 99/1 trace, no decay leaves dense max at `15/1` and shared
coverage at 0/3. Decay-on-win and decay-on-nonloss both reduce the final dense
max to `15/0`, restore 3/3 shared coverage, and keep sparse false-enable at
0.00%; the 25% dense off row remains off. The hardware implication is that
loss recovery can be done as a local counter transition rather than by adding
larger state or global arbitration.

The first noise matrix checks the same low-bit state under update-rate changes.
For seeds 1501 and 1601, 25% dense off and 75% dense on, and base/revision-80%/
cluster-60%/combined regimes, strict `loss=0` has two dense-on failures and
75.00% mean dense-on coverage. Tolerant `loss=1` has zero dense-on failures,
100.00% mean dense-on coverage, zero off-region enables, and 0.00% sparse
false-enable across every row. This keeps the hardware story simple: the
observed reliability gain comes from one additional low-bit comparison against
the existing loss counter, not from a global controller or a high-precision
score.

The deterministic randomized-noise smoke audit covers four more update-rate
samples: revision/cluster rates of 34%/50%, 33%/42%, 51%/42%, and 44%/20%.
Both `loss=0` and `loss=1` pass this sample with 100.00% mean dense-on
coverage, zero off-region enables, and 0.00% sparse false-enable. The sample
does not contain a dense-loss failure for `loss=1` to repair, so its value is
mainly a false-enable check: one extra loss-count tolerance did not increase
measured spurious activation under these randomized update rates.

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

The repeated-read budget curve keeps the same 512KB index and varies selected
block count while retaining the 2-block exact tail:

```text
selected=4    reads about 384 token positions   coverage about 5.6%   reduction about 170.7x
selected=8    reads about 640 token positions   coverage about 8.4%   reduction about 102.4x
selected=16   reads about 1152 token positions  coverage about 13.5%  reduction about 56.9x
selected=32   reads about 2176 token positions  coverage about 22.1%  reduction about 30.1x
selected=64   reads about 4222 token positions  coverage about 32.9%  reduction about 15.5x
selected=128  reads about 8310 token positions  coverage about 46.1%  reduction about 7.9x
```

The measured gap to an exact oracle top-block selector remains small, about
0.04 to 0.26 percentage points across this curve. This shifts the next hardware
question away from better block ranking and toward a policy decision: use sparse
block reads for rare/exact details, compressed recurrent summaries for
high-frequency distributed evidence, and repeated reads only when the query
requires more attention mass.

The first CSA/HCA routing policy adds a 4KB global low-bit summary and reads
only 2 bytes of global counters per query. The policy uses that estimate to skip
block scoring for frequent HCA-path queries and reserve CSA block reads for
low-frequency queries. With `csa_blocks=4`, `tail_blocks=2`, and threshold 8:

```text
HCA query rate: about 85.4%
CSA query rate: about 14.7%
hot -> HCA: 100.0%
cold -> CSA: 100.0% on the measured relevant cold subset
CSA-path hit/coverage: 100.0% / 100.0% on the routed relevant subset
average block-score reads: about 300B/query instead of 2KB/query
average token block reads: about 165 token positions/query
full-context token-read reduction: about 396x
```

This is the first useful read-policy metric. It should not be interpreted as
overall language quality. The sparse coverage over all relevant queries is low
by design because frequent queries are delegated to the HCA-like summary rather
than rereading many historical blocks.

The CSA block-summary state sweep keeps the same HCA gate (`width=2048`,
threshold 8) and asks how far the block index can be compressed:

```text
block=64   width=128  state=256KB  score=300B/query  csa hit/cov=96.0% / 95.3%   token reads=165.5  reduction=396.0x
block=64   width=256  state=512KB  score=300B/query  csa hit/cov=100% / 100%     token reads=165.5  reduction=396.0x
block=128  width=128  state=128KB  score=150B/query  csa hit/cov=68.9% / 68.9%   token reads=330.8  reduction=198.1x
block=128  width=256  state=256KB  score=150B/query  csa hit/cov=100% / 100%     token reads=330.9  reduction=198.1x
block=256  width=128  state=64KB   score=75B/query   csa hit/cov=40.5% / 40.5%   token reads=661.7  reduction=99.0x
block=256  width=256  state=128KB  score=75B/query   csa hit/cov=90.5% / 89.2%   token reads=661.6  reduction=99.1x
```

The best block-only SRAM tradeoff is `block_size=128`, `summary_width=256`: it
halves CSA block-summary state to 256KB and preserves measured CSA-path
reliability in this stream. The cost is doubled selected-token traffic versus
64-token blocks, but the full-context read reduction remains about 198x.

The rare-token block directory then repairs a lower-width CSA point with a
small exact sparse structure:

```text
block=128  width=128  dir_k=0  block=128KB  dir=0KB     combined=128KB    hit/cov=68.9% / 68.9%    token reads=330.8
block=128  width=128  dir_k=1  block=128KB  dir=28.6KB  combined=156.6KB  hit/cov=100% / 99.3%     token reads=331.6
block=128  width=128  dir_k=2  block=128KB  dir=30.7KB  combined=158.7KB  hit/cov=100% / 100%      token reads=331.6
block=128  width=128  dir_k=6  block=128KB  dir=30.8KB  combined=158.8KB  hit/cov=100% / 100%      token reads=331.6
block=256  width=256  dir_k=1  block=128KB  dir=27.5KB  combined=155.5KB  hit/cov=100% / 100%      token reads=662.1
```

The best current point is `block_size=128`, `summary_width=128`, `dir_k=6`, and
a safer HCA route threshold of 15. It uses about 158.8KB for CSA block summaries
plus exact rare-token block ids,
restores measured routed-CSA hit and coverage to 100%, and adds only about
0.48B/query of directory read traffic on the reference stream. This shifts rare exact location recall
out of the compressed block summary and into the exact sparse lane.

The stress sweep exposes the two failure modes that the reference stream hides:
false HCA admission and repeated rare names spread across many blocks. With the
safer threshold 15 and `block_size=128`, `summary_width=128`:

```text
rare_burst      dir_k=2  false-HCA=0.8%  repaired hit/cov=99.2% / 99.2%  reduction=85.9x
split_rare      dir_k=2  false-HCA=0.8%  repaired hit/cov=99.2% / 99.0%  reduction=85.2x
repeated_name   dir_k=2  false-HCA=0.8%  repaired hit/cov=99.2% / 67.5%  reduction=64.8x
repeated_name   dir_k=6  false-HCA=0.8%  repaired hit/cov=99.2% / 99.2%  reduction=52.2x
collision_noise dir_k=2  false-HCA=0.8%  repaired hit/cov=99.2% / 99.2%  reduction=85.8x
```

This makes the current policy explicit: `dir_k=2` is enough for compact
reference traffic, but `dir_k=6` is the safer repeated-name setting. Pure
rare-query stress spends more token reads than the average event profile,
because exact rare-detail preservation is intentionally more expensive.

An exact-directory guard is the conservative alternative to raising the HCA
threshold. With `dir_k=6`:

```text
t8_no_guard   repeated_name  false-HCA=75.0%  coverage=25.0%   dir read=4.9B/query   reduction=129.5x
t8_guard      repeated_name  false-HCA=0.0%   coverage=100.0%  dir read=19.5B/query  reduction=51.9x
t15_no_guard  repeated_name  false-HCA=0.8%   coverage=99.2%   dir read=19.4B/query  reduction=52.2x
```

On the reference stream, `t8_guard` adds about one 3.25B directory probe per
query without changing token reads. This is small in the average profile but
should be a policy mode, not always-on hidden behavior.

Separating stored fanout from read fanout gives the first explicit policy table:

```text
cheap_t15_read6   reference      false-HCA=0.0%  coverage=2.5%    dir read=0.5B/query   reduction=195.6x
cheap_t15_read6   repeated_name  false-HCA=0.8%  coverage=99.2%   dir read=19.4B/query  reduction=52.2x
guard_t8_read6    repeated_name  false-HCA=0.0%  coverage=100.0%  dir read=19.5B/query  reduction=51.9x
guard_t8_read2    repeated_name  false-HCA=0.0%  coverage=68.0%   dir read=6.5B/query   reduction=64.4x
cheap_t15_read2   repeated_name  false-HCA=0.8%  coverage=67.4%   dir read=6.5B/query   reduction=64.8x
```

The directory should therefore store enough block ids for repeated rare names,
but read fanout should be chosen from metadata. A learned policy can spend
`read6` only when the rare token is spread across many blocks.

The first metadata fanout proxy uses a small spread class: base fanout is two,
and the directory expands only when the stored rare-token block ids span at
least 128 blocks. On the guarded threshold-8 policy:

```text
guard_t8_span2to4  split_rare     coverage=100.0%  dir read=9.75B/query   avg read=3.0 blocks/hit
guard_t8_span2to4  repeated_name  coverage=93.0%   dir read=13.0B/query   avg read=4.0 blocks/hit
guard_t8_span2to5  repeated_name  coverage=98.4%   dir read=16.25B/query  avg read=5.0 blocks/hit
guard_t8_span2to6  repeated_name  coverage=100.0%  dir read=19.5B/query   avg read=6.0 blocks/hit
```

This is a usable Pareto knob for hardware: about 2.2KB of 2-bit spread metadata
in the current 64K-token context is enough to steer the read fanout table. The
next version should learn the thresholds instead of hand-setting the 128-block
span.

The first trained fanout LUT replaces the hand threshold with self-supervised
coverage labels. It uses entry-count, span-class, and CSA-overlap metadata. The
LUT table itself is only 42B; including per-row spread metadata, the current
stress runs account for about 2.3KB of fanout metadata:

```text
learned_lut  rare_burst     coverage=100.0%  dir read=3.25B/query   avg read=1.00 blocks/hit
learned_lut  split_rare     coverage=99.7%   dir read=6.50B/query   avg read=2.00 blocks/hit
learned_lut  repeated_name  coverage=98.4%   dir read=12.87B/query  avg read=3.96 blocks/hit
```

Compared with the hand `span2to5` point, the learned LUT keeps the same
repeated-name coverage but cuts directory reads from 16.25B/query to
12.87B/query by using CSA-overlap as an extra visible feature.

The next control table learns when to issue the early rare-directory probe. It
is a 40B HCA-confidence LUT indexed by HCA estimate, bank-counter spread, and
saturation count:

```text
confidence_probe  reference      probe=0.0%   coverage=2.5%   dir read=0.50B/query
hca_probe         reference      probe=84.7%  coverage=2.5%   dir read=3.25B/query
confidence_probe  split_rare     probe=77.3%  coverage=99.0%  dir read=6.45B/query
confidence_probe  repeated_name  probe=74.2%  coverage=97.7%  dir read=12.77B/query
hca_probe         repeated_name  probe=75.0%  coverage=98.4%  dir read=12.87B/query
```

The important hardware result is the reference row: the confidence LUT removes
the early probe for strong saturated HCA hits while preserving almost all rare
recall. The remaining false-HCA rate is about 0.8% on split/repeated stress,
which can be traded against probe traffic by changing the probe table training
target.

Sweeping HCA threshold under the same `confidence_probe` control shows the next
policy simplification:

```text
t6   split_rare     probe=97.7%  false-HCA=0.0%  coverage=0.0%   dir read=3.17B/query
t8   repeated_name  probe=77.3%  false-HCA=0.8%  coverage=98.3%  dir read=12.54B/query
t10  repeated_name  probe=25.0%  false-HCA=0.8%  coverage=98.3%  dir read=12.54B/query
t12  repeated_name  probe=5.5%   false-HCA=0.8%  coverage=98.3%  dir read=12.54B/query
t15  repeated_name  probe=0.0%   false-HCA=0.8%  coverage=98.3%  dir read=12.54B/query
t15  split_rare     probe=0.0%   false-HCA=0.8%  coverage=98.7%  dir read=6.45B/query
```

The earlier threshold-8 conservative mode is no longer the best default once
the learned exact-memory control plane exists. Threshold 15 keeps the same
measured recall in this stress set, avoids early probes, and still uses the LUT
fanout when the token is routed to CSA.

The first trained HCA route LUT tests whether the threshold can disappear from
inference. It is a 40B table over HCA estimate, bank spread, and saturation
count:

```text
hca_route_lut  reference      HCA=84.7%  false-HCA=0.0%  coverage=2.5%   dir read=0.50B/query
hca_route_lut  split_rare     HCA=0.8%   false-HCA=0.8%  coverage=99.0%  dir read=6.47B/query
hca_route_lut  repeated_name  HCA=0.8%   false-HCA=0.8%  coverage=97.7%  dir read=12.77B/query
```

This route LUT is not the new default because repeated-name coverage remains
below the threshold-15 joint policy's 98.3%. It does prove that HCA admission can
be encoded as a very small local table; the next route table should add recency
or topic/context metadata before replacing the hand threshold.

A directory-aware route LUT adds exactly one such visible feature: a rare-token
directory presence bit. The table doubles to 80B and the model charges a
0.125B/query presence-sidecar read before admission:

```text
dir_aware_route_lut  reference      HCA=84.7%  false-HCA=0.0%  coverage=2.5%    dir read=0.19B/query
dir_aware_route_lut  split_rare     HCA=0.0%   false-HCA=0.0%  coverage=100.0%  dir read=6.65B/query
dir_aware_route_lut  repeated_name  HCA=0.0%   false-HCA=0.0%  coverage=98.4%   dir read=13.00B/query
```

This is now the stronger learned-admission diagnostic: it beats the HCA-only
route LUT and slightly improves repeated-name coverage versus the threshold-15
joint policy, while preserving the reference HCA hot path. The remaining
hardware question is whether the presence feature is a true 1-bit sidecar read
or a more expensive associative probe.

The first presence-sidecar false-positive sweep models that question as a
Bloom-like summary. The route LUT is unchanged at 80B and still reads one
presence bit per query; the sidecar state varies with target false-positive
rate:

```text
fp=0%    reference  sidecar=1.10KB   fp_q=0.0%   HCA=84.7%  coverage=2.5%    reduction=195.6x
fp=1%    reference  sidecar=10.54KB  fp_q=2.8%   HCA=82.1%  coverage=2.6%    reduction=188.1x
fp=10%   reference  sidecar=5.27KB   fp_q=6.0%   HCA=80.0%  coverage=2.8%    reduction=182.4x
fp=25%   reference  sidecar=3.17KB   fp_q=41.7%  HCA=46.3%  coverage=3.6%    reduction=123.3x
fp=10%   split_rare sidecar=5.39KB   fp_q=0.0%   HCA=0.0%   coverage=100.0%  reduction=84.7x
fp=10%   repeated   sidecar=5.27KB   fp_q=0.0%   HCA=0.0%   coverage=98.4%   reduction=52.4x
```

The useful conclusion is asymmetric: sidecar false positives are mostly an
efficiency risk for hot reference traffic, not an exact-recall risk in this
stress set. A target around 1-10% looks plausible; 25% is too loose for the HCA
hot path.

The next sweep replaces the abstract false-positive target with a concrete
Bloom-style sidecar. It inserts the rare-directory token ids into a bit array,
queries `k` hashed bits per token, and measures false positives and bank
conflicts directly. On the reference stream:

```text
bpe=4   k=2  sidecar=4.40KB   read=0.25B/query  fp_q=9.3%   HCA=77.7%  q_bank_conflict=4.9%
bpe=4   k=3  sidecar=4.40KB   read=0.38B/query  fp_q=10.5%  HCA=76.0%  q_bank_conflict=48.0%
bpe=8   k=2  sidecar=8.80KB   read=0.25B/query  fp_q=3.1%   HCA=82.5%  q_bank_conflict=7.4%
bpe=8   k=3  sidecar=8.80KB   read=0.38B/query  fp_q=1.1%   HCA=84.2%  q_bank_conflict=19.9%
bpe=8   k=4  sidecar=8.80KB   read=0.50B/query  fp_q=0.7%   HCA=84.5%  q_bank_conflict=62.5%
bpe=12  k=3  sidecar=13.19KB  read=0.38B/query  fp_q=0.4%   HCA=84.4%  q_bank_conflict=30.9%
```

The current concrete candidate is `8 bits/entry, k=3, 8 banks`: it uses about
8.8-9.0KB sidecar SRAM across the stress cases, reads 3 bits/query, writes
3 bits per rare-directory insertion, spends about 0.052B of sidecar update
traffic per context token, keeps reference HCA routing at 84.2%, keeps
split-rare coverage at 100.0%, and keeps repeated-name coverage at 98.4%. The
next physical question is bank layout: `k=4` gives lower false positives but too
many same-bank read conflicts under the simple modulo-bank model.

A hash-salt robustness sweep tests the recommended `8 bits/entry, k=3, 8 banks`
point across 16 salts on the reference stream:

```text
mean over salts: HCA=82.9%  hot_fp=2.1%
worst salt:      fp_q=5.4%  hot_fp=5.9%  HCA=79.7%  q_bank_conflict=53.4%  reduction=181.7x
best salt:       fp_q=0.4%  hot_fp=0.2%  HCA=84.6%  q_bank_conflict=14.6%  reduction=195.2x
```

This turns hash choice into a first-class hardware/compiler knob. The sidecar
should choose salts against the hot-token query distribution, not only against
global Bloom false-positive rate.

The first bank-mapping sweep compares the same `8 bits/entry, k=3, 8 banks`
sidecar across 16 salts:

```text
modulo    mean HCA=82.9%  mean hot_fp=2.1%  mean q_bank_conflict=36.3%  HCA range=79.7%-84.6%
by_hash   mean HCA=82.9%  mean hot_fp=2.1%  mean q_bank_conflict=0.0%   HCA range=79.7%-84.6%
hash_slot mean HCA=82.9%  mean hot_fp=2.1%  mean q_bank_conflict=37.6%  HCA range=79.7%-84.6%
```

`by_hash` gives the first clean layout win: it assigns each Bloom hash function
to its own bank, so the sidecar keeps the same false-positive behavior but
removes same-query bank conflicts in this model. The physical caveat is that
each hash function now needs a banked address path, but that is a local SRAM
layout issue rather than a model-quality tradeoff.

With `by_hash` fixed, a salt-selection sweep chooses the Bloom salt that
minimizes hot-token false positives on a held-out reference selection stream:

```text
selected salt index=14  salt=30775
reference      fp_q=1.1%  hot_fp=0.9%  HCA=84.0%  q_bank_conflict=0.0%  reduction=193.5x
split_rare     fp_q=0.0%  hot_fp=0.0%  HCA=0.0%   q_bank_conflict=0.0%  coverage=100.0%
repeated_name  fp_q=0.0%  hot_fp=0.0%  HCA=0.0%   q_bank_conflict=0.0%  coverage=98.4%
```

This is better than leaving the salt fixed at an arbitrary average point:
reference HCA routing is near the ideal sidecar while the bank-conflict rate is
zero. Salt selection is now part of the sidecar compiler contract.

The next sweep tests whether that selected sidecar can be updated online with a
simple count-threshold insertion rule. It cannot be the default rule as-is.
`final_oracle` inserts only tokens that are rare at the end of the context, so
it is an upper bound rather than a streaming policy:

```text
reference final_oracle  inserted=9007  rare_in=100.0%  hot_poll=0.0%    update=0.05154B/token  hot_fp=0.9%    HCA=84.0%  reduction=193.5x
reference count1        inserted=9263  rare_in=100.0%  hot_poll=100.0%  update=0.05300B/token  hot_fp=100.0%  HCA=0.0%   reduction=85.3x
reference count2        inserted=911   rare_in=7.3%    hot_poll=100.0%  update=0.00521B/token  hot_fp=100.0%  HCA=0.0%   reduction=85.3x
reference count14       inserted=256   rare_in=0.0%    hot_poll=100.0%  update=0.00146B/token  hot_fp=100.0%  HCA=0.0%   reduction=85.3x
split_rare final_oracle inserted=9170  rare_in=100.0%  hot_poll=0.0%    update=0.05247B/token  coverage=99.5%  reduction=84.9x
split_rare count1       inserted=9425  rare_in=100.0%  hot_poll=100.0%  update=0.05393B/token  coverage=99.5%  reduction=84.9x
repeated final_oracle   inserted=9195  rare_in=100.0%  hot_poll=0.0%    update=0.05261B/token  coverage=99.1%  reduction=52.7x
repeated count1         inserted=9449  rare_in=100.0%  hot_poll=100.0%  update=0.05407B/token  coverage=99.1%  reduction=52.7x
```

The important failure mode is temporal. A token that is hot by the end of the
context still crosses count 1, 2, 4, 8, and 14 on the way there, so naive
streaming insertion pollutes the rare-token sidecar before the chip knows the
token is hot. This does not damage rare recall in these stress cases because it
routes more queries through CSA, but it destroys the hot HCA fast path. The next
sidecar should therefore be counting/deletable, or should use delayed promotion
with a hot-token retirement rule.

The first repair uses a counting Bloom sidecar with a fast 1-bit query plane and
4-bit update counters. Query traffic stays at 3 presence bits, but sidecar state
rises from about 8.8KB to about 44KB and updates write 5 bits per hash slot
instead of 1 bit. When a token reaches the HCA threshold, the sidecar deletes it
from rare-token presence:

```text
reference count1_retire15   state=44.0KB  active_rare=100.0%  hot_retired=100.0%  hot_poll=0.0%  update=0.27234B/token  HCA=84.0%  reduction=193.5x
reference count2_retire15   state=44.0KB  active_rare=7.3%    hot_retired=100.0%  hot_poll=0.0%  update=0.03339B/token  HCA=84.7%  reduction=195.6x
reference count14_retire15  state=44.0KB  active_rare=0.0%    hot_retired=100.0%  hot_poll=0.0%  update=0.01465B/token  HCA=84.7%  reduction=195.6x
split_rare count1_retire15  state=44.8KB  active_rare=100.0%  hot_retired=100.0%  hot_poll=0.0%  update=0.27695B/token  coverage=99.5%  reduction=84.9x
split_rare count2_retire15  state=44.8KB  active_rare=8.7%    hot_retired=100.0%  hot_poll=0.0%  update=0.03734B/token  coverage=99.5%  reduction=84.9x
repeated count1_retire15    state=44.9KB  active_rare=100.0%  hot_retired=100.0%  hot_poll=0.0%  update=0.27761B/token  coverage=99.1%  reduction=52.7x
repeated count2_retire15    state=44.9KB  active_rare=8.5%    hot_retired=100.0%  hot_poll=0.0%  update=0.03682B/token  coverage=99.1%  reduction=52.7x
```

This is the first online sidecar point that survives the hot-path pollution
test. The conservative hardware baseline is `count1_retire15`: it matches the
oracle sidecar contents at the end of the window and restores the reference HCA
route rate. The lower-update `count2_retire15` point is attractive, but it is a
quality-risk knob because many one-hit rare tokens are never inserted. The next
work item is therefore to compress or gate the retirement sidecar without losing
the exact rare-token contract.

The first compression sweep varies counting-Bloom bits per rare entry and
counter width while keeping `count1_retire15` retirement:

```text
reference bpe=4 counter=1  state=8.8KB   visible_rare=98.2%   fp_q=9.5%  HCA=76.9%  update=0.10894B/token
reference bpe=4 counter=2  state=13.2KB  visible_rare=100.0%  fp_q=11.0% HCA=75.5%  update=0.16340B/token
reference bpe=6 counter=1  state=13.2KB  visible_rare=98.8%   fp_q=5.3%  HCA=80.0%  update=0.10894B/token
reference bpe=6 counter=2  state=19.8KB  visible_rare=100.0%  fp_q=5.7%  HCA=79.6%  update=0.16340B/token
reference bpe=8 counter=1  state=17.6KB  visible_rare=98.9%   fp_q=1.0%  HCA=84.1%  update=0.10894B/token
reference bpe=8 counter=2  state=26.4KB  visible_rare=100.0%  fp_q=1.1%  HCA=84.0%  update=0.16340B/token
reference bpe=8 counter=4  state=44.0KB  visible_rare=100.0%  fp_q=1.1%  HCA=84.0%  update=0.27234B/token
split_rare bpe=8 counter=2 state=26.9KB  visible_rare=100.0%  coverage=99.5%
repeated bpe=8 counter=2   state=26.9KB  visible_rare=100.0%  coverage=99.1%
```

The normal-stream compressed point is `8 bits/entry, 2-bit counters`. It keeps
the measured rare-token visibility at 100% in this sweep, preserves the same HCA
and rare-stress coverage as 4-bit counters, and cuts sidecar state from about
44.9KB to about 26.9KB. The 1-bit counter point is attractive at about 18KB, but
it produces a small rare-token visibility loss, so it should remain an
aggressive candidate.

The delayed-promotion diagnostic keeps the robust `8 bits/entry, 3-bit counter`
sidecar but raises the insert threshold:

```text
reference     count1_retire15  visible_rare=100.0%  update=0.21787B/token  HCA=84.0%
reference     count2_retire15  visible_rare=7.3%    update=0.02671B/token  HCA=84.7%
reference     count3_retire15  visible_rare=0.3%    update=0.01243B/token  HCA=84.7%
split_rare    count1_retire15  visible_rare=100.0%  update=0.22156B/token  coverage=99.5%
split_rare    count2_retire15  visible_rare=8.7%    update=0.02987B/token  coverage=99.5%
split_rare    count3_retire15  visible_rare=1.8%    update=0.01543B/token  coverage=99.5%
repeated_name count1_retire15  visible_rare=100.0%  update=0.22208B/token  coverage=99.1%
repeated_name count2_retire15  visible_rare=8.5%    update=0.02946B/token  coverage=99.1%
repeated_name count3_retire15  visible_rare=1.7%    update=0.01529B/token  coverage=99.1%
```

Naive delayed promotion cuts updates sharply, but it breaks the exact sidecar
visibility contract by skipping one-hit rare facts. The coverage numbers remain
high only because CSA/fanout still reads many blocks; that is not a valid exact
sidecar replacement.

The probation-promotion diagnostic tests the missing local evidence. With
`count2_retire15` as the full-sidecar promotion gate, a persistent first-hit
presence plane restores 100% rare visibility but is rejected because it leaves
hot tokens permanently polluted; in the reference stream, hot pollution is
100% and sidecar false positives reach 85.2%. A deletable first-hit probation
plane with 1-bit counters is the useful candidate. At 8 probation bits/entry it
keeps 99.1%-99.4% rare visibility, holds hot pollution below 2.0%, and reduces
update traffic to about 0.136-0.141B/token. The hardware price is about
52.8-53.9KB total sidecar state and 0.75B/query sidecar read, because the route
must check both the delayed full sidecar and the probation plane. Four
probation bits/entry is the aggressive option: maximum state falls to about
44.9KB, rare visibility remains at least 98.5%, but hot pollution rises to
6.6% and reference false positives to 8.7%. Two bits/entry is too collision
heavy. The oracle directory-feedback row is only an upper bound: it shows that
the same count2 update traffic, about 0.027-0.030B/token, would be possible if a
local exact directory/probe signal exposed one-hit rare facts without Bloom
pollution.

The adversarial-collision sweep chooses hot tokens that share Bloom slots with
rare tokens before retiring them. It now varies both the number of rare-token
occurrences and the number of hot colliders per rare token:

```text
rare_occ=1  colliders/rare=1  counter=1  mean_overlap=1.59  visible_rare=0.8%    false_HCA=0.8%  coverage=99.2%
rare_occ=1  colliders/rare=1  counter=2  mean_overlap=1.59  visible_rare=97.7%   false_HCA=0.0%  coverage=100.0%
rare_occ=1  colliders/rare=1  counter=3  mean_overlap=1.59  visible_rare=100.0%  false_HCA=0.0%  coverage=100.0%
rare_occ=1  colliders/rare=8  counter=1  mean_overlap=8.95  visible_rare=0.0%    false_HCA=6.2%  coverage=93.8%
rare_occ=1  colliders/rare=8  counter=2  mean_overlap=8.95  visible_rare=62.5%   false_HCA=3.9%  coverage=96.1%
rare_occ=1  colliders/rare=8  counter=3  mean_overlap=8.95  visible_rare=100.0%  false_HCA=0.0%  coverage=100.0%
rare_occ=3  colliders/rare=1  counter=2  mean_overlap=1.60  visible_rare=98.4%   false_HCA=0.0%  coverage=94.5%
rare_occ=3  colliders/rare=8  counter=1  mean_overlap=8.95  visible_rare=0.0%    false_HCA=7.8%  coverage=88.0%
rare_occ=3  colliders/rare=8  counter=2  mean_overlap=8.95  visible_rare=62.5%   false_HCA=3.1%  coverage=92.2%
rare_occ=3  colliders/rare=8  counter=3  mean_overlap=8.95  visible_rare=100.0%  false_HCA=0.0%  coverage=95.3%
```

This demotes c2 from "robust baseline" to "normal-stream compression point."
The current robust sidecar target is `8 bits/entry, 3-bit counters`: it keeps
adversarial visible rare-token rate at 100% through the repeated-key
8-collider stress while cutting sidecar state from about 44.9KB to about
35.9KB. The remaining 95.3% repeated-key coverage at c3/c4 is no longer a
sidecar deletion problem; it points to the directory/fanout read budget.

The same repeated-key stress also confirms why a pure count gate is not enough:

```text
insert=1  visible_rare=100.0%  coverage=95.3%  update=0.06152B/token
insert=2  visible_rare=100.0%  coverage=95.3%  update=0.06152B/token
insert=3  visible_rare=100.0%  coverage=95.3%  update=0.06152B/token
insert=4  visible_rare=0.0%    coverage=88.0%  update=0.05859B/token
```

Because this constructed rare token appears exactly three times, thresholds up
to three survive but save no meaningful update traffic; threshold four fails.
The promotion gate therefore needs a richer local feature than count alone.

The repeated-key fanout-budget sweep keeps the same `retire128c3` sidecar and
the same low-bit fanout LUT, then varies the minimum directory reads and the
zero-overlap guard:

```text
min_read=2  zfloor=0  target=95%  lut=42B  dir_entries/q=2.00  dir_read=6.88B/q   visible_rare=100.0%  coverage=95.3%   token_read_reduction=78.2x
min_read=2  zfloor=3  target=95%  lut=42B  dir_entries/q=2.14  dir_read=7.33B/q   visible_rare=100.0%  coverage=100.0%  token_read_reduction=76.6x
min_read=3  zfloor=0  target=95%  lut=42B  dir_entries/q=3.00  dir_read=10.12B/q  visible_rare=100.0%  coverage=100.0%  token_read_reduction=76.6x
```

Raising the training target alone does not fix this repeated-key corner. A
global three-entry minimum directory read is robust but over-reads. The better
hardware guard is zero-overlap floor 3: when CSA-selected blocks overlap none of
the exact rare-directory entries, read at least three entries. It restores 100%
coverage with only about 0.45B/query more directory traffic in this stress.

The threshold-15 normal fanout-guard sweep checks whether that guard is too
expensive on non-adversarial cases:

```text
min_read=2  zfloor=0  reference      coverage=2.5%    avg_read=1.05  dir_read=3.25B/query   token_read_reduction=195.6x
min_read=2  zfloor=3  reference      coverage=2.5%    avg_read=1.05  dir_read=3.25B/query   token_read_reduction=195.6x
min_read=2  zfloor=0  rare_burst     coverage=100.0%  avg_read=1.00  dir_read=3.25B/query   token_read_reduction=85.4x
min_read=2  zfloor=3  rare_burst     coverage=100.0%  avg_read=1.00  dir_read=3.25B/query   token_read_reduction=85.4x
min_read=2  zfloor=0  split_rare     coverage=99.7%   avg_read=2.00  dir_read=6.50B/query   token_read_reduction=84.8x
min_read=2  zfloor=3  split_rare     coverage=100.0%  avg_read=2.01  dir_read=6.53B/query   token_read_reduction=84.7x
min_read=3  zfloor=0  split_rare     coverage=100.0%  avg_read=3.00  dir_read=9.75B/query   token_read_reduction=84.7x
min_read=2  zfloor=0  repeated_name  coverage=98.4%   avg_read=3.96  dir_read=12.87B/query  token_read_reduction=52.4x
min_read=2  zfloor=3  repeated_name  coverage=98.4%   avg_read=3.96  dir_read=12.87B/query  token_read_reduction=52.4x
```

The selective zero-overlap guard therefore supersedes the earlier global g3
rule: it fixes the repeated-key collision corner and split-rare coverage while
the reference and repeated-name normal paths remain at the same directory
traffic.

The HCA-like global summary is now measured separately. At threshold 8:

```text
width=512   state=1KB  saturation=89.6%  top64=25.0%  top256=32.0%   query route acc=85.4%
width=1024  state=2KB  saturation=27.8%  top64=53.1%  top256=48.4%   query route acc=94.6%
width=2048  state=4KB  saturation=11.8%  top64=42.2%  top256=94.1%   query route acc=100.0%
width=4096  state=8KB  saturation=6.1%   top64=51.6%  top256=100.0%  query route acc=100.0%
```

This separates route quality from dense-state quality. The 4KB global summary is
already enough for the current hand threshold policy on the query stream, but
top-64 frequency recall remains weak even at 8KB because 4-bit counters saturate
on the hottest tokens. A deployable HCA-like path therefore needs a better
frequency-preserving state, such as decay, group scales, per-block residual
summaries, or higher-precision metadata on selected channels.

Periodic decay is the first anti-saturation fix. With the same 4KB
`width=2048` global summary and a decayed-state threshold of 2:

```text
decay=64    top64=100.0%  top256=100.0%  route acc=100.0%  decay cells/token=128.0
decay=128   top64=100.0%  top256=100.0%  route acc=100.0%  decay cells/token=64.0
decay=256   top64=100.0%  top256=100.0%  route acc=100.0%  decay cells/token=32.0
decay=512   top64=100.0%  top256=100.0%  route acc=100.0%  decay cells/token=16.0
decay=1024  top64=98.4%   top256=99.6%   route acc=100.0%  decay cells/token=8.0
no decay    top64=42.2%   top256=94.1%   route acc=88.2%   decay cells/token=0.0
```

This validates decay as an HCA anti-saturation mechanism on the current
topic/noise stream. It also exposes the maintenance cost: synchronous all-cell
decay can dominate the 4-counter update cost unless it is scheduled as a
background tile operation, amortized, or replaced by scale metadata.

Lazy epoch decay is the first replacement for synchronous all-cell decay. Each
counter stores epoch metadata and applies the decay shift only when read or
updated:

```text
width=2048, counter_bits=4, epoch_bits=16
state: 20KB instead of 4KB
read bytes/query: 10B instead of 2B
update cells/token: 4.0
synchronous decay cells/token: 0.0 instead of 32.0 at decay=256
top64/top256 recall: 100.0% / 100.0%
route accuracy: 100.0%
```

The metric exposes a chip design tradeoff: epoch metadata increases SRAM and
read width, but removes global maintenance traffic and preserves the decayed
HCA target exactly enough for the current benchmark.

The first metadata-width sweep improves that tradeoff:

```text
epoch=8   decay=256   state=12KB  read=6B/query   top64=100.0%  top256=100.0%  route acc=100.0%
epoch=8   decay=512   state=12KB  read=6B/query   top64=100.0%  top256=100.0%  route acc=100.0%
epoch=8   decay=1024  state=12KB  read=6B/query   top64=98.4%   top256=99.6%   route acc=100.0%
epoch=4   decay=4096  state=8KB   read=4B/query   top64=98.4%   top256=81.2%   route acc=99.8%
epoch=4   decay=8192  state=8KB   read=4B/query   top64=84.4%   top256=93.8%   route acc=99.9%
epoch=16  decay=256   state=20KB  read=10B/query  top64=100.0%  top256=100.0%  route acc=100.0%
```

The best current HCA summary point is 8-bit epoch metadata at decay 256 or 512:
it keeps perfect measured quality while using 40% less state and read width than
the 16-bit epoch baseline. Four-bit epochs are possible only with longer decay
windows and show measurable dense-topic degradation.

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

## CA Wiki Cell v0 Metrics

The LLM-Wiki direction now has a cell-level mutable-memory proxy. The default
configuration stores 128 claims across eight source-page cells each, with
low-bit value/revision/confidence fields, local source links, and one 4-bit
error-book counter per claim. Total state is about 4.72KB.

The measured tradeoff separates query critical path from background repair:

```text
sample_no_repair:      38.38% recall, 2.00 source reads/query, 2.44 touch/event
flat_scan:            100.00% recall, 8.00 source reads/query, 7.24 touch/event
tile_update_ca:       100.00% recall, 2.00 source reads/query, 16.00 touch/event
error_book_ca:         83.59% recall, 2.00 source reads/query, 14.35 touch/event
hybrid_error_book_ca:  96.00% recall, 2.00 source reads/query, 17.49 touch/event
```

`tile_update_ca` also leaves zero stale source cells, while flat scan still
leaves 73.14% stale source cells because it answers by reading around
inconsistency. The chip metric lesson is therefore not "CA already wins total
traffic." It is: CA gives the lower-latency sparse query path and explicit
local consistency repair, but the next controller must learn when repair pulses
are worth their maintenance traffic.

The first repair-schedule controller is a 3.75B LUT over six fan-in/update
pressure buckets. It chooses from 28 local schedules encoded by radius,
update-repair ticks, update-repair period, and error-book repair ticks. Under a
90% recall / 85% recent-recall / 10% stale-source target:

```text
4 sources, 128 updates:  ca_r3_u1p4_e1, about 4.0 touch/event
4 sources, 256 updates:  ca_r3_u1p4_e1, about 5.3-5.4 touch/event
8 sources, 128 updates:  ca_r4_u0p1_e1, about 9.2-9.5 touch/event
8 sources, 256 updates:  ca_r4_u1p2_e1, about 14.8-15.2 touch/event
16 sources, 128 updates: ca_r4_u1p4_e1, about 30.3-32.4 touch/event
16 sources, 256 updates: ca_r4_u1p1_e1, about 55.1-56.7 touch/event
```

The learned table misses 2 of 24 evaluated rows at the chosen targets. That is
now a hardware metric, not a hidden caveat: fan-in wider than the local repair
radius quickly turns maintenance traffic into the dominant cost.

Strict/budget mode stores two learned policy ids per bucket, 7.50B total. The
strict target is 99% recall, 98% recent recall, and 1% stale source cells; the
budget target is 90%, 85%, and 10%. Evaluation over the same 24 rows gives:

```text
strict: 0/24 target failures, 22.22 mean touch/event
budget: 2/24 target failures, 20.12 mean touch/event
traffic saved by budget mode: 9.47%
```

Per-bucket budget savings versus strict mode:

```text
4 sources, 128 updates:   0.62%
4 sources, 256 updates:   4.24%
8 sources, 128 updates:   4.86%
8 sources, 256 updates:   6.06%
16 sources, 128 updates: 12.40%
16 sources, 256 updates: 10.42%
```

This is the first chip-facing quality mode for CA Wiki Cell: a tiny mode bit can
trade maintenance traffic for a measured failure budget.

A second-level claim-summary lane changes the high fan-in metric. On the
16-source, 256-update case:

```text
flat source scan:      100.00% recall, 16.00 reads/query, 13.63 touch/event, 79.83% stale sources
strict source repair:  100.00% recall,  2.00 reads/query, 62.40 touch/event,  0.00% stale sources
summary_only:          100.00% recall,  1.00 reads/query,  1.20 touch/event, 79.83% stale sources
summary_probe1:        100.00% recall,  2.00 reads/query,  2.00 touch/event, 79.83% stale sources
summary_period4:       100.00% recall,  2.00 reads/query,  4.35 touch/event, 65.19% stale sources
summary_error_repair:  100.00% recall,  2.00 reads/query, 10.23 touch/event, 13.18% stale sources
```

The summary lane adds 336B for 128 claims in this diagnostic. It is therefore a
better answer-path primitive than whole-source repair, but it leaves provenance
freshness as a separate background CA problem.

Source-subtile repair measures that background provenance path. With 16 sources
split into four-source subtiles:

```text
claim_error_repair:      100.00% recall, 2.00 reads/query, 10.19 touch/event, 14.65% stale sources
subtile_error_repair:    100.00% recall, 2.00 reads/query,  7.02 touch/event, 42.97% stale sources
subtile_probe2_repair:   100.00% recall, 3.00 reads/query,  9.68 touch/event, 27.39% stale sources
subtile_probe4_repair:   100.00% recall, 5.00 reads/query, 12.27 touch/event, 19.48% stale sources
subtile_period4_repair:  100.00% recall, 2.00 reads/query,  7.30 touch/event, 40.62% stale sources
subtile_update_repair:   100.00% recall, 2.00 reads/query,  4.20 touch/event, 65.62% stale sources
```

The answer path remains fixed through the claim summary, so the metric is now
source freshness per local maintenance byte. That is the right chip-facing
objective for provenance lanes.

The learned source-subtile controller stores one policy id per importance mode:
three modes and six candidates need 1.125B. Checked evaluation:

```text
loose  target <=46% stale: subtile_error_repair, 6.81-7.03 touch/event, 0/4 failures
normal target <=31% stale: subtile_probe2_repair, 9.21-9.68 touch/event, 0/4 failures
strict target <=15% stale: claim_error_repair, 9.85-10.19 touch/event, 0/4 failures
```

This separates hardware policy state from wiki content state: a small metadata
mode bit steers provenance freshness without touching the summary answer path.

The metadata-derived importance proxy adds:

```text
metadata per claim: trust/citation/recency/query_freq = 8 bits
metadata classifier LUT: 64B
provenance repair LUT: 1.125B
total policy LUT state: 65.125B
held-out metadata streams: 0/3 target failures
estimated provenance touch: 8.49-8.58 cells/event
```

This is still synthetic metadata, but it turns the CA Wiki Cell controller into
the desired hardware shape: narrow per-claim metadata plus tiny LUTs, not a
large learned dense controller.

Noisy metadata audit with the same policy-state budget:

```text
noise_std: 0.85
classifier LUT: 64B
repair LUT: 1.125B
total LUT: 65.125B
held-out accuracy: 75.68-76.95%
strict recall: 94.02-96.47%
under-strict rate: 2.05-2.93%
over-strict rate: 20.90-21.97%
estimated provenance touch: 8.57-8.62 cells/event
target failures: 0/4 rows
```

The over-strict rate is intentional in this diagnostic: repair traffic is cheaper
than silently under-repairing high-importance pages.

## Unified Event Profile

The project now includes a unified per-event proxy that combines:

- exact sparse-memory local reads;
- compressed dense-context counter updates;
- online candidate-cache updates, admission-gate reads, and shortlist scoring
  reads;
- CSA/HCA context-summary state and local reads;
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
legacy local bytes/event: about 51.46 KB
wide64 CSA/HCA local bytes/event: about 52.10 KB
compact128 CSA/HCA local bytes/event: about 52.28 KB
rare128 CSA/HCA local bytes/event: about 52.28 KB
joint128 CSA/HCA local bytes/event: about 52.28 KB
retire128c4 CSA/HCA local bytes/event: about 52.28 KB
retire128c2 CSA/HCA local bytes/event: about 52.28 KB
retire128c3 CSA/HCA local bytes/event: about 52.28 KB
retire128c3g3 CSA/HCA local bytes/event: about 52.28 KB
Transformer KV read/token: about 384 MB
legacy on-chip HARC-CA state: about 183.8 KB
wide64 CSA/HCA on-chip HARC-CA state: about 707.8 KB
compact128 CSA/HCA on-chip HARC-CA state: about 451.8 KB
rare128 CSA/HCA on-chip HARC-CA state: about 354.6 KB
joint128 CSA/HCA on-chip HARC-CA state: about 356.9 KB
retire128c4 CSA/HCA on-chip HARC-CA state: about 401.8 KB
retire128c2 CSA/HCA on-chip HARC-CA state: about 383.8 KB
retire128c3 CSA/HCA on-chip HARC-CA state: about 392.8 KB
retire128c3g3 CSA/HCA on-chip HARC-CA state: about 392.8 KB
```

The retire128c3g3 CSA/HCA-aware profile adds about 832.5B/event over the legacy
profile:

```text
HCA lazy summary read: about 6B/event
HCA lazy summary update: about 12B/event
learned probe/fanout LUT reads: about 0.17B/event
counting Bloom sidecar read: about 0.38B/event
counting Bloom sidecar update: about 0.22B/event
CSA block-summary score reads: about 150B/event
CSA rare-directory read: about 0.50B/event
CSA selected token-cell reads: about 663.2B/event
```

The g3 fanout guard does not add SRAM state beyond the existing 42B fanout LUT,
and it does not increase the normal reference directory read path in this
profile. Its extra read cost is scenario-dependent: spread-out rare-token cases
pay the third directory entry, while strong reference HCA queries still skip it.

The wide64 baseline spends less selected-token traffic, about 648B/event total
context traffic, but it needs about 512KB of CSA block summaries. The
compact128 point uses about 256KB of block summaries. The current joint128 point
uses about 128KB of block summaries, about 30.8KB of rare-token directory state,
2.28KB of learned probe/fanout control metadata, and 12KB of lazy HCA summary
metadata/counters. The retire128 family adds the online `count1_retire15`
counting Bloom sidecar. The first 4-bit-counter version raises state to about
401.8KB. The normal-stream 2-bit version lowers that to about 383.8KB, but the
adversarial-collision robust 3-bit version with the g3 fanout guard is now the
current budget point at about 392.8KB.

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

## CA Wiki Cell Trace Controller

The current LLM-Wiki-oriented CA Wiki Cell controller uses:

- 8 bits/claim of static trust/citation/recency/query metadata;
- a 2-bit local pressure bucket derived from query/update/stale-probe counters;
- a 1.00B pressure-to-importance classifier LUT;
- the existing 1.125B importance-to-provenance-repair LUT.

On three 1024-claim held-out traces with 4096 query events and 1024 update
events, the pressure controller selects strict mode for 31.25-32.62% of claims,
keeps strict recall at 100.00%, and has 0.00% under-strict rate. Estimated
provenance repair traffic is 8.44-8.54 touched cells/event. The classifier LUT
is small because the expensive information is not in the table; it is in the
local counters that the CA fabric maintains as events arrive.

The compiled-trace controller replaces the pressure bucket with three local
audit buckets:

- retrieval-error count bucket;
- contradiction-probe count bucket;
- stale-source-probe count bucket.

Those three 2-bit buckets index a 16.00B classifier LUT. With the same 1.125B
provenance repair table, the total controller LUT state is 17.125B. On four
1024-claim held-out traces with 4096 query events, 2048 update events, and 512
compile events, strict recall is 99.19-100.00%, under-strict rate is
4.00-6.05%, and estimated provenance repair traffic is 8.36-8.40 touched
cells/event. The relevant chip interpretation is that the CA fabric stores and
updates the counters beside the claim cell; global model weights are not needed
to decide when a compiled wiki page needs stricter provenance repair.

The text-source version keeps the same hardware-facing budget while changing
the label source. Raw source cells are controlled text snippets, compiled claim
summaries are markdown lines, and a lightweight parser feeds the same three
2-bit local audit counters. The classifier LUT remains 16.00B and the total
controller LUT state remains 17.125B. On four 1024-claim held-out text traces,
strict recall is 99.46-100.00%, under-strict rate is 4.39-5.37%, and estimated
provenance traffic is 8.39-8.42 touched cells/event. This does not yet price
the parser or text compiler hardware/software path; it prices the CA-side
state and policy decision once text edits have been compiled into local audit
counters.

The parser-noise audit keeps the classifier at 16.00B and the total controller
LUT at 17.125B, but changes the observed counters. Labels come from the clean
text state; inference counters come from a parser with 6% status misread and 3%
drop probability. The strict-recall gate still passes at 97.63-98.32%, and
under-strict rate is 3.71-4.69%. The traffic cost is the important number:
strict mode rises to 45.41-48.93% of claims, over-strict rate is
26.66-27.83%, and provenance repair traffic rises to 8.89-8.97 touched
cells/event. The next hardware feature to justify is not a larger importance
LUT; it is a tiny parser-miss or confidence counter that can suppress false
strict repair decisions.

The first guard tradeoff measures that counter. Three controller options are
now in the parser-noise audit:

- 16B baseline 3D LUT over observed retrieval-error, contradiction, and stale
  buckets;
- 16B baseline plus a 32B one-bit 4D safe downgrade guard;
- 64B 4D LUT that adds parser-miss bucket directly.

The safe guard keeps the same safety profile but only lowers mean over-strict
from 27.08% to 26.86%. The 64B miss-aware LUT is more meaningful: mean
over-strict falls to 24.17%, mean strict recall is 98.05%, mean under-strict is
4.49%, and mean provenance traffic falls from 8.92 to 8.84 touched cells/event.
The hardware implication is simple: the first parser-confidence feature is
worth at most tens of bytes of control SRAM in this toy setup, not a dense
model path.

The multi-field text controller keeps the control SRAM in the same range while
making the input more realistic. Four 2-bit local buckets are used:

- weighted observed retrieval error across status/priority/region/owner;
- core-field conflict for status and priority;
- weighted stale/source-disagreement score;
- parser-miss count.

This is a 64.00B classifier LUT plus the existing 1.125B provenance repair
table. With 6% misread and 3% drop parser noise, held-out strict recall is
97.57-98.96%, under-strict rate is 1.27-1.95%, and estimated provenance traffic
is 9.08-9.10 touched cells/event. Compared with the single-field parser-noise
controller, the multi-field signal spends more strict repair traffic because
there are more ways for a source to be partially stale, but under-strict risk is
lower and the controller has a clearer hardware interpretation.

Paragraph-style source/wiki cells keep the same 64.00B classifier and 1.125B
provenance LUT, but make the text side less idealized. Source paragraphs can
omit fields and include historical distractor sentences; the parser only emits
current status/priority/region/owner values plus misses. The default paragraph
controller uses a 10.00 under-estimation loss weight and 0.75 over-strict loss
weight. On four held-out paragraph traces, strict recall is 99.00-99.80% and
under-strict rate is 0.49-0.88%, but over-strict rate rises to 26.95-28.03% and
estimated provenance traffic rises to 9.35-9.40 touched cells/event. The chip
cost is therefore not a larger controller table; it is extra local repair
traffic caused by weak paragraph coverage/confidence signals.

The first coverage-confidence variant prices one such signal. Adding a 2-bit
weighted field-coverage gap bucket gives two hardware options:

```text
baseline_4d:     64B controller, 27.49% mean over-strict, 9.37 touch/event, 0/4 failures
coverage_guard: 192B controller, 26.93% mean over-strict, 9.36 touch/event, 0/4 failures
coverage_lut5d: 256B controller, 26.93% mean over-strict, 9.35 touch/event, 0/4 failures
```

This is a weak but useful positive result. The extra 128-192B of controller
state does not buy a large traffic reduction, so the next hardware metric
should split field coverage into separate summary-core and source-core
confidence counters before spending more LUT dimensions.

The split-confidence metric is stronger. It separates the extra signal into
summary-core coverage, source-core coverage, and source/summary agreement:

```text
baseline_4d:     64B controller, 27.49% mean over-strict, 9.37 touch/event, 0/4 failures
coverage_lut5d: 256B controller, 26.93% mean over-strict, 9.35 touch/event, 0/4 failures
split_guard7d:  2.06KB controller, 25.73% mean over-strict, 9.34 touch/event, 0/4 failures
factor_vote56b: 120B controller, 24.58% mean over-strict, 9.35 touch/event, 0/4 failures
factor_vote80b: 144B controller, 23.83% mean over-strict, 9.32 touch/event, 0/4 failures
factor_vote80b_covsafe: 144B controller, 24.22% mean over-strict, 9.32 touch/event, 0/4 failures
factor_vote80b_shiftguard: 144B controller, 24.29% mean over-strict, 9.33 touch/event, 0/4 failures
learned_shift_selector: 204B controller, 28.44% mean over-strict, 9.40 touch/event, 0/4 failures
two_branch_selector: 234B controller, 27.25% mean over-strict, 9.38 touch/event, 1/4 failures
two_branch_factor_selector: 174B controller, 26.81% mean over-strict, 9.37 touch/event, 0/4 failures
two_branch_mixer_selector: 294B controller, 24.56% mean over-strict, 9.33 touch/event, 1/4 failures
split_lut7d:    4.00KB controller, 26.39% mean over-strict, 9.35 touch/event, 1/4 failures
```

The direct 7D LUT is not the right default despite being larger. The safer
hardware primitive is a small conservative classifier plus a confidence guard
that only downgrades strict repair when local confidence buckets have enough
supporting training examples. The first factorized version is better than the
full 7D guard on this synthetic paragraph workload: `factor_vote80b` uses only
144B total control state, keeps strict recall at 98.32%, and lowers mean
over-strict to 23.83%.

The stress metric keeps the same training distribution and shifts eval only
over two held-out seeds per scenario:

```text
factor_vote80b default_1k:     75.34% accuracy, 98.18% strict recall, 1.17% under, 23.49% over, 2/2 pass
factor_vote80b parser_x2:      66.60% accuracy, 98.47% strict recall, 0.83% under, 32.57% over, 2/2 pass
factor_vote80b omit_x2:        76.37% accuracy, 97.01% strict recall, 2.15% under, 21.48% over, 0/2 pass
factor_vote80b distractor_x2:  73.24% accuracy, 98.21% strict recall, 1.32% under, 25.44% over, 1/2 pass
factor_vote80b large_2k:       73.85% accuracy, 98.12% strict recall, 1.29% under, 24.85% over, 1/2 pass
factor_vote80b_shiftguard default_1k:    74.85% accuracy, 98.38% strict recall, 1.07% under, 24.07% over, 2/2 pass
factor_vote80b_shiftguard parser_x2:     66.06% accuracy, 98.57% strict recall, 0.78% under, 33.15% over, 1/2 pass
factor_vote80b_shiftguard omit_x2:       75.54% accuracy, 97.40% strict recall, 1.95% under, 22.51% over, 0/2 pass
factor_vote80b_shiftguard distractor_x2: 73.14% accuracy, 98.42% strict recall, 1.22% under, 25.63% over, 1/2 pass
factor_vote80b_shiftguard large_2k:      73.36% accuracy, 98.17% strict recall, 1.27% under, 25.37% over, 1/2 pass
learned_shift_selector default_1k:       71.29% accuracy, 99.09% strict recall, 0.68% under, 28.03% over, 2/2 pass
learned_shift_selector parser_x2:        63.57% accuracy, 99.39% strict recall, 0.39% under, 36.04% over, 0/2 pass
learned_shift_selector omit_x2:          72.80% accuracy, 98.46% strict recall, 0.93% under, 26.27% over, 2/2 pass
learned_shift_selector distractor_x2:    70.07% accuracy, 99.47% strict recall, 0.49% under, 29.44% over, 2/2 pass
learned_shift_selector large_2k:         70.07% accuracy, 98.74% strict recall, 0.93% under, 29.00% over, 2/2 pass
two_branch_factor_selector default_1k:   73.34% accuracy, 99.40% strict recall, 0.59% under, 26.07% over, 2/2 pass
two_branch_factor_selector parser_x2:    63.92% accuracy, 99.49% strict recall, 0.34% under, 35.74% over, 0/2 pass
two_branch_factor_selector omit_x2:      73.58% accuracy, 99.32% strict recall, 0.98% under, 25.44% over, 2/2 pass
two_branch_factor_selector distractor_x2: 70.65% accuracy, 99.59% strict recall, 0.68% under, 28.66% over, 2/2 pass
two_branch_factor_selector large_2k:     71.58% accuracy, 99.11% strict recall, 0.83% under, 27.59% over, 2/2 pass
two_branch_mixer_selector default_1k:    74.66% accuracy, 98.28% strict recall, 1.07% under, 24.27% over, 1/2 pass
two_branch_mixer_selector parser_x2:     65.82% accuracy, 98.57% strict recall, 0.78% under, 33.40% over, 1/2 pass
two_branch_mixer_selector omit_x2:       75.44% accuracy, 97.49% strict recall, 1.71% under, 22.85% over, 0/2 pass
two_branch_mixer_selector distractor_x2: 73.00% accuracy, 98.42% strict recall, 1.17% under, 25.83% over, 1/2 pass
two_branch_mixer_selector large_2k:      73.14% accuracy, 98.12% strict recall, 1.27% under, 25.59% over, 1/2 pass
```

This separates efficiency from robustness. The 144B guard is efficient and
survives parser-noise shift, but field-coverage shift can make it downgrade too
many strict claims. The hand-coded shiftguard does not solve that matrix; it
mainly shows that manual core-gap/parser-miss thresholds are too brittle. A
multi-distribution learned selector solves the coverage-shift rows but stays
too strict under parser-noise shift. A chip-facing version should therefore use
a two-branch teacher: coverage uncertainty raises repair, parser uncertainty
suppresses false strict repair.

The first two-branch diagnostics narrow that requirement. A factor-first
coverage repair branch uses only 174B and passes default plus the coverage-side
stress rows, but parser_x2 remains 0/2. A local factor/learned mixer with 294B
state still fails to choose the right branch consistently. The next hardware
metric should therefore track a tiny rolling regime counter per region or tile:
parser-miss excess should select the factor branch, while coverage-gap excess
should select the repair branch.

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
bytes/cycle/tile. With the retire128c3g3 CSA/HCA-aware 392.8KB HARC-CA state and
52.28KB local bytes/event, a 32-tile configuration still fits:

```text
32 tiles: 512KB SRAM, 76.7% state utilization, 25 state tiles required, 5.2% bandwidth utilization
64 tiles: 1MB SRAM, 38.4% state utilization, 25 state tiles required, 2.6% bandwidth utilization
128 tiles: 2MB SRAM, 19.2% state utilization, 25 state tiles required, 1.3% bandwidth utilization
```

These are design-budget numbers. They do not prove timing, routing, area, yield,
or energy.
