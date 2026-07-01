# CA Wiki-Memory Track

This track asks whether the first strong HARC-CA product should be an external
knowledge fabric rather than a full Transformer replacement. The working answer
is yes: CA looks better suited to mutable, local, versioned knowledge storage
than to replacing every part of dense language modeling in one step.

## External Anchor

RAG framed the core problem clearly in 2020: parametric LMs store factual
knowledge in weights, but precise manipulation, provenance, and world-knowledge
updates remain difficult, so explicit non-parametric memory is useful for
knowledge-intensive tasks:

https://arxiv.org/abs/2005.11401

Recent LLM-Wiki work moves beyond flat embedding lookup. `Retrieval as
Reasoning` describes LLM-Wiki as a system that compiles documents into
structured pages with bidirectional links, supports search/read/link-following,
and keeps an Error Book for persistent structural and semantic correction:

https://arxiv.org/abs/2605.25480

WiCER identifies the compilation gap: blindly distilling raw documents into a
wiki can drop critical facts, while targeted diagnostic probes and refinement
recover much of the lost quality:

https://arxiv.org/abs/2605.07068

`Memory as Metabolism` reports that a cluster of personal wiki-style memory
architectures appeared around April 2026, including a Karpathy design proposal,
and frames long-term memory as a system that must triage, decay, contextualize,
consolidate, and audit:

https://arxiv.org/abs/2604.12034

## CA Fit

Wiki-memory has the shape CA wants:

- storage is spatial: pages, facts, links, summaries, and error records map to
  cells and local neighborhoods;
- updates are local: a changed fact should dirty nearby summaries and links, not
  rewrite a model;
- retrieval is routed: a query should follow summary gradients and links, not
  scan every page;
- provenance is explicit: returned evidence can carry page ids, versions, and
  contradiction markers;
- low-bit state is acceptable: counters, dirty bits, link strengths, recency,
  and confidence can be small integers or tiny LUT states.

This does not prove CA can replace a frontier decoder. It does suggest a better
near-term split: let a conventional or small neural decoder handle fluent text,
while CA hardware handles mutable world state, routing, evidence retrieval, and
local update propagation.

## Proposed Fabric

The first CA wiki-memory fabric has five lanes:

1. Page cells store compact page ids, topic hashes, recency, confidence, and
   version counters.
2. Fact cells store key/value fragments, source ids, and contradiction tags.
3. Link cells store bidirectional edges between pages/facts with low-bit
   strength and last-used counters.
4. Summary cells keep count-min or max-style sketches over descendant pages and
   facts.
5. Error-book cells preserve failed probes and force later refresh or
   consolidation.

Query flow:

```text
query hash -> summary route -> candidate pages -> linked facts -> evidence set
```

Update flow:

```text
new fact -> local insert -> dirty page/link summaries -> triggered refresh
        -> optional contradiction buffer -> audit/consolidate
```

The triggered group-summary result already gives a small prototype of the
control rule: dirty-count plus age can decide when local summaries refresh
without a global sweep.

## First Benchmark

Build a synthetic wiki with pages, links, mutable facts, and multi-hop queries.
Measure:

- single-hop and multi-hop answer recall;
- update latency after fact edits;
- stale-answer rate before and after triggered refresh;
- cells read per query;
- cells written per update;
- provenance precision;
- contradiction recovery after repeated error-book probes.

Baselines:

- flat dense-vector RAG over chunks;
- graph/wiki retrieval with CPU-style global search;
- full-context attention over the same pages when small enough to run.

## First Prototype Result

The first NumPy prototype is `experiments/wiki_memory_demo.py`. It builds a
256-page synthetic wiki with four facts per page, four links per page, 16-page
groups, and 4x256x4-bit page/group summaries. It now adds 32 contradiction
clusters with three source pages each. Queries mix single-hop fact reads, two-hop
link reads, and replicated-claim probes. Updates change the source-of-truth
first, then mark local pages and groups dirty; refresh or error-book repair is
required before routed memory cells see the new key, revised value, or
multi-source claim update. Half of non-cluster updates are value revisions, 30%
of updates hit contradiction clusters, and 25% of eligible queries replay an
error-book probe.

The exact-update baseline refreshes page and group summaries after every fact
edit. It reaches 100.0% recall, but writes about 20,255 score-equivalent cells
per update. The conservative triggered policy, `trigger16_age16`, refreshes
when 16 pages are dirty or the summary is 16 update steps old. On the same
event stream it reaches 94.73% overall recall, 92.08% recent-update recall, and
90.59% replicated-claim recall while writing about 14,466 cells/update. Adding
page-local error-book repair raises overall recall to 97.66%, repeated
failed-probe recall to 98.54%, and replicated-claim recall to 95.37%, at about
14,739 cells/update. Adding cluster repair makes every checked source in the
claim cluster consistent (`clu_ok=100.0%`) at about 14,914 cells/update, versus
93.06% for page-local repair. Reads stay about 356-357 cells/query versus 1,024
for a flat exact page-fact scan, a 65% read reduction. With no refresh, recall
drops to 50.39% and stale misses rise to 49.61%.

The flat/RAG-style page-summary baseline uses the same summaries and update
policies, but scans every page summary before reading selected exact facts. It
matches the exact-update and clusterbook accuracy points, but costs about 1,061
cells/query. The hierarchical CA route therefore cuts query reads by about
66.3% at the same write policy and accuracy. On this small four-facts/page
benchmark, flat page-summary scan is even slightly more expensive than scanning
all exact fact cells; the important point is that its read path grows with every
page, while the CA route spends reads on group summaries plus selected groups.

This is not yet a learned memory system, but it establishes the first measurable
wiki-memory claim: local dirty/age summary refresh can keep mutable facts mostly
queryable while avoiding full-wiki scans. The error-book repair path now has a
real workload: page repair improves answer recall, while cluster repair enforces
multi-source consistency across replicated claims.

The scaling sweep then holds the clusterbook policy fixed and increases the wiki
from 256 to 2,048 pages. Accuracy stays matched between hierarchical CA routing
and flat page-summary scan because they use the same summaries and repair
policy. Read cost diverges: at 256 pages CA reads about 357 cells/query versus
1,061 for flat scan; at 2,048 pages CA reads about 804 cells/query versus 8,228
for flat scan. The CA read reduction versus flat scan grows from 66.3% to
90.2%, while exact fact scan grows to 8,192 cells/query. This is the first
evidence that the wiki-memory route has the scaling shape we want.

The density sweep is the first hard warning. Holding pages at 1,024 and
increasing facts/page from 4 to 32, the current four-group CA route loses
accuracy under page-summary collision pressure. With 4x256x4-bit summaries,
CA recall falls from 98.83% at four facts/page to 77.93%, 30.47%, and 19.92%
at 8, 16, and 32 facts/page. Flat page-summary scan stays near 99.8% at width
256 because it can rank every page summary globally, though it pays about
4,132-4,378 cells/query. At width 128, both collision pressure and CA group
selection hurt: CA recall is already 72.46% at four facts/page, and flat scan
falls to 59.96% at 32 facts/page. The conclusion is architectural rather than
negative: dense pages need adaptive group fanout, wider/multi-feature summaries,
or a page-internal second stage.

The first adaptive group-fanout sweep validates that diagnosis. On the 1,024
page, 16 facts/page, width-256 stress case, fixed `selected_groups=4` reaches
only 30.47% CA recall at about 644 cells/query. Raising fixed fanout to 32
restores 99.80% recall but costs about 2,445 cells/query. Adaptive fanout starts
from four groups, expands on near-tied group-summary scores, caps at 32 groups,
and with margin 1 reaches the same 99.80% recall at about 1,991 cells/query.
That is 53.0% fewer reads than the flat page-summary scan and less traffic than
fixed 32-group routing. The next step is to learn this fanout decision from
local low-bit metadata rather than hand-setting the margin.

The first learned fanout LUT now replaces that hand margin with a small local
table. It trains from minimal-route self-supervision across 32,737 query states:
for each low-bit feature bucket, choose the smallest group fanout that reaches
the target route coverage. The conservative `learned_lut_t100` point uses about
1.1KB of table state and reaches the same 99.80% recall as flat scan, but reads
only about 1,566 cells/query. That is a 63.0% read reduction versus flat
page-summary scan and a further reduction versus the hand adaptive 1,991
cells/query point.

The learned fanout grid tests whether that is a single-point accident. At
8 facts/page, learned fanout matches flat recall at 512, 1,024, and 2,048 pages
while reading about 459, 604, and 996 cells/query, versus flat reads of about
2,120, 4,168, and 8,263. At 16 facts/page, learned fanout still matches flat
at 512 and 1,024 pages, but at 2,048 pages it matches the hand adaptive route
rather than flat because the 32-group cap misses too many candidate pages. At
32 facts/page, the limit is explicit: 512 pages can recover flat recall only by
nearly degenerating into a full group scan, while 1,024 and 2,048 pages stay far
below flat recall. The next architectural step is therefore not just a better
fanout table; dense pages need stronger summaries or a page-internal second
stage.

The dense routing-tile sweep is the first fix for that boundary. Instead of
keeping 16 pages per group, it uses four-page routing tiles and lets the learned
fanout LUT read up to 48 tiles. On the 1,024-page, 32 facts/page case, recall
returns from 59.38% to 99.80% while reads fall from the old learned route's
2,581 cells/query to 1,697 cells/query; flat scan reads 4,378 cells/query. On
2,048 pages and 32 facts/page, dense tiles reach 99.22% recall and 2,897
cells/query versus flat's 95.12% recall and 8,474 cells/query. The cost is
modest SRAM: about 96.6KB extra state at 1,024 pages, 192.6KB at 2,048 pages,
plus a 1.69KB fanout LUT.

The density-aware tile sweep adds the missing policy guard. It models a
2,048-page wiki split into sparse 8 facts/page regions and dense 32 facts/page
regions. A local quality probe enables four-page tiles only when they do not
lower dense-region recall. With 25% dense pages, the guard leaves the dense
region on the 16-page tile because the small region does not benefit; recall
stays 99.02% and the only extra state is a 256B density tag table. With 50% and
75% dense pages, the guard enables dense tiles, raising recall from 79.00% and
64.60% to 99.22% and 99.32%. Reads are about 1,159 and 1,843 cells/query,
versus flat's 4,281 and 5,360 cells/query. This is the first density-aware CA
memory policy rather than a global geometry setting.

The refresh-derived density tag sweep replaces the region oracle with a tag
that can be emitted during normal summary refresh. With 2-bit tags and an
8 facts/tag step, sparse 8 facts/page regions get tag 1 and dense 32 facts/page
regions get tag 3. That is enough to identify dense regions, but not enough to
choose geometry by itself: at 25% dense pages, density-only thresholding enables
four-page tiles and drops recall from 99.02% to 97.71%. The guard is now a
low-bit paired online counter: during a 128-query / 64-update probe window it
presents the same queries to the baseline and dense-tile routes, saturates
dense wins and dense losses into 4-bit counters, and enables dense tiles only
when `c_win >= 3` and `c_loss == 0`. At 25% dense pages the counter stays
`0/0` and is rejected, keeping 99.02% recall. At 50% and 75% dense pages the
raw probe sees 20 and 42 dense wins with zero losses; both saturate to
`c_win=15, c_loss=0`, so the same tag plus guard enables dense tiles and
recovers 99.22% and 99.32% recall while cutting flat reads by 72.94% and
65.62%. The counter state is only 128B for 2,048 pages when stored per
16-page guard block.

The mixed-stream counter diagnostic removes the separate sparse/dense probe
windows. One 512-query / 256-update event stream feeds both regions and updates
two 4-bit counters per guard block. The locality sweep tests 256, 512, and
1,024-page blocks with same-tag sharing radii 0, 1, and 2. Sparse false-enable
stays 0.00% for every tested setting. At 50% dense pages, local counters enable
2/4, 1/2, and 1/1 dense blocks for 256, 512, and 1,024-page blocks. Same-tag
radius-1 sharing lifts the 512-page case to 2/2, while the finer 256-page case
needs radius 2 to reach 4/4. At 75% dense, all tested block sizes are already
fully enabled locally. A 50% dense observation-window stress keeps shared
coverage at 2/2 and sparse false-enable at 0.00% from 128/64 through
1,024/512 query/update windows. Update-noise stress also holds: raising
revision updates to 80%, cluster updates to 60%, or both together keeps shared
dense coverage at 2/2 and sparse false-enable at 0.00%. The hardware lesson is
now sharper: density tags should gate short-range counter sharing, not just
local tile choice.

The learned guard LUT now chooses the full low-bit guard controller. Training
on the 25%, 50%, and 75% dense mixed streams with block sizes 256, 512, and
1,024 learns `256 -> radius 2/decay win/loss 0/dwin +1`,
`512 -> radius 1/decay win/loss 0/dwin +1`, and
`1024 -> radius 0/decay win/loss 0/dwin +1`. The table is 2.625B for these
three geometry entries: radius bits, one decay-mode code, one loss-threshold
bit, and one win-threshold-delta code. Training-seed evaluation hits the
target: 25% dense stays off, 50% dense rises from 50% local dense-block
coverage to 100% for 256 and 512-page blocks, 75% dense remains 100%, and
sparse false-enable stays 0.00%. The held-out seed audit that used to fail is
now repaired without permanent tolerance: seed 1501 has 99/1 dense wins/losses
at 75% dense, and decay-on-win with strict `loss == 0` restores 100% learned
dense coverage for 256, 512, and 1,024-page blocks with sparse false-enable
still 0.00%. The current controller is now a learned local rule over sharing
radius, event-driven loss decay, loss tolerance, and win threshold.

The held-out loss-tolerance audit isolates the 512-page/radius-1 geometry
and compares strict `loss=0` against tolerant `loss=1` on seeds 1201, 1301,
1401, and 1501. The strict gate has one dense-on failure: seed 1501 at 75%
dense drops to 0.00% shared dense coverage on a 99/1 dense wins/losses trace.
The tolerant gate repairs that row to 100.00%, raises mean dense-on coverage
from 87.50% to 100.00%, keeps 25% dense off, and keeps max sparse shared
false-enable at 0.00%. A high-update-noise check on seed 1501 with revision
updates at 80% and cluster updates at 60% keeps both `loss=0` and `loss=1` at
100.00% dense-on coverage with 0.00% sparse false-enable. This is still not a
full stability proof, but it turns the previous failure into a named regression
test.

The first loss-decay variant keeps the strict `loss=0` gate and changes only
the local counter dynamics. In the seed1501 99/1 dense wins/losses case, the
old `none` mode leaves the final dense max counter at `15/1` and shared dense
coverage at 0/3. The `win` mode decrements the local loss counter when a later
dense-route win arrives in the same guard block; `nonloss` decrements it on any
later non-loss query. Both modes turn the final dense max counter into `15/0`,
restore 3/3 shared dense coverage, and keep sparse shared false-enable at
0.00%. The 25% dense off row stays 0/1 for all modes. This is a cleaner CA-chip
rule than simply tolerating one loss forever because it lets later local
evidence repair a stale loss counter.

The follow-up noise-matrix audit keeps the same 512-page/radius-1 geometry but
shrinks the rows to 25% dense off and 75% dense on so it can sweep update-noise
regimes. Across seeds 1501 and 1601 under base, revision-80%, cluster-60%, and
combined revision-80%/cluster-60% regimes, strict `loss=0` has two dense-on
failures and 75.00% mean dense-on coverage. Tolerant `loss=1` repairs both,
reaches 100.00% mean dense-on coverage, keeps off-region enables at zero, and
keeps sparse shared false-enable at 0.00% for every audited row. The next audit
should randomize more seeds and noise rates, but the current matrix supports
the claim that a one-count loss tolerance is not merely overfitting one seed.

A deterministic pseudo-random noise smoke test adds four more held-out cases:
seeds 1701, 1801, 1901, and 2001 with revision rates from 33% to 51% and
cluster rates from 20% to 50%. This sample did not reproduce the strict
zero-loss dense-loss failure: both `loss=0` and `loss=1` reach 100.00% dense-on
coverage, zero off-region enables, and 0.00% sparse shared false-enable. The
value of this test is different from the seed1501 regression: it says the
tolerant gate did not introduce measured false-enable on randomized update
rates, while a larger randomized sweep is still needed to estimate rare failure
probability.

CA Wiki Cell v0 now makes the LLM-Wiki idea more literal. Instead of only
measuring page-summary routing, it stores each mutable claim across eight
source-page cells, with local source links and one 4-bit error-book counter per
claim. An update writes one source page first; a CA repair policy then spreads
the newer revision through local source links. In the default 128-claim,
1,024-query, 256-update diagnostic, sparse reads without repair reach only
38.38% recall because most recent queries miss the newest source. Flat scan
over all eight source pages reaches 100.00% recall, but query work is
8.00 source cells/query and the memory remains internally stale: only 16.41%
of claims are fully consistent at the end, with 73.14% stale source cells.
`tile_update_ca` uses one tile-local update pulse, keeps query reads at
2.00 source cells/query, reaches 100.00% recall, and leaves zero stale source
cells. The cost is visible: total local touch is 16.00 cells/event versus
7.24 for flat scan, because repair traffic has moved from query time into
background local maintenance. The lazy `error_book_ca` point is cheaper than
full tile repair but reaches only 83.59% recall; the hybrid error-book point
reaches 96.00%. The honest conclusion is that CA is already a better shape for
low-latency mutable reads, but repair scheduling must be learned before claiming
a total traffic win.

The first learned repair scheduler turns that conclusion into a tiny LUT. The
candidate bank contains 28 low-bit schedules over source read count, local
radius, update-repair ticks, update-repair period, and error-book repair ticks.
The LUT is indexed by fan-in and update-pressure buckets: four, eight, and
16 sources/claim crossed with 128 and 256 updates. With targets of at least
90.00% overall recall, 85.00% recent recall, and at most 10.00% stale source
cells, the table is 3.75B. The learned choices are conservative for hard
buckets and lazy for easy ones: eight-source/128-update chooses pure
error-book repair (`ca_r4_u0p1_e1`), while eight-source/256-update chooses
periodic repair (`ca_r4_u1p2_e1`) and keeps query reads at 2.00 cells/query
with about 92% recall and about 14.8-15.2 cells/event on held-out seeds. That
is lower maintenance traffic than the fixed 16.00 cells/event `tile_update_ca`
policy, but it is a quality-budgeted tradeoff rather than a strict replacement.
The held-out evaluation has 2 failures out of 24 rows: one four-source row
slightly exceeds the stale-source target, and one 16-source/128-update row
falls just below the recall/recent targets. This marks the next boundary:
large fan-in needs a second-level source tile or a learned radius/tick policy
with stronger local evidence.

The strict/budget comparison makes that boundary product-shaped. Running the
same candidate bank twice gives two policy tables: strict mode targets
99.00% recall, 98.00% recent recall, and at most 1.00% stale source cells;
budget mode keeps the 90.00% / 85.00% / 10.00% target. Each table is 3.75B, so
storing both policy ids costs 7.50B in this diagnostic. Strict mode has zero
target failures across 24 evaluation rows and mean local touch of 22.22
cells/event. Budget mode has two target failures and mean touch of 20.12
cells/event, a 9.47% maintenance-traffic reduction. The largest savings appear
where fan-in is wide: 16-source/128-update saves 12.40%, and
16-source/256-update saves 10.42%. This is a clearer chip knob than one fixed
policy: high-confidence memory pages can use strict repair, while ordinary
mutable wiki pages can use budget repair.

The second-level claim-summary diagnostic attacks the 16-source traffic problem
directly. For 128 claims with 16 sources/claim and 256 updates, flat source scan
gets 100.00% answer recall at 16.00 reads/query and 13.63 touch/event, but
leaves 79.83% source cells stale. Strict source repair keeps query reads at
2.00 and source staleness at 0.00%, but costs 62.40 touch/event. Adding one
low-bit summary cell per claim costs only 336B of extra state. `summary_only`
then reaches 100.00% answer recall with 1.00 read/query and 1.20 touch/event,
because updates write both the edited source and the claim summary. This is not
a complete provenance repair: source staleness remains 79.83%. The
`summary_error_repair` point keeps answer recall at 100.00%, uses 2.00
reads/query, costs 10.23 touch/event, and lowers source staleness to 13.18%.
This is the cleaner architecture for high fan-in claims: a summary lane answers
fast, while background CA repair handles source freshness according to the
strict/budget mode bit.

Source-subtile repair is now measured behind that summary lane. With 16
sources/claim split into four-source subtiles, all tested policies keep
100.00% answer recall because the claim summary remains the answer path.
Whole-claim error repair (`claim_error_repair`) uses one source probe and
refreshes all 16 sources when the probe finds staleness; it costs
10.19 touch/event and leaves 14.65% stale sources. One-probe subtile repair
refreshes only the touched four-source tile, dropping cost to
7.02 touch/event but leaving 42.97% stale sources. Two probes improve
freshness to 27.39% stale sources at 9.68 touch/event, and four probes reach
19.48% stale sources at 12.27 touch/event. This is the right shape for
provenance freshness: a local knob over probe count and tile repair scope,
not a forced whole-claim repair. It also shows the limit: once provenance needs
near-whole-claim freshness, claim-level repair is still competitive.

The first learned provenance controller maps page importance to that knob. It
uses a 1.125B LUT over three importance modes and six candidate policies.
`loose` pages target at most 46.00% stale sources and choose
`subtile_error_repair`; `normal` pages target 31.00% and choose
`subtile_probe2_repair`; `strict` pages target 15.00% and choose
`claim_error_repair`. Across train/eval seeds 2601, 2701, 2801, and 2901, all
12 rows meet their source-freshness targets. Mean behavior follows the intended
hardware mode split: loose costs about 6.8-7.0 touch/event, normal costs about
9.2-9.7, and strict costs about 9.9-10.2. This turns provenance freshness into
a small local policy output rather than a globally fixed repair rule.

The importance mode is now derived from local metadata in a synthetic proxy.
Each claim gets four 2-bit buckets: trust, citation density, recency, and query
frequency. A 64B classifier LUT maps the 256 metadata buckets to
`loose/normal/strict`; the existing learned provenance LUT adds 1.125B. On
three 512-claim held-out metadata streams, the proxy reaches 100.00%
classification accuracy and 100.00% strict recall against the deterministic
teacher rule, with estimated provenance touch around 8.49-8.58 cells/event.
This is not a real wiki-data result, but it completes the CA control chain:
local metadata -> importance mode -> provenance repair policy -> source
freshness budget.

A noisy-label audit now replaces the deterministic teacher with sampled
importance labels. The classifier table is still 64B and the provenance repair
table is still 1.125B, but the training loss penalizes under-estimating
importance more than over-strict repair. Held-out accuracy falls to
75.68-76.95%, which is expected under label noise. The chip-facing gates still
pass: strict recall is 94.02-96.47%, under-strict rate is 2.05-2.93%, and
estimated provenance touch is 8.57-8.62 cells/event. This is a more honest
result than the deterministic proxy: the tiny metadata LUT is not a perfect
classifier, but it can be biased toward protecting high-importance pages.

The Karpathy-style LLM-Wiki pivot makes the next audit more concrete: page
importance should come from update/query history, not only static metadata. A
metadata-only trace classifier was tried first and failed honestly: the 64B
table could not predict stochastic stale-probe pressure from trust/citation/
recency/query priors alone. The current diagnostic therefore promotes a more
CA-native state variable: a 2-bit local pressure bucket derived from
query-count, update-count, stale-probe count, trust, and citation. The
pressure-bucket classifier is only 1.00B, and the existing provenance repair
table is 1.125B. On three 1024-claim held-out traces with 4096 query events and
1024 update events, it reaches 100.00% accuracy and 100.00% strict recall with
0.00% under-strict rate. The strict mode rate is 31.25-32.62%, and estimated
provenance touch is 8.44-8.54 cells/event. This is not a solved real-wiki
importance model; it is a clean CA control primitive: local event counters ->
local pressure bucket -> tiny importance LUT -> local provenance repair mode.

The compiled-trace audit moves one step closer to an LLM-Wiki memory engine.
It simulates raw source cells, a compiled claim summary, source updates,
summary recompilation, and query-time audit probes. Importance labels now come
from actual local failures in that compiled wiki: retrieval errors when the
summary lags the current truth, contradiction probes when audited source cells
disagree, and stale-source probes when sources disagree with the compiled
summary. Inference uses only three 2-bit local buckets for those counters. The
classifier LUT is 16.00B and the reused provenance repair LUT is 1.125B. On
four held-out traces with 4096 queries, 2048 updates, and 512 compile events,
accuracy is 85.35-86.82%, strict recall is 99.19-100.00%, under-strict rate is
4.00-6.05%, and estimated provenance touch is 8.36-8.40 cells/event. This is
the strongest evidence so far that the CA-first chip target should start as a
mutable compiled-wiki maintenance fabric rather than a full decoder
replacement.

The text-source trace audit replaces the integer claim values with controlled
source snippets and compiled markdown summaries. Raw source cells now contain
status text fields, source edits rewrite those snippets, and the compiled
summary is a markdown sentence that can lag the raw sources. A lightweight
parser extracts the status field and feeds the same three local counters:
retrieval errors, contradiction probes, and stale-source probes. The chip-side
controller is unchanged: three 2-bit buckets index a 16.00B LUT, then the
1.125B provenance repair table selects the local repair policy. On four
held-out text traces, accuracy is 86.62-87.89%, strict recall is
99.46-100.00%, under-strict rate is 4.39-5.37%, and estimated provenance touch
is 8.39-8.42 cells/event. This is still not open-ended language understanding,
but it moves the evidence from numeric facts to compiled textual wiki cells.

The parser-noise audit keeps the same text-source setup but separates clean
teacher labels from noisy parser-observed counters. With a 6% status misread
rate and a 3% field-drop rate, the same 16.00B classifier still protects the
strict pages: held-out strict recall is 97.63-98.32% and under-strict rate is
3.71-4.69%. The cost is visible and important: accuracy falls to
68.46-69.34%, over-strict rate rises to 26.66-27.83%, strict mode is selected
for 45.41-48.93% of claims, and estimated provenance touch rises to
8.89-8.97 cells/event. This is a useful failure shape for chip design: noisy
parsing can be handled conservatively with tiny local tables, but a parser-miss
guard or confidence counter is needed to reduce unnecessary strict repair.

The first parser-miss guard tradeoff makes that next step concrete. It compares
the existing 16B 3D parser-noise LUT, a conservative 32B one-bit guard after
that LUT, and a 64B 4D LUT that adds parser-miss bucket as a fourth input. The
safe guard preserves the safety gates but barely moves the mean metrics:
over-strict falls from 27.08% to 26.86% and touch stays at 8.92 cells/event.
The 64B miss-aware LUT is the better hardware tradeoff so far: mean accuracy
rises from 68.82% to 71.34%, mean over-strict falls to 24.17%, mean touch drops
to 8.84 cells/event, and mean strict recall remains 98.05%. Under-strict rises
slightly from 4.10% to 4.49%, so this is not a free win; it is a useful
state/quality knob for parser-noisy compiled wiki pages.

The multi-field text trace is the next step toward real wiki snippets. Each
source cell now carries four fields: status, priority, region, and owner. The
teacher weights status and priority as core fields, while region and owner are
weaker metadata. Under the same 6% misread and 3% drop parser noise, the CA
controller uses four 2-bit local buckets: observed weighted error, core-field
conflict, observed weighted stale/source disagreement, and parser misses. The
classifier table is 64.00B and the provenance table remains 1.125B. On four
held-out traces, accuracy is 79.10-79.69%, strict recall is 97.57-98.96%,
under-strict rate is 1.27-1.95%, and estimated provenance touch is
9.08-9.10 cells/event. This is a better abstraction than the single-status
demo: the local state now distinguishes field importance and source agreement
without adding a dense text model to the CA fabric.

## Kill Criteria

This track is not useful if:

- CA routing cannot beat flat retrieval on cells read per successful answer;
- triggered refresh creates persistent stale-answer failure modes;
- wiki compilation drops facts without a practical error-book repair path;
- the decoder still needs to reread most pages to answer accurately.

## Decision

Do not abandon the transformer-like language-model track. Instead, split the
research:

- output reducer track: prove CA-like local reducers can replace expensive
  output-side scoring and content exposure;
- wiki-memory track: prove CA-like local storage can maintain and retrieve
  mutable knowledge better than weight updates or flat RAG.

If the wiki-memory track works first, it is the cleaner chip story: a CA memory
accelerator that serves any decoder, then gradually absorbs more reasoning and
generation logic.

This also reframes the Karpathy-style LLM-Wiki question. A CA fabric does not
need to beat the whole Transformer first. It can first become the mutable
knowledge substrate: local pages, local dirty summaries, local disagreement
counters, and low-bit learned guards that decide when facts are queryable or
need repair. That is a more natural first chip target than trying to replace
all dense attention and MLP computation at once.
