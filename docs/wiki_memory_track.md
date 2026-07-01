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
