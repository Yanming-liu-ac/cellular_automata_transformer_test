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
groups, and 4x256x4-bit page/group summaries. Queries mix single-hop fact reads
and two-hop link reads. Updates now change the source-of-truth first, then mark
only the local page and group dirty; refresh or error-book repair is required
before the routed memory cells see the new key or revised value. Half of the
updates are value revisions, and 25% of eligible queries replay an error-book
probe.

The exact-update baseline refreshes page and group summaries after every fact
edit. It reaches 100.0% recall, but writes about 18,460 score-equivalent cells
per update. The conservative triggered policy, `trigger16_age16`, refreshes
when 16 pages are dirty or the summary is 16 update steps old. On the same
event stream it reaches 96.48% overall recall, 93.81% recall on recently updated
pages, and 3.52% stale misses while writing about 11,418 cells/update. Adding
error-book repair raises recall to 97.66%, repeated failed-probe recall to
99.21%, and reduces value-stale misses to 0.39%, at about 11,910 cells/update.
Reads stay about 357 cells/query versus 1,024 for a flat exact page-fact scan, a
65.1% read reduction. With no refresh, recall drops to 56.45% and stale misses
rise to 43.55%.

This is not yet a learned memory system, but it establishes the first measurable
wiki-memory claim: local dirty/age summary refresh can keep mutable facts mostly
queryable while avoiding full-wiki scans. The error-book repair path now has a
real workload: it recovers all 12 repaired probes immediately and improves
repeated-probe recall from 90.70% to 99.21%. The next step is to make
contradiction clusters explicit instead of treating each value revision as an
independent fact edit.

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
