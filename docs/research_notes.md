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
With 4 Cellular-MoE ticks per event, the current HARC-CA profile moves about
51KB of local on-chip bytes per event and keeps about 182KB of on-chip state.
The tiny Transformer KV reference reads about 384MB per token at 16k context.

This ratio is intentionally not treated as a win. The HARC-CA prototype is not a
quality-equivalent model, and local SRAM/register traffic is not the same as
HBM/cache traffic. The useful conclusion is narrower: the current architecture
has a measurable path to keeping its toy next-token behavior inside local
low-bit traffic, which is the right bottleneck direction for a CA-first chip.

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
