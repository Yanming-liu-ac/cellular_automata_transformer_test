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

The first non-neural retrieval component is a hash-routed associative CA lane.
It is closer to a hardware primitive than to a trained model:

- a query routes through a logarithmic local tree using hash bits;
- it lands in one set-associative bucket;
- a small number of low-bit tags are compared in parallel;
- the value returns without scanning all sequence cells.

This is deliberately not exact Transformer attention. It tests whether a CA
fabric can provide sparse exact recall for copy and induction tasks.

The first sweep showed:

| Context | Buckets | Ways | Correct Recall | Cells Visited | Full Scan |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1,024 | 4 | 99.3% | 14 | 1,024 |
| 4,096 | 4,096 | 4 | 99.2% | 16 | 4,096 |
| 16,384 | 16,384 | 4 | 99.6% | 18 | 16,384 |

The warning sign is capacity pressure: at load factor 1.0, the same 4-way design
falls to roughly 80% recall due to bucket evictions. The chip architecture
therefore needs either more ways, better hashing, learned routing, overflow
lanes, or tiered memory for rare exact facts.

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
