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
