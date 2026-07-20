# AXIOM: Market Analysis

## Verdict

AXIOM is a research instrument, not a product, and it should be judged as leverage rather than
revenue. As a commercial artifact its direct market is near zero: the audience that wants a
configurable CA engine is small, technical, and served by free tools. As a research and
credibility asset in an under-tooled niche, it is worth building. **Confidence: high on the
"no direct revenue" read, moderate on the "worth building as leverage" read.**

The honest one-line summary: the value is the instrument existing and pointing at a thin seam
nobody else tools well, not anyone paying for it.

## Problem space

Three communities rarely share a codebase:

- **Cellular automata / artificial life**: Lenia, NCA, reaction-diffusion. Tools render well
  and measure little.
- **Graph machine learning**: message passing, GNNs, PageRank. A separate ecosystem, mostly
  Python.
- **World models / sub-quadratic sequence models**: state-space models, linear attention.

The gap AXIOM addresses is the intersection: expressing grid CA, swarms, and graph or hypergraph
dynamics as one configurable substrate, then measuring them with graph, topological, and
dynamical descriptors together. That combination is a research convenience, not a felt pain a
buyer pays to remove.

## Competition

| Tool | Covers | Gap AXIOM fills |
|---|---|---|
| Chakazul / Lenia notebooks | Lenia variants, reference | Not an engine; no graph or analysis layer |
| CAX (JAX) | Differentiable CA, NCA, fast | Grid-centric; not the graph/hypergraph seam |
| Leniabreeder | Quality-diversity over Lenia | Discovery only; no unified substrate or GPU/CPU engine |
| Golly / Ready | Life-like CA / reaction-diffusion | Fixed rule families; no learning, no graph analysis |
| PyTorch Geometric / DGL | Graph ML | No CA/swarm substrate; no live simulation instrument |

No single tool treats grid, particle, graph, and hypergraph as one configurable substrate with
PageRank and detection as first-class observers. That is the defensible position, and it is a
position in mindshare, not a market.

## Sizing

- **Top-down:** the serious alife-plus-graph-ML-plus-Rust audience is plausibly hundreds to low
  thousands of people worldwide. Willingness to pay is essentially zero; the norm is open source.
- **Bottom-up reframe:** the correct denominator is not users but experiments enabled. The engine
  pays off if it turns cross-domain ideas (a learned Hebbian rule on a hypergraph, PageRank over
  an evolving swarm, a spacetime-relaxed world model) into one-config experiments instead of
  separate projects. Value accrues to the builder, as research velocity and as a portfolio
  artifact, not as sales.

## Who benefits

- The builder: a reusable instrument for a seam they already work in, plus a credible public
  artifact.
- Alife / complex-systems researchers who want measurement, not just rendering.
- Nobody with a budget line for it, which is the point.

## Willingness to pay

Effectively nil for the software itself. Adjacent paths that could carry value: a hosted
exploration gallery, teaching material, or a paper that uses the instrument as its apparatus. All
of these monetize the output or the reputation, not the engine.

## Defensibility

<!-- bull -->
**Bull case.** The seam is thin and few are fluent in CA, graphs, and ML together. The analysis
layer (graph, topological, dynamical descriptors on the same live state) is genuinely rare. The
one-config-per-experiment leverage compounds: each new module multiplies with the existing ones.

<!-- bear -->
**Bear case.** Every individual mechanism already exists (Lenia, NCA, graph-CA, Hebbian, QD,
PageRank). The novelty is integration, which is not patentable and is reproducible by anyone who
decides the seam is worth their time. CAX plus a graph library gets a motivated researcher most
of the way. The moat is taste and follow-through, not technology.

## The demand read

There is no external demand to speak of, and manufacturing it would be a mistake. The right
framing is internal: AXIOM is worth the build because it lowers the cost of a class of
experiments the builder actually runs and produces a legible artifact in a niche they own. Judge
the next increment by whether it enables a specific experiment or strengthens the artifact, not by
a user count that will not exist.

## Legal and regulatory

None. Simulation software over synthetic data, no personal data, no regulated domain.
