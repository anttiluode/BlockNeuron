# Gate 0T — tiny TSP on hysteretic matter

This is a deliberately silly-but-clean probe prompted by the question: **can the ferroelectric-inspired persistent substrate do a travelling-salesman problem?**

The answer being tested is capability, not novelty or competitiveness.

## Mapping

For `n` cities, keep an `n x n` continuous state `p[i, t]` meaning roughly “city `i` occupies tour position `t`.” A Sinkhorn readout turns the field into an approximately doubly-stochastic soft permutation `X`.

The soft cyclic route cost is

```text
C(X) = sum_t sum_ij d_ij X[i,t] X[j,t+1]
```

The hysteretic solver evolves the local state with the same double-well term used by Gate 0H, but now the TSP gradient acts as an external field:

```text
dp/dt = p - p^3 + E_tsp
E_tsp = -gain * dC/dp
```

At the end, the soft assignment is projected to the highest-scoring valid permutation. Therefore every reported candidate is a legal Hamiltonian cycle.

This is closer in spirit to a small analog/Hopfield-style optimizer than to a conventional learned neural TSP solver.

## Controls

The executable compares:

1. **material** — double-well / Landau relaxation;
2. **plain relaxation** — identical soft permutation and TSP objective, but no double-well state term;
3. **multi-start nearest-neighbour + 2-opt** — a strong simple classical heuristic;
4. **exact exhaustive optimum** — feasible only because the test deliberately uses tiny `n`.

The ordinary controls are intentionally allowed to win. If they tie or beat the material system, the result is only that hysteretic state *can host* the optimization.

## Run

```bash
python3.13 experiments/gate0t_tsp.py
```

Default:

```text
7 cities
6 random Euclidean instances
8 random restarts
300 relaxation steps/restart
```

CI uses a smaller deterministic receipt.

## Receipt metrics

For every solver:

- tour length / exact optimum;
- exact-optimum hit rate.

For the material solver also report:

- mean `|p|`, to show whether the local states actually settle toward wells;
- mean maximum assignment probability per tour position.

## Pass / stop lines

The toy capability gate passes if the material solver reaches mean tour ratio `<= 1.05` and exact-hit rate `>= 0.5` on the deterministic suite.

That does **not** mean it is a good TSP solver.

Stop lines:

- if the plain continuous relaxation ties/beats it, hysteresis was not needed;
- if nearest-neighbour + 2-opt ties/beats it, there is no algorithmic advantage here;
- no claim about scaling beyond tiny instances;
- the exact assignment projection itself becomes factorial and is only a measurement/readout convenience;
- no physical energy claim follows from the software dynamics.

## Why this is still relevant

TSP is a useful stress test because it asks a persistent substrate to settle into one globally compatible configuration out of many competing local assignments. If Gate 0T works at all, it shows the same material-state idea can be used as an optimization landscape, not just as a latch/configuration memory. The harder question would then be whether local hysteresis provides any useful search or amortization property that ordinary continuous optimization does not.
