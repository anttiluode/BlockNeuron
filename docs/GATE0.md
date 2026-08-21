# Gate 0 — Same mass, different machine

## Question

Can a single fixed structural substrate expose measurably different effective input graphs when only low-dimensional chemical/rhythmic state changes?

This is a **mechanism gate**, not a novelty or superiority gate.

## Frozen toy

Input `x in R^8` is always sampled from the same isotropic Gaussian. There are four mutually orthogonal binary classification rules. Two context variables select the active rule:

- chemical mode `m in {-1,+1}` selects a pair of dendritic branches;
- phase `phi in {0,pi}` selects one branch inside the pair.

The block has four persistent structural branch vectors. Its effective weight is

```text
W_eff(m, phi) = sum_b gate_b(m, phi) W_b
```

with

```text
chemical_gate_b = sigmoid(beta * m * receptor_b)
phase_gate_b    = sigmoid(beta * cos(phi - psi_b))
gate_b          = chemical_gate_b * phase_gate_b
```

Only the four structural branch vectors are trained in Gate 0. Receptor signs and phase preferences are fixed so this gate cannot hide a routing-discovery problem inside the optimizer.

## Attackers

`PLAIN`: one unconditioned linear classifier. It has no task/context signal.

`HYPER`: an ordinary context-conditioned hyperlinear model. It directly generates an effective input vector from `[1, m, cos(phi), m*cos(phi)]`. This spans all four context-specific classifiers exactly. It is intentionally strong.

## Pass / stop lines

Mechanism pass:

- `BLOCK` full-context accuracy > 0.97;
- removing chemistry reduces accuracy below 0.82;
- removing phase reduces accuracy below 0.82;
- the four effective input vectors are not all the same.

Interpretation:

- if `HYPER` also solves the task, **unique capability is not established**;
- if `BLOCK` fails, stop and simplify the unit before adding fields, growth, or local learning.

## First receipt

Seed `18001`, 700 Adam steps:

```text
name   params   full     -chem    -phase   -both
BLOCK      33   0.9967   0.7491   0.7475   0.6658
HYPER      33   0.9958   0.7485   0.7502   0.6679
PLAIN       9   0.6644   0.6656   0.6673   0.6654
```

The four BLOCK effective input vectors were almost mutually orthogonal after training:

```text
[[ 1.000, -0.000, -0.001, -0.002],
 [-0.000,  1.000,  0.003, -0.002],
 [-0.001,  0.003,  1.000,  0.010],
 [-0.002, -0.002,  0.010,  1.000]]
```

Verdict:

> **Structural instrument established; unique capability NOT established.**

## Why this gate exists

The intended BlockNeuron equation is richer than a static point-neuron weight:

```text
g_e(t) = w_e * S(z_e(t)) * M(r_e^T m(t)) * R(delta_phi_e(t), psi_e)
```

Gate 0 isolates only `M` and `R`. Later gates must earn `z`, eligibility, diffusion, sparse growth, and development separately.
