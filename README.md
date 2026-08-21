# BlockNeuron

> **Find the next useful abstraction above the point neuron.**

A point neuron collapses a connection to roughly one scalar weight and a cell to weighted summation plus a nonlinearity. `BlockNeuron` asks whether a still-small artificial unit becomes useful when a connection has a tiny **life in time** and when one neuron has a few persistent **compartments** rather than one instantaneous scalar state.

This repository is **not** a brain simulation and does not claim that dendrites, oscillations, neuromodulators, developmental wiring, or three-factor learning are new. The research question is whether putting a deliberately small subset of those ideas into **one computational block**, with strong ordinary attackers, buys anything measurable.

## The block

The current target abstraction is:

```text
neuron i
    identity            x_i
    position            p_i
    dendritic states    d_i1 ... d_iB
    soma state          v_i
    oscillatory state   phi_i
    modulator emission  q_i
    branch receptors    r_ib
    sparse connections  E_i
```

Each connection stores only:

```text
w_e              slow structural strength / "mass"
z_e(t)           small fast synaptic/contact state
e_e(t)           eligibility trace
target branch    b_e
delay             delta_e
```

The central replacement for a static `w_ij` is

```text
g_e(t) = w_e
         * S(z_e(t))
         * M(r_e^T m(t))
         * R(phi_pre(t) - phi_post(t), psi_e)
```

A deliberately crude rhythmic factor can be

```text
R = 1 + a cos(phi_pre - phi_post - psi_e),     0 <= a < 1
```

and a branch state can start as nothing fancier than

```text
tau_d d_dot_ib = -d_ib + F(sum_{e -> (i,b)} g_e(t) s_pre(e))
```

followed by somatic integration across branches.

The point is the separation of timescales:

```text
fast activity / phase
        ↓
local branch + contact state
        ↓
slow structural weights / sparse routes
        ↓
very slow identity / developmental rule
```

Not everything gets to move at once.

## Gate 0 — same mass, different machine

The first executable gate is intentionally small. Four orthogonal classification rules share exactly the same input distribution. A one-dimensional chemical state selects a branch pair; phase selects a branch within that pair.

Run:

```bash
python -m pip install -e .
python experiments/gate0_same_mass_different_machine.py
pytest -q
```

The controls are part of the claim:

- `BLOCK` — four persistent branches, switched by chemistry × phase;
- `HYPER` — ordinary context-conditioned hyperlinear attacker that can represent all four rules exactly;
- `PLAIN` — unconditioned linear classifier.

Expected interpretation if both `BLOCK` and `HYPER` solve it:

> **Structural instrument established. Unique capability not established.**

See [`docs/GATE0.md`](docs/GATE0.md).

## Gate ladder

| Gate | Question | Main attacker |
|---|---|---|
| 0 | Can low-dimensional mode/phase make one fixed substrate behave like different effective graphs? | context-conditioned hyperlinear / FiLM |
| 1 | Do persistent dendritic compartments buy temporal/context multiplexing at matched parameters/state? | GRU / MLP / FiLM |
| 2 | Can pre/post eligibility + delayed modulator learn without immediate weight updates? | BPTT / e-prop-style controls |
| 3 | Can a **diffusing** low-dimensional modulator route useful credit or state better than broadcast context at matched cost? | broadcast scalar/vector, learned message passing |
| 4 | Can identity + compatibility + geometry grow a useful sparse graph without storing `N^2` pair decisions? | learned sparse graph / kNN / random geometric graph |
| 5 | Does slow growth consolidate recurring interactions so familiar computation becomes cheaper without catastrophic interference? | fixed sparse net / continual-learning baseline |

Every gate can kill an ingredient. We do not proceed by declaring that all ingredients must survive.

## What is already active research?

As of August 2026, essentially every ingredient has a serious neighboring literature, but mostly in separate systems:

- **Dendritic architectures:** Chavlis & Poirazi (Nature Communications, 2025) show structured dendritic ANNs can match/outperform vanilla ANNs with substantially fewer trainable parameters on several image tasks. A 2026 survey now treats dendritic learning as a substantial architecture/learning literature.
- **Dendritic state as computation:** DendriCL (Shen, Wu & Chen, July 2026) embeds leaky online LMS in the subthreshold dynamics of one apical compartment, with fixed inference-time synaptic weights.
- **Dendrites + local/biologically motivated learning:** Kubo (May 2026) combines dendritic networks with equilibrium propagation and reports advantages over standard EP on harder/deeper settings.
- **Rhythmic state:** AKOrN was an ICLR 2025 oral; KoPE (2026) adds evolving Kuramoto-like phase state to vision transformers and reports efficiency/structured-reasoning gains.
- **Three-factor oscillatory systems:** *Phasor Agents* (2026) combines Stuart-Landau oscillator graphs, eligibility traces, sparse modulators, and phase-timed write windows.
- **Diffusive credit:** Barretto-Bittar et al. (March 2026) diffuse neuromodulatory error/credit locally through sparse recurrent spiking networks.
- **Neuromodulatory mode switching:** Tsuda et al. (Neural Computation, 2026) show that a highly simplified broadcast neuromodulator can make a common recurrent substrate express multiple, even opposed, behaviours.
- **Growth/development:** Neural Developmental Programs grow networks through local communication; Butkus et al. (2026) optimize growth in breadth, depth, and time; Barabási & Barabási's genetic-connectome model shows how neuronal identity plus compatibility operators can describe wiring without explicitly encoding every edge.

So the plausible open slot is **not** "nobody studies dendrites/oscillations/modulation/growth." It is the engineering/scientific question of whether a compact unit that combines a few of these mechanisms has a measurable advantage over ordinary conditional/recurrent/sparse machinery.

## Related repos in this project family

`BlockNeuron` should inherit mechanisms, not mythology:

- [`DifferentMachine`](https://github.com/anttiluode/DifferentMachine): repeated useful interaction can amortize future work into persistent sparse structure; inherited rule is not the acquired graph.
- [`SplatNeuronPlusField`](https://github.com/anttiluode/SplatNeuronPlusField): addressed/topological and metric/shared state are different address vocabularies; shared diffusive history only helped when task structure matched metric geometry.
- [`GeometricNeuronV23`](https://github.com/anttiluode/GeometricNeuronV23): keep local temporal dynamics attached to spatial addresses and attack the structured-vs-shuffled contrast.

Those results argue for **separating mechanisms and timescales**, not throwing every biological idea into one simulation.

## Literature ledger

Primary sources / useful neighbors:

- Lei, Gu & Gao, *Dendritic Learning for AI: A Survey...* (2026), DOI `10.53941/jaia.2026.100006`
- Chavlis & Poirazi, *Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning* (2025), DOI `10.1038/s41467-025-56297-9`
- Shen, Wu & Chen, *Dendritic In-Context Learning in a Single-Layer Spiking Neural Network* (2026), arXiv `2607.02283`
- Kubo, *Dendritic Neural Networks with Equilibrium Propagation* (2026), arXiv `2605.08135`
- Miyato et al., *Artificial Kuramoto Oscillatory Neurons* (ICLR 2025 Oral)
- Xiao et al., *Kuramoto Oscillatory Phase Encoding* (2026), arXiv `2604.07904`
- Trappe, *Phasor Agents* (2026), arXiv `2601.04362`
- Barretto-Bittar et al., *Diffusion of Neuromodulators for Temporal Credit Assignment* (2026), arXiv `2603.08949`
- Tsuda et al., *Neuromodulators Generate Multiple Context-Relevant Behaviors in Recurrent Neural Networks* (2026), DOI `10.1162/NECO.a.1489`
- Najarro, Sudhakaran & Risi, *Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs* (2023), arXiv `2307.08197`
- Butkus, Garzón Gupta & Kriegeskorte, *Growing a Neural Network in Breadth, Depth, and Time* (2026), arXiv `2605.25174`
- Barabási & Barabási, *A Genetic Model of the Connectome* (Neuron, 2020), DOI `10.1016/j.neuron.2019.10.031`

## Status

**Gate-0 research prototype. No novelty claim. No biological-equivalence claim.**
