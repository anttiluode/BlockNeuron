# BlockNeuron

> **Find the next useful abstraction above the point neuron.**

A point neuron collapses a connection to roughly one scalar weight and a cell to weighted summation plus a nonlinearity. `BlockNeuron` asks whether a still-small artificial unit becomes useful when a connection has a tiny **life in time** and when one neuron has a few persistent **compartments** rather than one instantaneous scalar state.

This repository is **not** a brain simulation and does not claim that dendrites, oscillations, neuromodulators, developmental wiring, dynamic connectivity, or three-factor learning are new. Historical work reaches surprisingly close to several individual gates. The research question is whether putting a deliberately small subset of those ideas into **one computational block**, with strong ordinary and historical attackers, buys anything measurable in state/parameter budget, interference, learning accessibility, wiring/description cost, or executed work.

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

The first executable gate is intentionally small. Four orthogonal classification rules share exactly the same input distribution. A one-dimensional modulatory state selects a branch pair; phase selects a branch within that pair.

Run:

```bash
python -m pip install -e .
python experiments/gate0_same_mass_different_machine.py
pytest -q
```

The controls are part of the claim:

- `BLOCK` — four persistent branches, switched by modulation × phase;
- `HYPER` — ordinary context-conditioned hyperlinear attacker that can represent all four rules exactly;
- `PLAIN` — unconditioned linear classifier.

Expected interpretation if both `BLOCK` and `HYPER` solve it:

> **Structural instrument established. Unique capability not established.**

That tie is now unsurprising: multiplicative gating, hypernetworks and related conditional machinery are part of a common multiplicative-interaction family. Historical dendritic-gating work also already shows that branch gating can create context-dependent effective computation.

See [`docs/GATE0.md`](docs/GATE0.md) and [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## Gate ladder

| Gate | Question | Main attacker / score |
|---|---|---|
| 0 | Can low-dimensional mode/phase make one fixed substrate behave like different effective graphs? | hyperlinear / FiLM / DGN-style random half-space gating |
| 0b | Can a shared oscillatory/field operator select useful dynamic connectivity with less explicit routing? | explicit pairwise router, low-rank router, random/matched-smooth bases; **score wiring/description/communication cost at matched error** |
| 1 | Do **persistent** dendritic compartments buy temporal/context multiplexing at matched parameters/state? | GRU / SSM / FiLM / instantaneous DGN gates |
| 2 | Can pre/post eligibility + delayed modulator learn without immediate weight updates? | BPTT / e-prop / canonical three-factor controls |
| 3 | Can a **diffusing** low-dimensional modulator route useful credit or state better than broadcast context at matched cost? | broadcast scalar/vector, learned message passing, matched-spectrum state |
| 4 | Can identity + compatibility + geometry grow useful sparse addressing without storing `N^2` pair decisions? | DGN random half-space gates, random formation + stabilization, kNN/random geometric, learned sparse graph |
| 5 | Does slow growth consolidate recurring interactions so familiar computation becomes cheaper without catastrophic interference? | fixed sparse net / DGN-style continual learner / standard continual-learning baselines |

Every gate can kill an ingredient. We do not proceed by declaring that all ingredients must survive.

## Historical constraints that now shape the project

Several old results are too close to ignore:

- **Mel (1992), Clusteron:** a deliberately simplified dendritic neuron with nonlinear subunits and Hebbian learning already asks for a useful abstraction above a point neuron.
- **Poirazi & Mel (2001):** nonlinear dendritic subunits plus random synapse formation and activity-dependent stabilization/elimination; selective addressing onto dendritic subunits is itself a memory mechanism.
- **Sezener et al. (2021/22), Dendritic Gated Networks:** fixed random half-space gates select dendritic branches; local learning is data-efficient and resistant to forgetting. Random gating is therefore a mandatory attacker, not a weak baseline.
- **Von der Malsburg (1981):** fast synaptic modulation / temporary functional links on fixed anatomy.
- **Hoppensteadt & Izhikevich (1999):** a common forced medium imposes dynamic connectivity among oscillators; importantly, they make an `O(n)` junctions versus `O(n^2)` explicit-connections resource argument.
- **Bargmann (2012):** one anatomical wiring diagram can encode multiple functional circuits, some active and some latent under neuromodulatory state.
- **Tsodyks & Markram (1997):** synaptic state changes what temporal feature a connection transmits; future `S(z)` should start from canonical short-term-plasticity baselines.
- **Jayakumar et al. (ICLR 2020):** gating, attention, hypernetworks and dynamic convolution can be viewed through multiplicative interactions. Gate 0's BLOCK/HYPER tie is therefore expected; expressivity is not the interesting axis.
- **Noest (1987/88):** phasor neural networks already use continuous phase variables for associative memory, constraining how older Janus/Zeta phase-storage experiments should be framed.

The old literature does **not** collapse the whole BlockNeuron ladder. In particular, DGN-style gating is instantaneous rather than persistent branch dynamics, and none of these results by itself establishes the Gate-5 claim that repeated experience can slowly create structure that reduces future executed work.

## What is already active research now?

As of August 2026, essentially every ingredient also has a serious modern neighboring literature:

- **Dendritic architectures:** Chavlis & Poirazi (Nature Communications, 2025) show structured dendritic ANNs can match/outperform vanilla ANNs with substantially fewer trainable parameters on several image tasks.
- **Dendritic state as computation:** DendriCL (Shen, Wu & Chen, July 2026) embeds leaky online LMS in the subthreshold dynamics of one apical compartment, with fixed inference-time synaptic weights.
- **Dendrites + local/biologically motivated learning:** Kubo (May 2026) combines dendritic networks with equilibrium propagation and reports advantages over standard EP on harder/deeper settings.
- **Dendrites + temporal waves:** Kubo's Dendritic Wave RNN (July 2026) combines nonlinear basal dendrites with traveling-wave recurrent dynamics.
- **Rhythmic state:** AKOrN was an ICLR 2025 oral; KoPE (2026) adds evolving Kuramoto-like phase state to vision transformers.
- **Three-factor oscillatory systems:** *Phasor Agents* (2026) combines Stuart-Landau oscillator graphs, eligibility traces, sparse modulators, and phase-timed write windows.
- **Diffusive credit:** Barretto-Bittar et al. (March 2026) diffuse neuromodulatory error/credit locally through sparse recurrent spiking networks.
- **Neuromodulatory mode switching:** Tsuda et al. (Neural Computation, 2026) show that a simplified broadcast neuromodulator can make a common recurrent substrate express multiple, even opposed, behaviours.
- **Growth/development:** Neural Developmental Programs grow networks through local communication; Butkus et al. (2026) optimize growth in breadth, depth, and time; Barabási & Barabási's genetic-connectome model generates wiring from neuronal identity and compatibility rather than explicit pair decisions.

So the plausible open slot is **not** "nobody studies dendrites/oscillations/modulation/growth." It is whether a compact unit combining a few mechanisms produces a measurable **resource or learning advantage** over ordinary conditional/recurrent/sparse machinery and over historical dendritic/oscillatory baselines.

## Related repos in this project family

`BlockNeuron` should inherit mechanisms, not mythology:

- [`DifferentMachine`](https://github.com/anttiluode/DifferentMachine): repeated useful interaction can amortize future work into persistent sparse structure; inherited rule is not the acquired graph.
- [`SplatNeuronPlusField`](https://github.com/anttiluode/SplatNeuronPlusField): addressed/topological and metric/shared state are different address vocabularies; shared diffusive history only helped when task structure matched metric geometry.
- [`GeometricNeuronV23`](https://github.com/anttiluode/GeometricNeuronV23): keep local temporal dynamics attached to spatial addresses and attack the structured-vs-shuffled contrast.

Those results argue for **separating mechanisms and timescales**, not throwing every biological idea into one simulation.

## Literature ledger

Historical anchors:

- von der Malsburg, *The Correlation Theory of Brain Function* (1981)
- Mel, *The Clusteron: Toward a Simple Abstraction for a Complex Neuron* (1992)
- Tsodyks & Markram, *The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability* (PNAS, 1997), DOI `10.1073/pnas.94.2.719`
- Hoppensteadt & Izhikevich, *Oscillatory Neurocomputers with Dynamic Connectivity* (PRL, 1999), DOI `10.1103/PhysRevLett.82.2983`
- Poirazi & Mel, *Impact of Active Dendrites and Structural Plasticity on the Memory Capacity of Neural Tissue* (Neuron, 2001), DOI `10.1016/S0896-6273(01)00252-5`
- Bargmann, *Beyond the connectome: How neuromodulators shape neural circuits* (BioEssays, 2012), PMID `22396302`
- Noest, *Phasor Neural Networks* (NIPS 1987 proceedings / 1988 publication)
- Jayakumar et al., *Multiplicative Interactions and Where to Find Them* (ICLR 2020)
- Sezener et al., *A rapid and efficient learning rule for biological neural circuits* (bioRxiv 2021/2022), DOI `10.1101/2021.03.10.434756`

Recent neighbors:

- Barabási & Barabási, *A Genetic Model of the Connectome* (Neuron, 2020), DOI `10.1016/j.neuron.2019.10.031`
- Najarro, Sudhakaran & Risi, *Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs* (2023), arXiv `2307.08197`
- Huang et al., scalable deep dendritic SNN / branch gating work, arXiv `2412.06355`
- Chavlis & Poirazi, *Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning* (2025), DOI `10.1038/s41467-025-56297-9`
- Miyato et al., *Artificial Kuramoto Oscillatory Neurons* (ICLR 2025 Oral)
- Lei, Gu & Gao, *Dendritic Learning for AI: A Survey...* (2026), DOI `10.53941/jaia.2026.100006`
- Shen, Wu & Chen, *Dendritic In-Context Learning in a Single-Layer Spiking Neural Network* (2026), arXiv `2607.02283`
- Kubo, *Dendritic Neural Networks with Equilibrium Propagation* (2026), arXiv `2605.08135`
- Xiao et al., *Kuramoto Oscillatory Phase Encoding* (2026), arXiv `2604.07904`
- Trappe, *Phasor Agents* (2026), arXiv `2601.04362`
- Barretto-Bittar et al., *Diffusion of Neuromodulators for Temporal Credit Assignment* (2026), arXiv `2603.08949`
- Tsuda et al., *Neuromodulators Generate Multiple Context-Relevant Behaviors in Recurrent Neural Networks* (2026), DOI `10.1162/NECO.a.1489`
- Butkus, Garzón Gupta & Kriegeskorte, *Growing a Neural Network in Breadth, Depth, and Time* (2026), arXiv `2605.25174`

## Status

**Gate-0 research prototype. No novelty claim. No biological-equivalence claim.**
