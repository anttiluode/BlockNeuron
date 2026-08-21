# Prior-art boundary — August 2026

`BlockNeuron` is not based on the claim that dendrites, oscillations, neuromodulation, three-factor learning, diffusion, or developmental growth are individually new. They are all active research areas.

The useful question is narrower:

> Can a compact artificial-neuron unit combine a few of these mechanisms in one falsifiable object and beat strong ordinary conditional/recurrent/sparse controls on some measurable resource or learning property?

## Closest neighboring systems found

### Dendritic ANN architecture

Chavlis & Poirazi (Nature Communications, 2025) use structured dendritic connectivity and restricted input sampling. Their dendritic ANNs match or outperform vanilla ANNs on several image tasks while using substantially fewer trainable parameters. This is strong evidence that a less-degenerate unit can be computationally useful, but it does not supply BlockNeuron's explicit chemical/rhythmic functional-graph switching, local eligibility, diffusive field, or developmental sparse graph.

### Dendritic dynamics as an algorithm

DendriCL — Shen, Wu & Chen (arXiv:2607.02283, July 2026) — treats the apical compartment itself as a dynamical computational substrate. Its subthreshold recurrence implements leaky online Widrow-Hoff LMS and a probe recovers the reference learner trajectory at R²=0.93. This is unusually close to the principle that *the unit's internal state is computation*, rather than dendrites merely being extra feed-forward nonlinearities.

### Dendrites plus equilibrium propagation

Kubo, *Dendritic Neural Networks with Equilibrium Propagation* (arXiv:2605.08135, May 2026), combines dendritic architecture with EP and reports improvement over standard EP on harder/deeper settings.

Important bibliographic correction: `2605.08135` is this Kubo paper; it is **not** DendriCL. DendriCL is `2607.02283`.

### Dendrites plus temporal wave dynamics — very close on two axes

Kubo, *Dendritic Wave Recurrent Neural Networks* (bioRxiv 2026.07.03.736415, posted July 2026), adds nonlinear basal dendritic branches to a wave recurrent neural network while preserving traveling-wave recurrent dynamics. It reports small but consistent accuracy gains and lower across-seed variability on sMNIST, psMNIST, and noisy sequential CIFAR-10.

This occupies an important part of the design space: **dendritic computation + explicitly spatiotemporal recurrent dynamics is already active work.**

It still differs from BlockNeuron's target object: there is no low-dimensional diffusing modulatory medium selecting temporary effective graphs, no per-contact three-factor eligibility as the central learning primitive, and no identity/geometry-driven developmental connectivity in the same unit.

### Oscillators plus three-factor local learning

Trappe, *Phasor Agents* (arXiv:2601.04362, January 2026), uses coupled Stuart-Landau oscillators. Phase carries relative timing, amplitude carries gain, and coupling weights use eligibility traces gated by sparse modulators and oscillation-timed write windows without backpropagation. This is probably the closest neighbor to BlockNeuron's **phase + eligibility + modulator** corner.

It is a flat oscillator graph rather than a neuron with internal dendritic compartments, and it does not include spatial diffusive modulation or developmental connectivity.

### Diffusing modulation for credit

Barretto-Bittar et al., *Diffusion of Neuromodulators for Temporal Credit Assignment* (arXiv:2603.08949, March 2026), diffuse error information locally through sparse recurrent spiking networks. Learning depends on local concentration of a diffusing credit signal and improves over an eligibility-propagation baseline on three benchmark tasks.

This is directly relevant to a future BlockNeuron field gate. It means `diffusive modulation` itself cannot be the novelty claim; the question must be whether it buys something when coupled to branch-specific receptors / local compartment state and compared with broadcast or learned-message controls.

### One substrate, multiple behaviors under neuromodulation

Tsuda et al., *Neuromodulators Generate Multiple Context-Relevant Behaviors in Recurrent Neural Networks* (Neural Computation, 2026) demonstrate that even a highly simplified broadcast neuromodulator can let a common recurrent substrate express multiple stored behaviors, including opposed behaviors.

This is the strongest warning for Gate 0: ordinary low-dimensional conditioning can already produce "same weights, different behavior." Therefore BlockNeuron must beat explicit FiLM/hypernetwork/context controls rather than treating mode switching itself as the result.

### Rhythmic state in modern architectures

Artificial Kuramoto Oscillatory Neurons (AKOrN, ICLR 2025 Oral) and Kuramoto Oscillatory Phase Encoding (KoPE, 2026) treat phase/synchronization as computational state. Again, phase is an occupied ingredient; BlockNeuron has to test the *interaction* between phase, compartment, local state, and learning.

### Growth / developmental encoding

Neural Developmental Programs grow networks by repeated local cell communication. Butkus et al., *Growing a Neural Network in Breadth, Depth, and Time* (arXiv:2605.25174, 2026), optimize resource costs so computation grows organically in spatial and temporal dimensions. Barabási & Barabási's *A Genetic Model of the Connectome* (Neuron, 2020) is especially relevant to the compact-growth idea: neuronal identity plus compatibility rules can specify connectivity without explicitly storing every pairwise edge decision.

`DifferentMachine` independently arrived at the engineering version we want to carry over: the inherited object can be a small developmental rule rather than an acquired graph.

## Current boundary

I did **not** find, in this search, one established artificial-neuron abstraction that simultaneously makes all of the following first-class inside the same compact unit:

```text
persistent dendritic compartments
+ tiny per-contact fast state
+ explicit oscillatory phase
+ low-dimensional branch-specific modulatory state
+ local eligibility / delayed third-factor consolidation
+ sparse identity/geometry-generated connectivity
+ slower growth / pruning
```

Absence from a search is not proof of novelty. Several neighboring systems are close enough that the scientific strategy must be **ablation and strong attackers**, not a compositional novelty claim.

The repo therefore treats every ingredient as guilty until it earns its place.
