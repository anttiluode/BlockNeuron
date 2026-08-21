# Prior-art boundary — August 2026

`BlockNeuron` is not based on the claim that dendrites, oscillations, neuromodulation, three-factor learning, diffusion, or developmental growth are individually new. They are all established research areas, several with important precursors decades older than the current deep-learning wave.

The useful question is narrower:

> Can a compact artificial-neuron unit combine a few of these mechanisms in one falsifiable object and beat strong ordinary conditional/recurrent/sparse controls on some measurable resource, interference, learning, or execution property?

The historical literature changes the emphasis of the project. **Mode switching and multiplicative gating are not the target result.** The likely axes that remain worth testing are persistent local state, interference/sample efficiency, wiring/description cost, locally available learning signals, and whether repeated experience can consolidate computation into cheaper persistent structure.

## Historical spine — the ideas are older than this repo

### Dendritic subunits as an abstraction, not a biophysical simulation

Mel's *Clusteron* (1992) explicitly asked for a simple abstraction of a complex neuron. It treated nonlinear dendritic clustering as a virtual hidden layer inside one neuron and showed that a Hebbian-type rule could exploit spatial ordering of contacts.

Poirazi & Mel, *Impact of Active Dendrites and Structural Plasticity on the Memory Capacity of Neural Tissue* (Neuron, 2001), is even closer to later BlockNeuron gates. They compare linear dendritic integration with separately thresholded nonlinear dendritic subunits, obtain much larger storage capacity for the nonlinear case, and show that the capacity is accessible to a structural rule combining **random synapse formation with activity-dependent stabilization/elimination**. Their interpretation is directly relevant to Gate 4: long-term information may reside partly in **which contacts are addressed to which dendritic subunits**, not only in scalar connection strengths.

This means that `compartments + structural addressing` is old territory. Gate 4 must test an engineering/resource question under strong random-formation baselines rather than present selective dendritic addressing as a new principle.

### Dendritic Gated Networks — random half-space gating is a mandatory attacker

Sezener et al., *A rapid and efficient learning rule for biological neural circuits* (bioRxiv 2021/2022), introduce Dendritic Gated Networks (DGNs). Each branch has a binary gate controlled by the input. In their simulations the mapping from input to gates is **not learned**: they use fixed random half-space gates

```text
g_b(x) = 1[v_b · x >= theta_b]
```

with the random `v_b, theta_b` held fixed during learning. The authors emphasize that the random mapping is a key ingredient in representing nonlinear functions efficiently. Weight learning is local to each unit and the system is resistant to catastrophic forgetting.

This is a serious prior-art constraint:

- it attacks Gate 0's broad claim that branch gating creates context-dependent effective machines;
- it is a mandatory Gate 4 attacker because **random generated addressing may already be enough**;
- its continual-learning result is a Gate 5 attacker on interference.

But it does **not** pre-empt the actual Gate 1 question: DGN gates are instantaneous functions of the current input, not persistent branch states with their own temporal history. It also does not establish Gate 5's proposed execution claim that repeated interaction slowly builds structure that makes familiar computation cheaper.

### Dendritic branch gating at deep-SNN scale

Huang, Fang, Ma, Li & Tian, arXiv `2412.06355` (first posted 2024; later versions/titles vary), develop scalable deep dendritic spiking networks and a dendritic branch-gating mechanism for task-incremental learning. A context/task signal modulates branch strengths and reduces inter-task interference.

This is very close to the **context selects dendritic branches** portion of Gate 0 at realistic network scale. It reinforces the Gate-0 verdict: mode-conditioned branch selection is an occupied mechanism, not a novelty claim.

### Dynamic functional connectivity on fixed anatomy — 1981 and 1999

Von der Malsburg's *Correlation Theory of Brain Function* (1981) proposed fast synaptic modulation that switches connections between conducting and non-conducting states on a faster timescale than long-term plasticity. The broad idea of temporary functional links on fixed anatomy is therefore much older than modern attention/gating language.

Hoppensteadt & Izhikevich, *Oscillatory Neurocomputers with Dynamic Connectivity* (Physical Review Letters 82, 2983–2986, 1999), is the strongest historical warning for a field/oscillation gate. Their architecture uses oscillators with distinct frequencies coupled weakly through a common medium. External forcing can impose **dynamic connectivity** even though the physical coupling is homogeneous. They show oscillatory associative properties and explicitly make a resource argument: a conventional fully connected `n`-unit machine uses order `n^2` pairwise connections, while their common-medium architecture needs order `n` oscillator-to-medium junctions.

This changes the right question for a future BlockNeuron field gate. Do not ask only whether a field basis improves accuracy. Ask:

> At matched task error, can a shared medium select useful effective interactions with materially less stored routing/wiring/communication than explicit pairwise machinery?

That is a stronger and older target than simply demonstrating phase-dependent effective graphs.

### Neuromodulation: one anatomy can contain multiple latent circuits

Bargmann, *Beyond the connectome: How neuromodulators shape neural circuits* (BioEssays, 2012), states the biological version unusually clearly: ultrastructural wiring diagrams are incomplete because neuromodulators change neuronal dynamics, excitability and synaptic function; a single anatomical connectivity map can encode multiple functional circuits, some active and some latent at a given time.

This is extremely close to the repo's phrase **same slow mass, different effective machine**. The biological phenomenon is prior art. BlockNeuron can only contribute an artificial abstraction/resource result, not the principle that modulation reconfigures fixed anatomy.

### Dynamic synapses: `S(z)` already has a canonical cheap family

Tsodyks & Markram (PNAS, 1997) showed that activity-dependent synaptic depression changes what aspect of presynaptic activity is transmitted: slow depression emphasizes firing rate while fast depression emphasizes temporal coherence. Related Tsodyks–Markram short-term-plasticity models provide compact facilitation/depression state variables.

Therefore future BlockNeuron contact state `z_e(t)` should start with a canonical short-term-plasticity baseline rather than inventing arbitrary state dynamics. The scientific question is what the contact state buys **inside the larger block**, not whether synapses can have fast local history.

### Multiplicative interactions unify many modern attackers

Jayakumar et al., *Multiplicative Interactions and Where to Find Them* (ICLR 2020), treat multiplicative interaction as a common framework spanning gating, attention, hypernetworks and dynamic convolution. They show that multiplicative layers enrich representable function classes and argue that their likely payoff is inductive bias for fusing streams and conditional computation.

This explains the current Gate-0 tie rather than weakening it:

```text
BLOCK ~= context-conditioned HYPER
```

is expected because both are multiplicatively conditioned function families. **Expressivity is not the interesting axis.** Later gates should measure sample efficiency, interference, state/parameter budget, wiring/description cost, local-learning accessibility, and executed work.

### Phasor / complex-valued memory predates Janus

Noest's *Phasor Neural Networks* (NIPS 1987 proceedings / 1988 publication) uses unit-length 2-vectors / continuous phase as local variables and analyzes associative-memory networks over phase patterns. Later complex-valued associative-memory and holographic-representation work expands this family.

This is relevant to the older Janus/Zeta experiments: phase-addressed storage and interpolation are not conceptually new merely because the implementation is a coordinate MLP/CVNN. A useful Janus follow-up would therefore be a **capacity/crosstalk curve** rather than another visual interpolation demo: number of stored phase slots versus endpoint error, cross-talk, and quality of intermediate phase states.

## Closest recent neighboring systems

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

This is another strong warning for Gate 0: ordinary low-dimensional conditioning can already produce "same weights, different behavior." Therefore BlockNeuron must beat explicit FiLM/hypernetwork/context controls rather than treating mode switching itself as the result.

### Rhythmic state in modern architectures

Artificial Kuramoto Oscillatory Neurons (AKOrN, ICLR 2025 Oral) and Kuramoto Oscillatory Phase Encoding (KoPE, 2026) treat phase/synchronization as computational state. Again, phase is an occupied ingredient; BlockNeuron has to test the *interaction* between phase, compartment, local state, and learning.

### Growth / developmental encoding

Neural Developmental Programs grow networks by repeated local cell communication. Butkus et al., *Growing a Neural Network in Breadth, Depth, and Time* (arXiv:2605.25174, 2026), optimize resource costs so computation grows organically in spatial and temporal dimensions. Barabási & Barabási's *A Genetic Model of the Connectome* (Neuron, 2020) is especially relevant to the compact-growth idea: neuronal identity plus compatibility rules can specify connectivity without explicitly storing every pairwise edge decision.

`DifferentMachine` independently arrived at the engineering version we want to carry over: the inherited object can be a small developmental rule rather than an acquired graph.

## Field / eigenmode warning

A geometry-derived orthogonal basis is not automatically an advantage. Pang et al. (Nature, 2023) showed that cortical geometric eigenmodes compactly reconstruct macroscale brain activity, but Faskowitz et al. challenged the specificity of this result to the exact cortical shape/orientation; Pang et al. replied that appropriate nulls recover specificity. The debate is a useful experimental warning rather than a settled verdict for BlockNeuron.

Therefore a future field experiment should include matched smooth/orthogonal/null bases **and** should not use accuracy alone as its target. The more defensible axis is the Hoppensteadt–Izhikevich one: what useful effective connectivity can a shared physical/operator structure provide per stored parameter, junction, communication event, or synchronization cost?

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

Absence from a search is not proof of novelty. More importantly, several subsets already have deep historical literatures. The scientific strategy must therefore be **ablation, strong attackers and explicit resource accounting**, not a compositional novelty claim.

## Mandatory attacker ledger after historical review

```text
Gate 0
  context-conditioned hyperlinear / FiLM / multiplicative interaction
  DGN-style random half-space dendritic gating

Gate 0b / shared medium
  explicit pairwise router
  low-rank factorized router
  random orthogonal basis
  matched-smoothness basis
  score accuracy AND wiring/description/communication cost

Gate 1
  GRU / SSM / matched-state recurrent layer
  DGN instantaneous gates (tests whether persistence actually matters)

Gate 2
  BPTT / e-prop / canonical three-factor rules
  canonical Tsodyks-Markram contact dynamics where applicable

Gate 3
  broadcast scalar/vector modulation
  learned message passing
  fixed diffusion with matched spectrum/state count

Gate 4
  DGN random half-space gating
  random geometric / kNN graph
  Poirazi-Mel-style random formation + activity-dependent stabilization
  learned sparse graph/router

Gate 5
  fixed sparse network
  DGN-style continual learner
  standard continual-learning baseline
  developmental system with growth disabled after initialization
```

The repo therefore treats every ingredient as guilty until it earns its place.
