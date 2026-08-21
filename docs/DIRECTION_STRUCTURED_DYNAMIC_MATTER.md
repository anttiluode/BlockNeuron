# Direction — structured dynamic matter, without a false either/or

The project should not be framed as

```text
structured matter  VS  explicitly stored computation
```

Both are useful. The stronger working picture is a hierarchy in which explicit parameters configure a structured dynamical substrate, and the substrate supplies transformations that do not have to be re-described as a fresh matrix at every instant.

## Three places computation can live

### 1. Explicit slow parameters

Examples:

```text
synaptic strengths
branch identities / receptors
morphological parameters
oscillator frequencies
small developmental rules
readout weights
```

These are ordinary stored information. BlockNeuron is not trying to eliminate weights.

### 2. Implicit structural computation

A fixed morphology has transfer functions even before a task-specific controller is evaluated.

A reduced dendritic tree can be represented by a small cable / graph operator `L_M`:

```text
morphology M  ->  L_M
```

Lengths, branching, capacitance, attenuation and nonlinear local integration determine how signals propagate and combine. Aizenbud et al. (PNAS 2026) and Beniaguev et al. (Neuron 2021) are important here: detailed dendritic morphology plus nonlinear synaptic integration makes the I/O map of one biological neuron substantially more difficult for an ordinary neural-network surrogate to reproduce.

This is computation embodied in structure, but not computation without parameters or state.

### 3. Transient dynamical computation

Local state and rhythmic phase determine what that structure is doing now.

```text
contact state z_e(t)
compartment state d_b(t)
phase phi(t)
modulator m(t)
```

The current effective operator can therefore differ from the slow anatomical operator.

The key new phrase for the repo is:

> **passing computational modes**

These modes do not have to be clean, orthogonal, or separately addressable. They can overlap strongly. What matters is that the same stateful substrate passes through a reproducible sequence of effective transformations.

## Minimal microtree equation

A better descendant of the current four-scalar-branch instrument is a tiny internal tree of compartments.

Let `d(t)` be the vector of compartment states and `L_M` a morphology-induced cable/Laplacian operator. A deliberately reduced model could be

```text
tau_d dd/dt = -d - kappa L_M d
              + N(B u(t) + C(phi(t), m(t)) d + b)
```

where

```text
L_M                  slow morphology / cable structure
N                    local nonlinear integration
B                    input landing map
C(phi,m)             phase/modulator-dependent effective recurrent conductance
```

A restricted version of the changing conductance could be

```text
C(phi,m) = D_out(phi,m) W D_in(phi,m)
```

rather than a separately stored matrix for every phase.

The local linearization is then

```text
J(phi,m,d) = dF/dd.
```

As phase moves, `J` moves. Its instantaneous eigenvectors/effective pathways are the passing modes. They need not remain orthogonal; mode mixing can be part of the computation.

Over a complete periodic cycle, the important object is not one `J(phi)` but the ordered whole-cycle map. In the linearized continuous-time case this is the monodromy / time-ordered exponential

```text
Phi(T) = Texp( integral_0^T J(phi(t)) dt ).
```

That gives a precise version of the intuition:

> **rhythm can turn one structured unit into a sequence of temporary computational regimes, and state composes those regimes into a whole-cycle computation.**

## Why this is not new by itself

Several established literatures cover large pieces of this picture.

### Phase-organized neural computation

Hippocampal work has long proposed distinct encoding and retrieval regimes at different theta phases. Siegle & Wilson (2014) provided phase-specific causal manipulation evidence; Colgin & Wilson (2015) explicitly describe oscillatory cycles as functional units in which distinct phases organize distinct information/computations.

### Dynamic connectivity through oscillations

Hoppensteadt & Izhikevich (1999) showed that a common forced medium can impose temporary effective connectivity between oscillators despite homogeneous physical coupling.

### Physical reservoir computing

Physical reservoir computing deliberately exploits the natural transient dynamics and fading memory of physical systems. The reservoir can remain fixed while only a readout is trained.

Especially relevant is **single-node time-multiplexed reservoir computing**: one nonlinear physical node plus delay/transient dynamics is sampled at many sub-times and treated as many virtual nodes. Thus

```text
one physical thing
    -> many useful temporal states
    -> virtual computational population
```

is already a mature idea, including optical, electronic, mechanical and spintronic implementations.

### Physical neural networks

Physical neural networks go further and train controllable physical transformations directly. Therefore the general thesis `matter can perform transformations instead of digitally simulating every primitive` is established territory.

## What remains interesting for BlockNeuron

The target combination is narrower:

```text
small trainable morphology / internal graph
+ local persistent compartments
+ tiny per-contact state
+ autonomous phase
+ phase/modulator-dependent effective conductance
+ local eligibility / slow consolidation
+ sparse external connectivity / growth
```

The unit is intended as a **software/hardware-neutral computational abstraction**. It should be useful even before any claim about a particular physical substrate.

The question is not whether its dynamics are expressive. Ordinary RNNs, neural ODEs, hypernetworks and reservoirs will usually emulate them.

The useful questions are:

```text
Does the structure give a better inductive bias?
Does it reduce stored routing / parameter description?
Does it reduce executed work or memory traffic?
Does locality make learning signals cheaper?
Does repeated experience consolidate future computation into structure?
Can one small structural object support many useful transient regimes?
```

## Aizenbud changes the artificial block

The 2026 Aizenbud paper argues against representing morphology by branch count alone.

Their reported morphology/complexity relationships include:

```text
total dendritic area                    R^2 = 0.74
sum of bifurcation-branch length        R^2 = 0.45
longest bifurcation branch              R^2 = 0.44
number of bifurcations                  R^2 = 0.29
```

and combinations involving total area plus branch geometry explain substantially more variance.

The artificial interpretation is not `copy human dendrites`. It is:

> **internal geometry should determine signal transfer and state timescales, not merely label a set of independent scalar experts.**

That suggests moving from four scalar branches toward a tiny tree/cable graph before investing heavily in growth or chemistry.

## Proposed Gate 1M — microtree before biology pile-up

### Unit

Start with seven compartments:

```text
          soma
         /    \
       b0      b1
      /  \    /  \
     l0  l1  l2  l3
```

Each tree edge has a fixed or low-parameter coupling/time constant. Inputs land on leaves. Compartment state persists. Soma reads the root.

### Task family

Use branch-local temporal relations rather than generic sequence classification.

Examples:

```text
A then B on sister leaves within window -> class 1
same events on different subtree        -> class 0
B then A                                -> class 0
```

Then add a phase-organized write/read version:

```text
first half-cycle   accumulate / integrate locally
second half-cycle  propagate / read out
```

The point is not biological realism. The task forces geometry, state and ordered phase to have distinct roles that can each be ablated.

### Mandatory controls

```text
small GRU at matched persistent-state bytes
SSM / tied-weight RNN
instantaneous DGN-style gates
same microtree with phase clamped
same oscillator with compartments fully mixed
input-to-leaf assignment shuffled
random tree with matched degree/spectrum
single-node time-multiplexed reservoir
Fourier/time-conditioned recurrent attacker
```

### Metrics

Do not score accuracy alone.

```text
test error
sample efficiency
trainable parameters
persistent state bytes
executed multiply-adds / touched state
routing / communication events
latency per sequence
robustness to timing jitter
recovery after task/context shift
```

### Stop lines

Kill or demote an ingredient when:

- a dense matched-state RNN learns the same task with equal data and lower work;
- random morphology performs as well as task-aligned morphology;
- phase clamping does not hurt after parameter matching;
- the only advantage comes from giving BlockNeuron more hidden state;
- a single-node time-multiplexed reservoir provides the same effect more cheaply;
- the internal tree is only a verbose reparameterization of a small dense matrix.

## Current synthesis

The project is converging on a hybrid statement:

> **Explicit weights store slow possibilities. Structure makes some transformations implicit. Local state stores what just happened. Rhythm moves the same structure through passing computational modes. Learning decides which of those temporary regimes should become easier to enter in the future.**

That is a stronger target than either `physics replaces weights` or `a richer artificial neuron is automatically better`.
