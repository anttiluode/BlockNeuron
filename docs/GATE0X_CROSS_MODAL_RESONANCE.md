# Gate 0X — concept/percept resonance

## Core idea

Do not assign a concept to one phase angle.

```text
bear != phi = pi/2
```

Instead, treat a concept/percept as a **distributed basin or resonance manifold** in the joint state of several subsystems.

A high-level semantic/query subsystem and a visual subsystem are trained on paired experience. The important relation is not that both map to one static coordinate, but that their joint dynamics become mutually compatible.

The CLIP-like static analogue is

```text
similarity(c, v) = c^T v
```

for concept/text state `c` and visual state `v`.

The dynamical version is closer to

```text
(c, v, phi) -> coupled evolution -> mutually consistent state
```

where `phi` is now a vector of local phases, not one global angle.

## Neuroscience-inspired decomposition

Do not assume that the frontal cortex literally stores a complete `bear waveform` and transmits it to visual cortex.

A more defensible abstraction is:

```text
executive / query state
        |
        v
phase / gain / routing bias
        |
        v
content-bearing sensory + semantic subnetworks
```

Visual imagery literature supports top-down access to content-specific representations in visual cortex, while recent visual-working-memory work reports content-selective alpha synchronization in visual areas alongside a more content-general frontoparietal alpha network.

This motivates separating **control** from **content**.

The controller asks for a compatible state. The distributed substrate supplies the content.

## Resonance energy

A toy joint objective can be written

```text
E(c, v, Phi)
  = E_sem(c; q)
  + E_vis(v; x)
  - sum_k lambda_k a_k(c) b_k(v) cos(Delta Phi_k)
  + regularization.
```

Here:

```text
q          high-level query / concept cue
x          bottom-up visual evidence; may be absent for imagery
c          semantic state
v          visual state
Phi        vector of local phases
```

The cross-modal term says that semantic and visual features that belong together are easier to maintain when their local dynamical states become coherent.

A minimal continuous system could use

```text
dc/dt       = -dE/dc
dv/dt       = -dE/dv
dphi_k/dt   = omega_k - eta dE/dphi_k
              + sum_j K_kj sin(phi_j - phi_k)
```

This is only a research instrument. It should not be presented as a model of literal cortical energy minimization.

## Four operating cases

### 1. Recognition

Bottom-up visual evidence is present.

```text
webcam image
    -> visual state
    -> candidate semantic modes become compatible
    -> one distributed coalition stabilizes
    -> compact public readout: identity / object / relation
```

The percept is not a permanently occupied memory slot. It is a metastable joint state supported by the current sensory evidence.

### 2. Imagery

Bottom-up visual evidence is absent or weak.

```text
query: "bear"
    -> semantic state
    -> top-down coupling changes gain / phase / routing
    -> visual dynamics move into a bear-compatible basin
    -> sensory-like representation appears
```

This is the dynamic analogue of mapping text and images into related latent spaces, except the relation is realized through state evolution rather than only a cosine similarity.

### 3. Search / attention

Both query and sensory evidence are present.

```text
query: "find bear"
    + scene
    -> search dynamics
    -> compatible visual region/subsystem locks more strongly
    -> irrelevant coalitions lose gain / coherence
```

### 4. Held percept / object file

Once a coalition is stable, the full sensory machinery need not be broadcast to every other subsystem.

A low-dimensional public consequence can persist while the detailed visual microstate remains local:

```text
private visual state
      |
      v
public latent: "that person / object here"
      |
      +--> action
      +--> language
      +--> memory
      +--> prediction
```

This is compatible with the broader project principle that receivers need consequences relevant to them, not a copy of the sender's full internal state.

## The key geometrical change from Janus

Janus has one externally controlled phase coordinate.

```text
phi in S^1
```

A realistic BlockNeuron system should instead have many coupled phases

```text
Phi = (phi_1, ..., phi_K) in T^K
```

and a concept is not one coordinate on that torus.

A concept corresponds to a **region / trajectory / attractor family** in the joint space

```text
(c, v, d, z, Phi).
```

So

```text
bear != 90 degrees
```

but rather

```text
bear ~ a basin in which several semantic, visual, dendritic and phase relations become self-consistent.
```

That naturally allows overlap. Related concepts can share large parts of the same basin geometry and diverge only in some local modes.

## Why Janus bleed becomes interesting

If phase slots bleed, the intermediate states can be interpreted as states where multiple learned constraints are partially active.

That does not mean the intermediate images are semantically meaningful. Janus was not trained to make them meaningful.

But it suggests a useful mechanism:

```text
partial compatibility with A
        -> mixed effective operator
        -> partial compatibility with B
```

A controller that can alter its own phase trajectory could use those transitional states as **roads between realizations** rather than treating them as storage corruption.

## Gate 0X protocol

The first executable version should remain deliberately synthetic.

### Data

Create paired semantic and visual patterns with independently generated nuisance structure.

```text
concept i  <->  image family i
```

The model should see many visual variants per concept so that direct instance memorization is not enough.

### Systems

`STATIC-ALIGN`
: CLIP-like contrastive/static embedding baseline.

`DIRECT-RETRIEVER`
: ordinary MLP / low-rank map from semantic cue to visual latent and vice versa.

`RESONANT-BLOCK`
: two persistent state populations coupled only through phase/gain-modulated low-rank receptors.

`GRU-COUPLER`
: matched-state generic recurrent attacker.

`HOPFIELD`
: modern associative-memory attacker.

### Tests

1. **Image -> concept retrieval**
2. **Concept -> visual-family completion**
3. **Partial/noisy image + concept query -> disambiguation**
4. **Mismatched concept/image -> failure to lock or higher residual**
5. **Cue removed after lock -> persistence / decay curve**
6. **Phase coupling shuffled -> mechanism ablation**
7. **Cross-modal pairing shuffled -> destroys learned resonance**

### Hard anti-cheat

Do not give the controller a concept index or target phase.

The only available information should be the actual distributed semantic state, visual state, local compatibility/error and local phase state.

### Metrics

```text
retrieval accuracy
completion error
lock time
state bytes
trainable parameters
executed updates / touched state
energy proxy or control effort
robustness to nuisance variation
persistence after cue removal
cross-talk between related concepts
```

## What would count as something new/useful?

Not:

```text
oscillators can associate images and concepts
```

That is already occupied by associative memory, predictive-processing, multimodal-learning and oscillatory-network literatures.

The interesting result would be one of:

```text
1. autonomous phase/gain control finds useful cross-modal states
   with less explicit routing than a direct controller;

2. one structured substrate supports many overlapping concept/percept
   coalitions with lower interference than a matched recurrent baseline;

3. after repeated use, slow structural adaptation reduces lock/search time;

4. a stabilized percept can collapse into a cheap public latent while
   the detailed private subsystem remains available for re-expansion;

5. related concepts reuse internal morphology/modes instead of requiring
   separate stored routes.
```

## Current hypothesis

The most useful synthesis is:

> **A concept is not sent to sensory cortex as a complete picture. A high-level query changes the dynamical conditions under which distributed sensory and semantic states interact. If a compatible coalition exists, the system can entrain into it; perception, imagery and search then differ mainly in where the evidence driving that coalition originates.**

For BlockNeuron, the engineering question is whether this can be implemented as a small local primitive whose structure, state and rhythm do real reusable work rather than merely reparameterizing a conventional recurrent net.
