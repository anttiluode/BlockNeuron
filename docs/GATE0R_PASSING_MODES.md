# Gate 0R — phase is an operator orbit, not a clean address

## Why this gate exists

The first Gate 0 used phase as a selector between approximately orthogonal effective weights. That is useful as an instrument, but it invites the wrong mental model:

```text
phase 0      -> memory / machine A
phase pi/2   -> memory / machine B
```

The Janus experiments already warned against that interpretation. Phase slots bleed into one another and the intermediate states are often the interesting part.

The stronger hypothesis is:

> **An oscillation can make one fixed substrate pass through a sequence of overlapping computational modes. Phase is then closer to a program counter than a clean memory address.**

The instantaneous machine at phase `phi` can be written

```text
W_eff(phi)
```

but a stateful system does not compute with one `W_eff`. Over a cycle it computes an ordered composition

```text
h_{k+1} = F(h_k; W_eff(phi_k))
```

and therefore, in the linear case,

```text
h_T = W_eff(phi_T) ... W_eff(phi_2) W_eff(phi_1) h_0.
```

If the operators do not commute, traversing the same phase set in another order gives another computation.

In continuous time the corresponding object is a periodically time-varying dynamical system

```text
dh/dt = F(h, x; phi(t)),    dphi/dt = omega
```

with a phase-dependent local linearization `J(phi)`. The whole-cycle map is the time-ordered exponential / monodromy operator

```text
Phi(T) = Texp( integral_0^T J(phi(t)) dt ).
```

This is the Floquet-style view that matters for BlockNeuron: **the computation is the ordered orbit through temporary modes, not any one mode in isolation.**

## Biological precedent

This principle is not new neuroscience.

- Hasselmo and colleagues proposed separate theta phases for encoding and retrieval in hippocampal circuits.
- Siegle & Wilson (eLife, 2014) causally manipulated hippocampal activity at different theta phases and found phase-specific effects consistent with different encoding/retrieval regimes.
- Colgin & Wilson, *Phase organization of network computations* (Current Opinion in Neurobiology, 2015), explicitly frame individual oscillatory cycles as functional units and note that distinct phases can organize distinct computations and information.
- Communication-through-coherence work likewise treats relative phase as a control on effective information transfer.

So `rhythm -> passing functional regime` is established biological motivation. The engineering question is whether a compact artificial unit can exploit it under strong ordinary controls.

## Connection to dendritic morphology

Aizenbud et al., *Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons* (PNAS, 2026; DOI `10.1073/pnas.2533168123`) strengthens a different side of the block.

Their Functional Complexity Index asks how difficult it is for a fixed DNN surrogate to reproduce the millisecond-scale I/O mapping of a detailed biophysical neuron. Human cortical pyramidal neuron models are harder to emulate than rat models. The paper attributes the difference primarily to morphology and further amplification by nonlinear NMDA-mediated integration.

Especially relevant observations:

- large dendritic extensions can create semi-independent computational subregions;
- morphology alone preserves a significant human/rat complexity difference when synapse type is controlled;
- total dendritic area is the strongest single morphological predictor reported (`R^2 = 0.74`);
- branching geometry matters jointly with area; branch count alone is much weaker (`R^2 = 0.29`);
- stronger nonlinear NMDA integration further increases functional complexity.

This is evidence for **structured matter contributing computation inside one neuron**. It does *not* show oscillatory mode switching, local learning, or an AI resource advantage. But it argues against reducing the block to "four scalar branches": morphology plus nonlinear local integration can hide a substantial spatiotemporal computation inside what a point-neuron abstraction calls one unit.

The BlockNeuron hypothesis can therefore be sharpened to:

> **A compact artificial unit may be a reusable structured microcircuit whose slow morphology stores possibilities while fast state and rhythm move the unit through temporary computational regimes.**

## Frozen toy

File: `experiments/gate0r_passing_modes.py`

Four non-commuting 2x2 primitive transforms define two programs:

```text
FORWARD = [0, 1, 2, 3, 0]
REVERSE = [0, 3, 2, 1, 0]
```

Both programs:

- see the same input distribution;
- use exactly the same multiset of phase anchors;
- start at phase zero;
- end at phase zero.

They differ only in the order in which the internal state passes through the modes.

### Passing-mode block

The block owns four trainable 2x2 branch operators. Phase selects them with deliberately **overlapping** von-Mises-like windows:

```text
g_b(phi) = softmax_b( kappa cos(phi - psi_b) )
W_eff(phi) = sum_b g_b(phi) W_b
```

with `kappa=3`.

At a phase anchor the gate is approximately

```text
[0.9074, 0.0452, 0.0022, 0.0452]
```

up to rotation.

So this gate explicitly rejects clean phase addressing. Neighboring modes are present at every anchor.

### Matched attacker

`FOURIER RNN ATTACKER` directly generates the recurrent step matrix from

```text
[1, cos(phi), sin(phi), cos(2 phi)]
```

and has exactly the same **16 trainable parameters**.

This control can represent arbitrary matrices at the four phase anchors. If it wins or ties, the result is ordinary periodically conditioned recurrence rather than a branch-specific advantage.

### Endpoint-only lower control

Both programs finish at the same phase. A memoryless endpoint model cannot infer which path was traversed. Its least-squares optimum is the average of the two target maps.

This control is not a serious architecture attacker; it only verifies that path history is genuinely required by the toy.

## First local receipt

Seed `18001`, 500 optimization steps:

```text
teacher forward/reverse matrix gap : 0.631583
phase-window mean max gate         : 0.907398
adjacent bleed at phase 0          : 0.045177

model                 params       fit_mse      wrong_order    phase_clamp
PASSING-MODE BLOCK        16    ~2.5e-14        ~0.193          ~1.23
FOURIER RNN ATTACKER      16    ~5.3e-14        ~0.195         large
ENDPOINT-ONLY              4     ~0.0495             -              -
```

Five checked seeds all passed the same mechanism screen.

## Verdict

**PASS as a structural/dynamical instrument.**

The toy demonstrates all of the following:

1. phase windows need not be clean or orthogonal;
2. overlapping transient modes can still support precise computation;
3. the same phase set can implement different mappings purely through traversal order;
4. clamping the rhythm removes the passing-mode computation;
5. a generic matched recurrent phase-conditioned attacker solves the same task.

Therefore do **not** claim that oscillatory mode traversal has unique expressivity.

The supported statement is:

> **A fixed recurrent substrate can use a rhythmic trajectory as an ordered schedule of overlapping effective operators. In a stateful system, phase can organize computation through the path taken around the cycle rather than act as a discrete address.**

## What would make this scientifically interesting?

The next tests have to remove conveniences from this toy.

### 1. One slow mass, not four free matrices

Replace four independent branch matrices by one slow structural matrix plus local phase-dependent diagonal/receptor gains, for example

```text
W_eff(phi) = D_out(phi) W D_in(phi).
```

Then ask how many useful cycle operators can be realized per stored structural parameter.

### 2. Autonomous phase

Do not supply a list of phase labels. Give the unit an oscillator

```text
dphi/dt = omega + coupling + modulation
```

and let its own dynamics carry the state through the modes.

### 3. Persistent dendritic state

Add branch-local state so that an early phase can modify what a later phase sees:

```text
tau_d dd_b/dt = -d_b + F(input_b, local contact state, modulation).
```

This is the real bridge to Gate 1.

### 4. Strong recurrent attackers

Compare against GRU/SSM/tied-weight RNNs and Fourier/time-conditioned recurrent models at matched:

- trainable parameters;
- persistent state bytes;
- update FLOPs / memory traffic;
- sequence latency.

### 5. Resource question

The interesting endpoint is not "rhythm can compute." It is:

> **Can autonomous rhythmic reuse of structured local matter obtain useful effective depth/routing with less explicitly stored machinery or less per-event work than the ordinary recurrent alternative?**

That is the version consistent with both the historical literature and the rest of this repo family.
